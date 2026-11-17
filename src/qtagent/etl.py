"""数据加工（ETL）：抓取 → 清洗 → 落库 → 导出离线快照。

清洗规则（写进代码，避免「数据脏了不知道为什么」）：
- 行情：去重、按日期排序、剔除价格 ≤ 0 的记录、缺失 OHLC 的行直接丢弃
- 海关：金额 ≤ 0 视为无效月；同比/环比保留缺失（不填 0，否则会污染 z-score）
- 新闻：正文与标题拼接后打分，去重后入库；情感分与标签一并落库，便于复查
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config
from .datasource import build_client
from .lexicon import label_of, score_text
from .storage import get_engine, init_schema, upsert_df
from .storage.db import count_rows
from .storage.schema import dim_agent, dim_region, fact_fx_daily, fact_market_daily, fact_news, fact_trade_monthly

logger = logging.getLogger(__name__)

AGENT_META = [
    ("collector", "数据采集 Agent", "抓取行情/海关/新闻，校验完整性并产出统一特征", 0.10),
    ("technical", "技术分析 Agent", "均线/MACD/RSI/布林/量价六组规则投票", 0.40),
    ("sentiment", "新闻情绪 Agent", "财经快讯情感分按时间衰减聚合，缺失时降级为量价代理", 0.15),
    ("macro", "宏观贸易 Agent", "海关进出口同比与贸易差额的 12 个月 z-score 传导", 0.20),
    ("risk", "风险管理 Agent", "波动率目标 + VaR + 回撤约束给出仓位上限", 0.25),
    ("decision", "交易决策 Agent", "加权规则合成 + 确认机制 + 调仓带，输出三分类信号", 0.00),
]

REGIONS = [
    ("TOTAL", "全部贸易伙伴", "total", "海关公布的中国进出口总额"),
    ("US", "美国", "country", "中美双边数据接入位（当前数据源未提供，预留）"),
    ("EU", "欧盟", "bloc", "中欧双边数据接入位（预留）"),
    ("ASEAN", "东盟", "bloc", "中国-东盟双边数据接入位（预留）"),
]


# ---------------- 清洗 ----------------
def clean_market(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["trade_date", "open", "high", "low", "close"])
    out = out[(out["close"] > 0) & (out["low"] > 0)]
    out = out.drop_duplicates(subset=["index_code", "trade_date"]).sort_values("trade_date")
    out["source"] = source
    return out.reset_index(drop=True)


def clean_trade(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=["trade_month"])
    out = out[(out["export_usd"] > 0) | (out["import_usd"] > 0)]
    out = out.drop_duplicates(subset=["trade_month"]).sort_values("trade_month")
    out["region_code"] = "TOTAL"
    out["source"] = source
    return out.reset_index(drop=True)


def score_news(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    text = (out["title"].fillna("") + " " + out["summary"].fillna("")).str.strip()
    out["sentiment_score"] = text.map(score_text).round(6)
    out["sentiment_label"] = out["sentiment_score"].map(label_of)
    return out


# ---------------- 主流程 ----------------
def build_db(cfg: Config, export_snapshot: bool = True) -> dict[str, Any]:
    provider = cfg.data.get("provider", "eastmoney")
    client = build_client(cfg)
    engine = get_engine(cfg)
    init_schema(engine)

    index_code = cfg.data.get("index_code", "000300")
    start = cfg.data.get("start", "2019-01-01")
    end = cfg.data.get("end") or "20500101"

    benchmark_code = cfg.data.get("benchmark_code", "000300")
    codes = [index_code] if index_code == benchmark_code else [index_code, benchmark_code]
    logger.info("拉取行情日线 %s ...", codes)
    market = pd.concat(
        [clean_market(client.index_daily(code, start, end), provider) for code in codes],
        ignore_index=True,
    )
    market = market.sort_values(["index_code", "trade_date"]).reset_index(drop=True)

    logger.info("拉取海关月度进出口 ...")
    trade = clean_trade(client.customs_monthly(), provider)

    logger.info("拉取汇率 ...")
    fx = client.fx_daily("USDCNH", start)
    fx["source"] = provider

    logger.info("拉取财经快讯 ...")
    news = score_news(client.fast_news(pages=int(cfg.data.get("news_pages", 40))))

    counts = {
        "market": upsert_df(engine, fact_market_daily, market, ["index_code", "trade_date"]),
        "trade": upsert_df(engine, fact_trade_monthly, trade, ["trade_month", "region_code"]),
        "fx": upsert_df(engine, fact_fx_daily, fx, ["pair", "trade_date"]),
        "news": upsert_df(engine, fact_news, news, ["news_id"]),
    }

    # 维度表种子数据（幂等）
    upsert_df(
        engine,
        dim_agent,
        pd.DataFrame(AGENT_META, columns=["agent_code", "agent_name", "responsibility", "weight"]),
        ["agent_code"],
    )
    upsert_df(
        engine,
        dim_region,
        pd.DataFrame(REGIONS, columns=["region_code", "region_name", "region_type", "note"]),
        ["region_code"],
    )

    if export_snapshot:
        export_snapshot_files(cfg, market, trade, fx, news)

    summary = {
        **counts,
        "db_url": cfg.db_url,
        "market_range": {code: f"{g['trade_date'].min()} ~ {g['trade_date'].max()}"
                         for code, g in market.groupby("index_code")},
        "trade_range": f"{trade['trade_month'].min()} ~ {trade['trade_month'].max()}",
        "news_range": (
            f"{news['publish_time'].min()} ~ {news['publish_time'].max()}" if not news.empty else "空"
        ),
        "tables": {t: count_rows(engine, t) for t in
                   ["fact_market_daily", "fact_trade_monthly", "fact_fx_daily", "fact_news"]},
    }
    return summary


def export_snapshot_files(
    cfg: Config,
    market: pd.DataFrame,
    trade: pd.DataFrame,
    fx: pd.DataFrame,
    news: pd.DataFrame,
) -> Path:
    out = cfg.path("data/snapshot")
    out.mkdir(parents=True, exist_ok=True)
    market.to_csv(out / "market_daily.csv", index=False, encoding="utf-8")
    trade.to_csv(out / "trade_monthly.csv", index=False, encoding="utf-8")
    fx.to_csv(out / "fx_daily.csv", index=False, encoding="utf-8")
    if not news.empty:
        n = news.copy()
        n["publish_time"] = pd.to_datetime(n["publish_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        n.to_csv(out / "news.csv", index=False, encoding="utf-8")
    logger.info("离线快照已导出到 %s", out)
    return out
