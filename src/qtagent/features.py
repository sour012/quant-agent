"""特征工程：把行情、贸易、新闻合成到「每个交易日一行」。

关键约束（防未来函数）：
1. 所有滚动指标只使用当日及以前的数据；
2. 海关数据存在发布滞后，日频对齐时按「交易日往前推 25 天所在月份」取最新可得月度值；
3. 新闻只使用发布时间早于当日 0 点的记录（收盘后发布的新闻计入下一交易日）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .config import Config


def build_market_features(market: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    p = cfg.agent("technical")
    df = market.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)

    close = df["close"]
    ret = close.pct_change()

    for w in p.get("ma_windows", [5, 20, 60]):
        df[f"ma{w}"] = ind.sma(close, int(w))
        df[f"bias{w}"] = close / df[f"ma{w}"] - 1.0

    dif, dea, hist = ind.macd(close, *p.get("macd", [12, 26, 9]))
    df["macd_dif"], df["macd_dea"], df["macd_hist"] = dif, dea, hist

    df["rsi"] = ind.rsi(close, int(p.get("rsi_period", 14)))

    low, mid, up = ind.bollinger(close, int(p.get("boll_window", 20)), float(p.get("boll_k", 2.0)))
    df["boll_up"], df["boll_mid"], df["boll_low"] = up, mid, low
    df["boll_pos"] = ((close - low) / (up - low)).clip(0, 1)

    if "ma20" in df.columns:
        df["ma20_slope5"] = df["ma20"].pct_change(5)  # 中期均线斜率，判断趋势是否在走强

    df["atr"] = ind.atr(df, int(p.get("atr_period", 14)))
    df["atr_pct"] = df["atr"] / close

    df["vol20"] = ind.annualized_vol(ret, 20)
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(
        int(p.get("volume_window", 20)), min_periods=5
    ).mean()

    df["ret1"] = ret
    df["ret5"] = close.pct_change(5)
    df["ret20"] = close.pct_change(20)
    df["dd_from_peak"] = close / close.cummax() - 1.0
    return df


def build_trade_features(trade: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """月度贸易序列 → 带 z-score 的月度特征表。"""
    m = cfg.agent("macro")
    df = trade.copy().sort_values("trade_month").reset_index(drop=True)
    df["balance_yoy"] = df["balance_usd"].pct_change(12) * 100

    win = int(m.get("zscore_window", 12))
    for col in ["export_yoy", "import_yoy", "balance_yoy"]:
        df[f"{col}_z"] = ind.rolling_zscore(df[col], win)
    # 3 个月动量：进出口同比的方向变化
    df["export_yoy_mom3"] = df["export_yoy"].diff(3)
    df["import_yoy_mom3"] = df["import_yoy"].diff(3)
    return df


def _daily_news_features(news: pd.DataFrame, trading_days: pd.Series, cfg: Config) -> pd.DataFrame:
    """按指数衰减聚合新闻情绪，输出每日一行。"""
    s = cfg.agent("sentiment")
    half_life = float(s.get("decay_halflife_days", 2.0))
    lookback = int(s.get("lookback_days", 5))

    empty = pd.DataFrame(
        {"news_score": 0.0, "news_count": 0, "news_pos": 0, "news_neg": 0},
        index=pd.DatetimeIndex(trading_days),
    )
    if news is None or news.empty:
        return empty

    n = news.copy()
    n["publish_time"] = pd.to_datetime(n["publish_time"], errors="coerce")
    n = n.dropna(subset=["publish_time"])
    if n.empty:
        return empty
    if "sentiment_score" not in n.columns:
        n["sentiment_score"] = 0.0
    n["day"] = n["publish_time"].dt.normalize()

    # 收盘后（>=15:00）发布的新闻顺延到下一个自然日，避免使用未来信息
    late = n["publish_time"].dt.hour >= 15
    n.loc[late, "day"] = n.loc[late, "day"] + pd.Timedelta(days=1)

    daily = n.groupby("day").agg(
        mean_score=("sentiment_score", "mean"),
        count=("sentiment_score", "size"),
        pos=("sentiment_score", lambda x: int((x > 0).sum())),
        neg=("sentiment_score", lambda x: int((x < 0).sum())),
    )

    calendar = pd.date_range(daily.index.min(), max(trading_days.max(), daily.index.max()), freq="D")
    daily = daily.reindex(calendar).fillna({"mean_score": 0.0, "count": 0, "pos": 0, "neg": 0})

    scores = daily["mean_score"].to_numpy()
    counts = daily["count"].to_numpy()
    pos = daily["pos"].to_numpy()
    neg = daily["neg"].to_numpy()
    ages = np.arange(lookback)[::-1]  # 0 = 当天
    decay = 0.5 ** (ages / half_life)

    rows = []
    day_index = {d: i for i, d in enumerate(calendar)}
    for d in pd.DatetimeIndex(trading_days):
        i = day_index.get(d.normalize())
        if i is None:
            rows.append((0.0, 0, 0, 0))
            continue
        lo = max(0, i - lookback + 1)
        window = slice(lo, i + 1)
        w = decay[-(i + 1 - lo):]
        cnt = counts[window]
        sc = scores[window] * cnt
        denom = float((cnt * w).sum())
        rows.append(
            (
                float((sc * w).sum() / denom) if denom > 0 else 0.0,
                int(cnt.sum()),
                int(pos[window].sum()),
                int(neg[window].sum()),
            )
        )

    return pd.DataFrame(
        rows, columns=["news_score", "news_count", "news_pos", "news_neg"], index=pd.DatetimeIndex(trading_days)
    )


def build_feature_frame(
    market: pd.DataFrame, trade: pd.DataFrame, news: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """行情 + 贸易 + 新闻 → 单表特征矩阵（index = 交易日）。"""
    mf = build_market_features(market, cfg)

    # ---- 贸易数据按发布滞后对齐 ----
    tf = build_trade_features(trade, cfg) if trade is not None and not trade.empty else pd.DataFrame()
    if not tf.empty:
        tf["month_dt"] = pd.to_datetime(tf["trade_month"] + "-01")
        lag_days = int(cfg.agent("macro").get("publish_lag_days", 25))
        # 交易日往前推 lag_days，所在的月份即为「当日已知」的最新月份
        align = mf[["trade_date"]].copy()
        align["known_month"] = (align["trade_date"] - pd.Timedelta(days=lag_days)).dt.to_period("M").dt.to_timestamp()
        merged = pd.merge_asof(
            align.sort_values("known_month"),
            tf.drop(columns=["trade_month"]).sort_values("month_dt"),
            left_on="known_month",
            right_on="month_dt",
            direction="backward",
        )
        trade_cols = [c for c in merged.columns if c not in {"trade_date", "known_month", "month_dt"}]
        mf = pd.concat([mf.reset_index(drop=True), merged[trade_cols].reset_index(drop=True)], axis=1)
        # 当月对齐到的贸易数据实际滞后天数（用于宏观 Agent 判断数据新鲜度）
        mf["trade_lag_days"] = (merged["known_month"] - merged["month_dt"]).dt.days
    else:
        for col in ["export_yoy", "import_yoy", "balance_yoy", "export_yoy_z", "import_yoy_z", "balance_yoy_z"]:
            mf[col] = np.nan

    # ---- 新闻情绪 ----
    news_feat = _daily_news_features(news, mf["trade_date"], cfg)
    mf = pd.concat([mf.reset_index(drop=True), news_feat.reset_index(drop=True)], axis=1)

    return mf.set_index("trade_date")
