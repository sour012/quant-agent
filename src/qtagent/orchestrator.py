"""编排层：串起五个 Agent，并明确数据流转顺序与各 Agent 的输入边界。

流转顺序（每一步只读上游已产出的东西，保证不互相干扰）：
    Collector  → 特征行 + 数据质量
    Technical / Sentiment / Macro  → 并行（互不依赖，各自只看自己那份特征）
    Risk       → 只看行情与组合状态，产出仓位上限
    Decision   → 汇总四个投票者 + 上下文记忆，输出动作与归因
"""

from __future__ import annotations

import bisect
import json
import logging
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .agents import (
    AgentContext,
    Decision,
    DecisionAgent,
    PortfolioState,
    Signal,
    build_agents,
)
from .config import Config
from .features import build_feature_frame
from .storage import get_engine, load_market, load_news, load_trade

logger = logging.getLogger(__name__)

PIPELINE_ORDER = ("collector", "technical", "sentiment", "macro", "risk")


class TradingPipeline:
    WARMUP_BARS = 60

    def __init__(
        self,
        cfg: Config,
        market: pd.DataFrame | None = None,
        trade: pd.DataFrame | None = None,
        news: pd.DataFrame | None = None,
        benchmark: pd.DataFrame | None = None,
    ) -> None:
        self.cfg = cfg
        if market is None or trade is None or news is None:
            engine = get_engine(cfg)
            index_code = cfg.data.get("index_code", "000300")
            market = load_market(engine, index_code)
            trade = load_trade(engine)
            news = load_news(engine)
        if market is None or market.empty:
            raise RuntimeError("行情数据为空，请先执行 `python -m qtagent build-db`")

        if benchmark is None and market is not None:
            bench_code = cfg.data.get("benchmark_code")
            if bench_code and bench_code != cfg.data.get("index_code"):
                try:
                    benchmark = load_market(get_engine(cfg), bench_code)
                except Exception:  # noqa: BLE001 - 基准缺失不阻塞主流程
                    benchmark = None
        self.benchmark = benchmark if benchmark is not None else pd.DataFrame()
        if not self.benchmark.empty:
            self.benchmark = self.benchmark.copy()
            self.benchmark["trade_date"] = pd.to_datetime(self.benchmark["trade_date"])

        self.market = market.copy()
        self.market["trade_date"] = pd.to_datetime(self.market["trade_date"])
        self.market = self.market.sort_values("trade_date").reset_index(drop=True)
        self.trade = trade if trade is not None else pd.DataFrame()
        self.news = news if news is not None else pd.DataFrame()

        self.features = build_feature_frame(self.market, self.trade, self.news, cfg)
        self.dates: list[pd.Timestamp] = list(self.features.index)

        self.agents = build_agents(cfg)
        self.decision_agent = DecisionAgent(cfg)

        # 新闻按时间预排序，用二分查找切片，避免每日全表扫描
        if not self.news.empty:
            n = self.news.copy()
            n["publish_time"] = pd.to_datetime(n["publish_time"], errors="coerce")
            n = n.dropna(subset=["publish_time"]).sort_values("publish_time").reset_index(drop=True)
            self.news = n
            self._news_times = n["publish_time"].to_numpy()
        else:
            self._news_times = np.array([], dtype="datetime64[ns]")

    # ---------------- 对外接口 ----------------
    def reset(self) -> None:
        self.decision_agent.reset()

    def index_of(self, date: str | pd.Timestamp) -> int:
        ts = pd.Timestamp(date).normalize()
        idx = bisect.bisect_left(self.dates, ts)
        if idx >= len(self.dates) or self.dates[idx] != ts:
            # 允许传入非交易日：取不超过该日期的最近交易日
            idx = max(0, idx - 1)
        return idx

    def context(self, idx: int, portfolio: PortfolioState | None = None) -> AgentContext:
        date = self.dates[idx]
        row = self.features.iloc[idx].to_dict()
        row["close"] = float(self.market["close"].iloc[idx])

        n = int(np.searchsorted(self._news_times, np.datetime64(date), side="left"))
        news_slice = self.news.iloc[:n] if not self.news.empty else self.news
        trade_slice = (
            self.trade[self.trade["trade_month"] <= date.strftime("%Y-%m")]
            if not self.trade.empty
            else self.trade
        )
        return AgentContext(
            trade_date=date.strftime("%Y-%m-%d"),
            market=self.market.iloc[: idx + 1],
            trade=trade_slice,
            news=news_slice,
            features=row,
            portfolio=portfolio or PortfolioState(),
            cfg=self.cfg,
        )

    def step(
        self, idx: int, portfolio: PortfolioState | None = None
    ) -> tuple[Decision, dict[str, Signal]]:
        ctx = self.context(idx, portfolio)
        signals: dict[str, Signal] = {}
        for name in PIPELINE_ORDER:
            sig = self.agents[name].run(ctx)
            signals[name] = sig
            ctx.peer_signals[name] = sig  # 下游可以读上游结论，反之不行
        decision = self.decision_agent.run(ctx)
        return decision, signals

    def explain(self, date: str | pd.Timestamp, portfolio: PortfolioState | None = None) -> dict[str, Any]:
        idx = self.index_of(date)
        decision, signals = self.step(idx, portfolio)
        return {
            "trade_date": decision.trade_date,
            "decision": {
                "action": decision.action,
                "score": round(decision.score, 4),
                "confidence": round(decision.confidence, 4),
                "target_position": decision.target_position,
                "position_cap": decision.position_cap,
                "contributions": decision.contributions,
                "reasons": decision.reasons,
            },
            "agents": {
                name: {
                    "title": sig.agent if hasattr(sig, "agent") else name,
                    "score": round(sig.score, 4),
                    "confidence": round(sig.confidence, 4),
                    "reasons": sig.reasons,
                    "evidence": _jsonable(sig.evidence),
                }
                for name, sig in signals.items()
            },
        }

    def run_range(
        self,
        start: str | None = None,
        end: str | None = None,
        portfolio: PortfolioState | None = None,
    ) -> Iterable[tuple[pd.Timestamp, Decision, dict[str, Signal]]]:
        self.reset()
        begin = self.index_of(start) if start else 0
        stop = self.index_of(end) if end else len(self.dates) - 1
        for idx in range(begin, stop + 1):
            if idx < self.WARMUP_BARS:
                continue
            decision, signals = self.step(idx, portfolio)
            yield self.dates[idx], decision, signals


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def dump_explain(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
