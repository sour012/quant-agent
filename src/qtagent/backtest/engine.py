"""回测引擎（T+1 执行、含交易成本）。

执行时序（关键，决定结果是否可信）：
    T 日收盘后 → 五 Agent 出信号 → 得出目标仓位
    T+1 日开盘 → 按开盘价 + 滑点成交，佣金按成交金额计提
因此信号永远不包含 T+1 的价格信息，不存在未来函数。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..config import Config
from ..orchestrator import TradingPipeline
from ..agents import PortfolioState
from .metrics import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)


class BacktestEngine:
    def __init__(
        self,
        pipeline: TradingPipeline,
        cfg: Config,
        start: str | None = None,
        end: str | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.cfg = cfg
        bt = cfg.backtest
        self.initial_cash = float(bt.get("initial_cash", 1_000_000))
        self.commission = float(bt.get("commission", 0.0003))
        self.slippage = float(bt.get("slippage", 0.0005))
        self.rf = float(bt.get("risk_free_rate", 0.02))
        self.lot = int(bt.get("lot_size", 100))
        self.start = start
        self.end = end

    # ---------------- 主循环 ----------------
    def run(self) -> BacktestResult:
        p = self.pipeline
        p.reset()
        market = p.market
        dates = p.dates

        begin = max(p.WARMUP_BARS, p.index_of(self.start) if self.start else 0)
        stop = p.index_of(self.end) if self.end else len(dates) - 1
        if stop <= begin:
            raise ValueError("回测区间过短，无法产生有效信号（需要至少 60 根预热 K 线）")

        cash = self.initial_cash
        shares = 0
        cost_basis = 0.0          # 当前持仓累计买入成本
        entry_cost = 0.0          # 本轮持仓累计买入金额
        open_sells = 0.0          # 本轮累计卖出回款
        entry_date: str | None = None
        entry_idx = 0
        entry_px = 0.0
        pending_target: float | None = None
        last_action = "HOLD"

        records: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []

        for idx in range(begin, stop + 1):
            date = dates[idx]
            date_str = date.strftime("%Y-%m-%d")
            open_px = float(market["open"].iloc[idx])
            close_px = float(market["close"].iloc[idx])

            # ---- 1) 执行上一交易日收盘后生成的目标仓位 ----
            if pending_target is not None:
                equity_open = cash + shares * open_px
                target_value = equity_open * pending_target
                target_shares = int(target_value / max(open_px, 1e-9) // self.lot * self.lot)
                delta = target_shares - shares
                if delta > 0:
                    buy_px = open_px * (1 + self.slippage)
                    cost = delta * buy_px * (1 + self.commission)
                    if cost <= cash + 1e-6:
                        cash -= cost
                        shares += delta
                        cost_basis += cost
                        entry_cost += cost
                        if entry_date is None:
                            entry_date, entry_idx = date_str, idx
                            entry_px = buy_px
                elif delta < 0:
                    sell_px = open_px * (1 - self.slippage)
                    proceeds = -delta * sell_px * (1 - self.commission)
                    cash += proceeds
                    open_sells += proceeds
                    sold = -delta
                    cost_basis -= cost_basis * (sold / shares) if shares > 0 else 0.0
                    shares += delta
                    if shares == 0 and entry_date is not None:
                        pnl = open_sells - entry_cost
                        trades.append(
                            {
                                "entry_date": entry_date,
                                "exit_date": date_str,
                                "entry_price": round(entry_px, 4),
                                "exit_price": round(sell_px, 4),
                                "pnl": round(pnl, 2),
                                "pnl_pct": round(pnl / entry_cost, 6) if entry_cost else 0.0,
                                "hold_days": idx - entry_idx,
                            }
                        )
                        entry_cost, open_sells = 0.0, 0.0
                        entry_date = None
                pending_target = None

            # ---- 2) 当日收盘估值 ----
            equity = cash + shares * close_px
            position = (shares * close_px / equity) if equity > 0 else 0.0
            pf = PortfolioState(
                equity=equity,
                cash=cash,
                position_ratio=position,
                last_action=last_action,
                days_in_position=(idx - entry_idx) if shares > 0 else 0,
            )

            # ---- 3) 收盘后出信号 ----
            decision, _signals = p.step(idx, pf)
            last_action = decision.action
            if abs(decision.target_position - position) > 1e-9:
                pending_target = decision.target_position

            decisions.append(decision.to_record())
            records.append(
                {
                    "trade_date": date_str,
                    "close": round(close_px, 4),
                    "cash": round(cash, 2),
                    "shares": shares,
                    "position": round(position, 6),
                    "equity": round(equity, 2),
                    "target": decision.target_position,
                    "action": decision.action,
                    "score": round(decision.score, 6),
                }
            )

        daily = pd.DataFrame(records)
        # 基准净值：优先用配置的基准指数（沪深300），日期对齐后前向填充
        bench_df = getattr(p, "benchmark", pd.DataFrame())
        bench_close = (
            bench_df.set_index(pd.to_datetime(bench_df["trade_date"]))["close"]
            if not bench_df.empty
            else market.assign(_d=pd.to_datetime(market["trade_date"])).set_index("_d")["close"]
        )
        aligned = bench_close.reindex(pd.to_datetime(daily["trade_date"])).ffill().bfill()
        daily["benchmark_equity"] = (self.initial_cash * aligned.to_numpy() / float(aligned.iloc[0])).round(2)
        daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
        daily["benchmark_drawdown"] = (
            daily["benchmark_equity"] / daily["benchmark_equity"].cummax() - 1.0
        )
        trades_df = pd.DataFrame(trades)
        metrics = compute_metrics(
            daily.set_index("trade_date")["equity"],
            daily.set_index("trade_date")["benchmark_equity"],
            daily.set_index("trade_date")["position"],
            rf=self.rf,
            trades=trades_df,
        )
        metrics["final_equity"] = float(daily["equity"].iloc[-1])
        metrics["start"] = daily["trade_date"].iloc[0]  # type: ignore[assignment]
        metrics["end"] = daily["trade_date"].iloc[-1]    # type: ignore[assignment]
        return BacktestResult(daily=daily, trades=trades_df, metrics=metrics, decisions=decisions)
