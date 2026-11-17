"""绩效指标计算：全部基于日频净值序列，口径写死在文档里以免歧义。

- 年化收益：几何年化 CAGR
- 年化波动：日收益标准差 × √252
- 夏普： (年化收益 - 无风险) / 年化波动
- 最大回撤：净值 / 历史峰值 - 1 的最小值
- 换手：每期 |仓位变化| 之和（单边，单位「倍」）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def compute_metrics(
    equity: pd.Series,
    benchmark: pd.Series | None = None,
    position: pd.Series | None = None,
    rf: float = 0.02,
    trades: pd.DataFrame | None = None,
) -> dict[str, float]:
    equity = equity.astype(float).dropna()
    if len(equity) < 2:
        return {}

    ret = equity.pct_change().dropna()
    n = len(ret)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = n / TRADING_DAYS
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    ann_vol = float(ret.std(ddof=0) * np.sqrt(TRADING_DAYS))
    sharpe = (cagr - rf) / ann_vol if ann_vol > 0 else 0.0

    downside = ret[ret < 0]
    sortino = (cagr - rf) / (downside.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else 0.0

    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    out: dict[str, float] = {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "ann_vol": ann_vol,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "calmar": float(calmar),
        "days": float(n),
    }

    if position is not None:
        pos = position.astype(float).fillna(0.0)
        out["exposure"] = float(pos.mean())
        out["turnover"] = float(pos.diff().abs().fillna(pos.abs()).sum())

    if benchmark is not None and len(benchmark.dropna()) > 1:
        b = benchmark.astype(float).dropna()
        b_ret = b.iloc[-1] / b.iloc[0] - 1.0
        b_dd = float((b / b.cummax() - 1.0).min())
        out["benchmark_return"] = float(b_ret)
        out["benchmark_max_drawdown"] = b_dd
        out["excess_return"] = float(total_return - b_ret)
        out["excess_cagr"] = float(cagr - ((1 + b_ret) ** (1 / years) - 1 if years > 0 else 0.0))

    if trades is not None and not trades.empty:
        closed = trades[trades["pnl"].notna()]
        if not closed.empty:
            out["trades"] = float(len(closed))
            out["win_rate"] = float((closed["pnl"] > 0).mean())
            out["avg_win"] = float(closed.loc[closed["pnl"] > 0, "pnl"].mean()) if (closed["pnl"] > 0).any() else 0.0
            out["avg_loss"] = float(closed.loc[closed["pnl"] < 0, "pnl"].mean()) if (closed["pnl"] < 0).any() else 0.0

    return out


def format_metrics(metrics: dict[str, float]) -> str:
    pct = {"total_return", "cagr", "ann_vol", "max_drawdown", "exposure",
           "win_rate", "benchmark_return", "benchmark_max_drawdown", "excess_return", "excess_cagr"}
    lines = []
    for k, v in metrics.items():
        if not isinstance(v, (int, float)):
            lines.append(f"| {k} | {v} |")
        elif k in pct:
            lines.append(f"| {k} | {v:.2%} |")
        elif k in {"days", "trades"}:
            lines.append(f"| {k} | {int(v)} |")
        else:
            lines.append(f"| {k} | {v:.4f} |")
    return "\n".join(lines)
