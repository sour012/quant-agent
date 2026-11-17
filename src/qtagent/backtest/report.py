"""回测报告：净值图 + Markdown 结论（每次运行覆盖，保证结论与代码同步）。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .engine import BacktestResult  # noqa: E402
from .metrics import format_metrics  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_equity(daily: pd.DataFrame, path: Path, dpi: int = 150) -> Path:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}
    )
    x = pd.to_datetime(daily["trade_date"])
    ax1.plot(x, daily["equity"] / daily["equity"].iloc[0], label="策略净值", linewidth=1.6, color="#c0392b")
    ax1.plot(
        x,
        daily["benchmark_equity"] / daily["benchmark_equity"].iloc[0],
        label="沪深300 基准",
        linewidth=1.2,
        linestyle="--",
        color="#7f8c8d",
    )
    ax1.set_ylabel("净值（初始=1）")
    ax1.set_title("策略净值 vs 基准")
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(alpha=0.25)

    ax2.fill_between(x, daily["drawdown"] * 100, 0, color="#c0392b", alpha=0.35, label="策略回撤")
    ax2.plot(x, daily["benchmark_drawdown"] * 100, color="#7f8c8d", linewidth=1.0, linestyle="--", label="基准回撤")
    ax2.set_ylabel("回撤 (%)")
    ax2.legend(loc="lower left", frameon=False)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def build_markdown(result: BacktestResult, title: str, extra: dict | None = None) -> str:
    m = result.metrics
    lines = [f"# {title}", ""]
    lines += [
        f"- 回测区间：{m.get('start')} ~ {m.get('end')}",
        f"- 期末权益：{m.get('final_equity', 0):,.0f} 元",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        format_metrics(m),
        "",
    ]
    if extra:
        lines += ["## 配置与数据", ""]
        for k, v in extra.items():
            lines.append(f"- {k}：{v}")
        lines.append("")

    lines += ["## 交易记录（前 20 笔）", ""]
    if result.trades.empty:
        lines.append("区间内没有完整买卖回合。")
    else:
        t = result.trades.head(20)
        lines.append("| 建仓 | 平仓 | 持仓天数 | 盈亏（元） | 收益率 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for _, r in t.iterrows():
            lines.append(
                f"| {r['entry_date']} | {r['exit_date']} | {int(r['hold_days'])} | "
                f"{r['pnl']:,.0f} | {r['pnl_pct']:.2%} |"
            )
    lines.append("")
    return "\n".join(lines)
