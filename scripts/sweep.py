"""参数敏感性研究：样本内选参、样本外验证。

用法：
    python scripts/sweep.py            # 默认网格
    python scripts/sweep.py --quick    # 快速小网格

输出 reports/sweep.csv，按样本外夏普降序。
注意：这是研究工具，结论必须同时看两段（样本内/样本外）是否同向，
单看全样本最优的组合大概率是过拟合。
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qtagent import setup_logging  # noqa: E402
from qtagent.backtest import BacktestEngine  # noqa: E402
from qtagent.config import Config  # noqa: E402
from qtagent.orchestrator import TradingPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--in-sample", type=float, default=0.65)
    args = parser.parse_args()
    setup_logging("WARNING")

    cfg = Config.load()
    pipeline = TradingPipeline(cfg)
    dates = pipeline.dates
    split = dates[int(len(dates) * args.in_sample)].strftime("%Y-%m-%d")
    first = dates[pipeline.WARMUP_BARS].strftime("%Y-%m-%d")
    last = dates[-1].strftime("%Y-%m-%d")

    if args.quick:
        grid = dict(
            mode=["dip_in_trend", "trend"],
            confirm_days=[2],
            buy_threshold=[0.05],
            sell_threshold=[-0.20],
            full_score=[0.35],
        )
    else:
        grid = dict(
            mode=["trend", "reversal", "dip_in_trend"],
            confirm_days=[2, 3],
            buy_threshold=[0.05, 0.12],
            sell_threshold=[-0.15, -0.25],
            full_score=[0.35],
        )

    keys = list(grid)
    rows = []
    combos = list(itertools.product(*[grid[k] for k in keys]))
    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        mode = params.pop("mode")
        cfg.raw["agents"]["technical"]["mode"] = mode
        cfg.raw["agents"]["decision"].update(params)
        params = {"mode": mode, **params}
        ins = BacktestEngine(pipeline, cfg, start=first, end=split).run().metrics
        oos = BacktestEngine(pipeline, cfg, start=split, end=last).run().metrics
        rows.append(
            {
                **params,
                "in_cagr": round(ins.get("cagr", 0.0), 4),
                "in_sharpe": round(ins.get("sharpe", 0.0), 3),
                "in_mdd": round(ins.get("max_drawdown", 0.0), 4),
                "oos_cagr": round(oos.get("cagr", 0.0), 4),
                "oos_sharpe": round(oos.get("sharpe", 0.0), 3),
                "oos_mdd": round(oos.get("max_drawdown", 0.0), 4),
                "oos_excess": round(oos.get("excess_return", 0.0), 4),
                "trades": int(oos.get("trades", 0)),
            }
        )
        print(f"[{i}/{len(combos)}] {params} in={rows[-1]['in_cagr']:+.2%} oos={rows[-1]['oos_cagr']:+.2%} trades={rows[-1]['trades']}", flush=True)

    df = pd.DataFrame(rows).sort_values("oos_sharpe", ascending=False)
    out = cfg.report_dir / "sweep.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\n样本内 {first} ~ {split} / 样本外 {split} ~ {last}")
    print(df.head(12).to_string(index=False))
    print(f"\n已保存 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
