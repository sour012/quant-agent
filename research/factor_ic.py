"""因子有效性检验（IC / 分位收益），驱动技术 Agent 的模式选择。

用法：
    python research/factor_ic.py

输出：各因子与未来 5/20 日收益的秩相关（Spearman IC），以及按分位分组的平均未来收益。
结论会写进 README 与 reports/factor_ic.md —— 这是「为什么用反转模式」的证据来源。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qtagent.config import Config  # noqa: E402
from qtagent.orchestrator import TradingPipeline  # noqa: E402

FACTORS = ["ret20", "bias20", "bias5", "ma20_slope5", "rsi", "boll_pos", "vol20", "vol_ratio"]


def main() -> int:
    cfg = Config.load()
    pipeline = TradingPipeline(cfg)
    f = pipeline.features.copy()
    f["fwd5"] = f["close"].shift(-5) / f["close"] - 1
    f["fwd20"] = f["close"].shift(-20) / f["close"] - 1

    rows = []
    for col in FACTORS:
        if col not in f:
            continue
        sub = f[[col, "fwd5", "fwd20"]].dropna()
        rows.append(
            {
                "factor": col,
                "IC5": round(sub[col].corr(sub["fwd5"], method="spearman"), 4),
                "IC20": round(sub[col].corr(sub["fwd20"], method="spearman"), 4),
                "n": len(sub),
            }
        )
    ic = pd.DataFrame(rows).sort_values("IC20")
    print("\n=== 因子 IC（Spearman，2023-2026 沪深300ETF）===")
    print(ic.to_string(index=False))

    print("\n=== ret20 五分位 → 未来 20 日平均收益 ===")
    sub = f[["ret20", "fwd20"]].dropna()
    q = pd.qcut(sub["ret20"], 5, labels=[f"Q{i}" for i in range(1, 6)])
    print((sub.groupby(q, observed=True)["fwd20"].mean() * 100).round(2).to_string())

    print("\n=== bias20 五分位 → 未来 20 日平均收益 ===")
    sub2 = f[["bias20", "fwd20"]].dropna()
    q2 = pd.qcut(sub2["bias20"], 5, labels=[f"Q{i}" for i in range(1, 6)])
    print((sub2.groupby(q2, observed=True)["fwd20"].mean() * 100).round(2).to_string())

    out = cfg.report_dir / "factor_ic.csv"
    ic.to_csv(out, index=False, encoding="utf-8")
    print(f"\n已保存 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
