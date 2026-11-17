"""宏观/贸易 Agent。

职责：把「中美贸易—经济数据库」里的进出口同比、贸易差额变化，翻译成对权益的方向判断。
因子的构造逻辑（可解释）：
  出口同比走强  → 外需扩张，制造业与出口链盈利预期改善 → 偏多
  进口同比走强  → 内需修复，但同时可能是进口价格（能源/芯片）上行 → 小幅偏多
  贸易差额改善  → 结汇与外汇占款改善，流动性边际宽松 → 偏多
每个因子先做 12 个月滚动 z-score（剔除季节性水平差异），再加权求和。
"""

from __future__ import annotations

from typing import Any

from .base import AgentContext, BaseAgent, Signal


def _f(features: dict[str, Any], key: str) -> float | None:
    v = features.get(key)
    if isinstance(v, (int, float)) and v == v:
        return float(v)
    return None


class MacroAgent(BaseAgent):
    name = "macro"
    title = "宏观贸易 Agent"
    responsibility = "基于海关进出口同比与贸易差额变化，判断外需与流动性对权益的传导"
    default_weight = 0.20

    def run(self, ctx: AgentContext) -> Signal:
        cfg = ctx.cfg
        params = cfg.agent(self.name) if cfg else {}
        w_e = float(params.get("weight_export", 0.5))
        w_i = float(params.get("weight_import", 0.2))
        w_b = float(params.get("weight_balance", 0.3))

        f = ctx.features
        z_e, z_i, z_b = _f(f, "export_yoy_z"), _f(f, "import_yoy_z"), _f(f, "balance_yoy_z")
        if z_e is None and z_b is None:
            return Signal(
                agent=self.name, score=0.0, confidence=0.0,
                reasons=["贸易宏观数据不可用，宏观 Agent 弃权"], evidence={"available": False},
            )

        def norm(z: float | None) -> float:
            return 0.0 if z is None else max(-1.0, min(1.0, z / 2.0))  # |z|≥2 视为极端

        score = w_e * norm(z_e) + w_i * norm(z_i) + w_b * norm(z_b)

        reasons: list[str] = []
        lag = _f(f, "trade_lag_days")
        reasons.append(
            f"最新可得月份数据滞后 {int(lag)} 天"
            if lag is not None else "贸易数据已对齐至最新可得月份"
        )
        if _f(f, "export_yoy") is not None:
            reasons.append(
                f"出口同比 {_f(f, 'export_yoy'):+.1f}%（z={_f(f, 'export_yoy_z'):+.2f}）→ "
                f"{'外需扩张' if (_f(f, 'export_yoy_z') or 0) > 0 else '外需偏弱'}"
            )
        if _f(f, "import_yoy") is not None:
            reasons.append(
                f"进口同比 {_f(f, 'import_yoy'):+.1f}%（z={_f(f, 'import_yoy_z'):+.2f}）"
            )
        balance_yoy = _f(f, "balance_yoy")
        if balance_yoy is not None:
            reasons.append(
                f"贸易差额同比 {balance_yoy:+.1f}% → "
                f"{'顺差扩张，结汇与流动性偏友好' if balance_yoy > 0 else '顺差收窄'}"
            )

        # 置信度：数据越新鲜越可信
        if lag is None:
            confidence = 0.5
        elif lag <= 45:
            confidence = 0.9
        elif lag <= 75:
            confidence = 0.75
        else:
            confidence = 0.55

        return Signal(
            agent=self.name,
            score=score,
            confidence=confidence,
            reasons=reasons,
            evidence={
                "available": True,
                "z_export": z_e, "z_import": z_i, "z_balance": z_b,
                "weights": {"export": w_e, "import": w_i, "balance": w_b},
                "trade_lag_days": lag,
            },
        ).clamp()
