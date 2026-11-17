"""风险管理 Agent。

它不负责「看多看空」，而负责「允许多大仓位」：
  1. 波动率目标法给出仓位上限（vol target / realized vol）；
  2. 波动异常放大或净值深度回撤时强制降档（risk-off）；
  3. 输出 VaR、ATR 止损位等风险披露，供决策 Agent 与报告使用。
方向分只做防御性扣减，避免和趋势类 Agent 抢同一个职能。
"""

from __future__ import annotations

from typing import Any

from .. import indicators as ind
from .base import AgentContext, BaseAgent, Signal


def _f(features: dict[str, Any], key: str) -> float | None:
    v = features.get(key)
    if isinstance(v, (int, float)) and v == v:
        return float(v)
    return None


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class RiskAgent(BaseAgent):
    name = "risk"
    title = "风险管理 Agent"
    responsibility = "以波动率目标、VaR 与回撤约束计算仓位上限，并做风险敞口披露"
    default_weight = 0.25

    def run(self, ctx: AgentContext) -> Signal:
        cfg = ctx.cfg
        p = cfg.agent(self.name) if cfg else {}
        vol_target = float(p.get("vol_target", 0.18))
        max_pos = float(p.get("max_position", 1.0))
        risk_off_mult = float(p.get("risk_off_vol_multiple", 1.6))
        var_conf = float(p.get("var_confidence", 0.95))
        stop_mult = float(p.get("stop_loss_atr", 2.5))

        f = ctx.features
        vol20 = _f(f, "vol20")
        dd = _f(f, "dd_from_peak") or 0.0
        atr_pct = _f(f, "atr_pct")
        close = _f(f, "close") or ctx.last_price

        if vol20 is None or vol20 <= 0:
            return Signal(
                agent=self.name, score=-0.2, confidence=0.4,
                reasons=["波动率不可计算，按保守仓位处理"],
                evidence={"position_cap": round(0.5 * max_pos, 3)},
            ).clamp()

        ratio = vol20 / vol_target
        # 波动率目标：实时波动越高，允许仓位越低
        cap = min(max_pos, vol_target / vol20) if vol20 > 0 else max_pos
        risk_off = ratio > risk_off_mult
        if risk_off:
            cap *= 0.5
        if dd < -0.15:
            cap *= 0.7
        cap = round(_clip(cap, 0.0, max_pos), 3)

        score = -0.6 * _clip(ratio - 1.0) - 0.4 * _clip((-dd - 0.10) / 0.20)

        var95 = ind.hist_var(ctx.returns.tail(250), var_conf)
        stop_price = close * (1 - (atr_pct or 0.02) * stop_mult)

        reasons = [
            f"20 日年化波动 {vol20:.1%}，为目标的 {ratio:.2f} 倍 → 仓位上限 {cap:.0%}",
            f"距区间高点回撤 {dd:.1%}，历史 VaR({var_conf:.0%}) 约 {var95:.2%}/日",
            f"ATR 止损参考位 {stop_price:.0f}（{stop_mult}x ATR）",
        ]
        if risk_off:
            reasons.append(f"波动超过目标 {risk_off_mult} 倍，触发 risk-off 降仓")

        return Signal(
            agent=self.name,
            score=score,
            confidence=0.95 if ctx.bars >= 60 else 0.5,
            reasons=reasons,
            evidence={
                "vol20": vol20, "vol_target": vol_target, "vol_ratio": ratio,
                "position_cap": cap, "var95": var95, "stop_price": round(stop_price, 2),
                "drawdown": dd, "risk_off": risk_off,
            },
        ).clamp()
