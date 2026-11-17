"""交易决策 Agent：可解释的加权规则投票 + 风险预算约束。

分两层，职责不混：

  方向层   score = Σ(w_i · s_i · c_i) / Σ(w_i)      i ∈ {技术, 情绪, 宏观}
           三个 Agent 各自给 [-1,1] 的方向分，乘置信度后按权重合成。

  预算层   cap   = 风险管理 Agent 给出的仓位上限（波动率目标 + VaR + 回撤）
           target = cap · clip(score / full_score, 0, 1)       （A 股现货不开空，仓位 ≥ 0）

为什么风险 Agent 不参与方向投票：它的分数天然 ≤ 0（高波动只会降仓不会看多），
若放进加权平均会长期把综合分往下拽，导致系统几乎不建仓；把它放到预算层，
既保留了风控作用，又不污染方向判断。

方向之上再叠加三条「上下文规则」，这是单个 Agent 看不到的：
  1. 确认机制：连续 N 日同向才建仓，过滤单日噪声；
  2. 中性带：分数介于 exit_threshold 与 entry_threshold 之间时维持现有仓位，不来回折腾；
  3. 记忆：保留最近 M 日决策，用于计算确认天数与归因复盘。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .base import AgentContext, BaseAgent, Signal

VOTERS = ("technical", "sentiment", "macro")
AGENT_CN = {
    "technical": "技术面",
    "sentiment": "情绪面",
    "macro": "宏观面",
    "risk": "风险预算",
}


@dataclass
class Decision:
    trade_date: str
    action: str                 # BUY / SELL / HOLD
    score: float
    confidence: float
    target_position: float
    position_cap: float
    contributions: dict[str, float] = field(default_factory=dict)
    effective_scores: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    agent_reasons: dict[str, list[str]] = field(default_factory=dict)
    memory_days: int = 0
    confirm_days: int = 0

    def to_record(self) -> dict[str, Any]:
        return {
            "trade_date": self.trade_date,
            "action": self.action,
            "score": round(self.score, 6),
            "confidence": round(self.confidence, 6),
            "target_position": round(self.target_position, 6),
            "contributions": self.contributions,
            "reasons": self.reasons,
        }


class DecisionAgent(BaseAgent):
    name = "decision"
    title = "交易决策 Agent"
    responsibility = "合成方向票 + 套用风险预算，输出买入/卖出/持有与完整归因"
    default_weight = 0.0

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._memory: deque[dict[str, Any]] = deque(maxlen=120)
        self._confirm_streak = 0
        self._streak_sign = 0

    def reset(self) -> None:
        self._memory.clear()
        self._confirm_streak = 0
        self._streak_sign = 0

    # ---------------- 主流程 ----------------
    def run(self, ctx: AgentContext) -> Decision:  # type: ignore[override]
        cfg = ctx.cfg
        p = cfg.decision if cfg else {}
        weights = cfg.weights if cfg else {k: 1 / len(VOTERS) for k in VOTERS}
        entry_thr = float(p.get("buy_threshold", 0.12))
        exit_thr = float(p.get("sell_threshold", -0.25))
        band = float(p.get("rebalance_band", 0.15))
        confirm_days = int(p.get("confirm_days", 2))
        full_score = float(p.get("full_score", 0.45))

        # ---- 1) 方向层：加权合成 ----
        contributions: dict[str, float] = {}
        effective: dict[str, float] = {}
        agent_reasons: dict[str, list[str]] = {}
        total_w = 0.0
        for name in VOTERS:
            sig: Signal | None = ctx.peer_signals.get(name)
            if sig is None or sig.confidence <= 0:
                contributions[name] = 0.0
                effective[name] = 0.0
                continue
            w = float(weights.get(name, 0.0))
            eff = sig.score * sig.confidence
            effective[name], contributions[name] = eff, w * eff
            agent_reasons[name] = list(sig.reasons)
            total_w += w
        raw = (sum(contributions.values()) / total_w) if total_w > 0 else 0.0

        # ---- 2) 预算层：风险 Agent 只给上限 ----
        tech = ctx.peer_signals.get("technical")
        tech_mode = str(tech.evidence.get("mode", "trend")) if tech is not None else "trend"
        tech_regime = str(tech.evidence.get("regime", "")) if tech is not None else ""
        trend_rules = tech_mode == "trend"

        risk_sig = ctx.peer_signals.get("risk")
        cap = float((risk_sig.evidence.get("position_cap") if risk_sig else None) or 1.0)
        cap = max(0.0, min(1.0, cap))

        current = ctx.portfolio.position_ratio
        reasons: list[str] = []

        # ---- 3) 目标仓位 ----
        if raw >= entry_thr:
            target = cap * min(1.0, raw / max(1e-9, full_score))
            band_note = (
                f"方向分 {raw:+.3f} ≥ 建仓阈值 {entry_thr:.2f}，"
                f"按 {min(1.0, raw / full_score):.0%} 的比例使用风险预算（上限 {cap:.0%}）"
            )
        elif raw <= exit_thr:
            target = 0.0
            band_note = f"方向分 {raw:+.3f} ≤ 清仓阈值 {exit_thr:.2f}，清仓"
        else:
            target = current
            band_note = (
                f"方向分 {raw:+.3f} 落在中性带（{exit_thr:.2f} ~ {entry_thr:.2f}），维持现有仓位"
            )
        reasons.append(band_note)

        # ---- 4) 上下文规则 ----
        self._update_streak(raw)

        if trend_rules and bool(p.get("entry_require_trend", False)) and target > 0 and current <= 1e-9:
            if tech_regime != "up":
                target = 0.0
                reasons.append("技术面未确认上升趋势（MA5/MA20 结构不成立），暂不建仓")
        if tech_mode == "dip_in_trend" and current > 1e-9 and tech is not None:
            if not tech.evidence.get("above_ma60", True):
                target = 0.0
                reasons.append("价格跌破 MA60，中期趋势失守，按回踩策略纪律退出")
        if trend_rules and bool(p.get("exit_on_trend_break", False)) and current > 1e-9:
            if tech_regime == "down":
                target = 0.0
                reasons.append("技术面转为下降趋势（价格与 MA5 同时跌破 MA20），趋势破位清仓")

        if target > 0 and current <= 1e-9 and self._confirm_streak < confirm_days:
            target = 0.0
            reasons.append(f"方向仅连续 {self._confirm_streak} 日同向，未达确认阈值 {confirm_days} 日，暂不建仓")

        if target > 0 and current > 0 and abs(target - current) < band:
            target = current
            reasons.append(f"目标与当前仓位差异小于调仓带 {band:.0%}，维持不动")

        target = round(target, 4)
        if target > current + 1e-9:
            action = "BUY"
        elif target < current - 1e-9:
            action = "SELL"
        else:
            action = "HOLD"

        # ---- 5) 归因 ----
        reasons.insert(0, self._attribution(raw, effective, contributions, total_w, entry_thr))
        if cap < 0.999:
            reasons.append(f"风险预算上限 {cap:.0%}（20 日波动 {risk_sig.evidence.get('vol20', 0):.1%}）"
                           if risk_sig else f"风险预算上限 {cap:.0%}")
        for name in VOTERS:
            if contributions.get(name, 0.0) != 0:
                reasons.append(
                    f"{AGENT_CN[name]}贡献 {contributions[name]:+.3f}"
                    f"（有效分 {effective[name]:+.2f} × 权重 {weights.get(name, 0):.2f}）"
                )

        conf = min((s.confidence for s in ctx.peer_signals.values() if s.confidence > 0), default=0.0)
        decision = Decision(
            trade_date=ctx.trade_date,
            action=action,
            score=raw,
            confidence=conf,
            target_position=target,
            position_cap=cap,
            contributions={k: round(v, 6) for k, v in contributions.items()},
            effective_scores={k: round(v, 6) for k, v in effective.items()},
            reasons=reasons,
            agent_reasons=agent_reasons,
            memory_days=len(self._memory),
            confirm_days=self._confirm_streak,
        )
        self._memory.append(decision.to_record())
        return decision

    # ---------------- 内部 ----------------
    def _update_streak(self, raw: float) -> None:
        sign = 1 if raw > 0.02 else (-1 if raw < -0.02 else 0)
        if sign == 0:
            self._confirm_streak, self._streak_sign = 0, 0
        elif sign == self._streak_sign:
            self._confirm_streak += 1
        else:
            self._streak_sign, self._confirm_streak = sign, 1

    @staticmethod
    def _attribution(raw, effective, contributions, total_w, entry_thr) -> str:
        if not contributions:
            return "无可用 Agent 信号，保持空仓"
        best = max(contributions, key=lambda k: abs(contributions[k]))
        stance = "偏多" if raw > entry_thr else ("偏空" if raw < -entry_thr else "多空均衡")
        return (
            f"方向分 {raw:+.3f}（有效权重合计 {total_w:.2f}）→ {stance}，"
            f"主要驱动为{AGENT_CN[best]}（贡献 {contributions[best]:+.3f}，有效分 {effective[best]:+.2f}）"
        )
