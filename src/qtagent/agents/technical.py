"""技术分析 Agent：两套可切换的规则体系 + 一套「趋势内回踩」。

背景（见 research/factor_ic.py，可复跑）：
在本仓库覆盖的 2023-2026 区间，沪深300ETF 的动量类因子（ret20、bias20、MA20 斜率）
与未来 5/20 日收益的 IC 均为负（ret20 IC20 ≈ -0.03，bias20 IC20 ≈ -0.04），
说明该区间指数层面以**短期反转**为主。因此本 Agent 提供三种模式：

  mode = "trend"          纯顺势：均线结构 / MACD / 斜率 / 量价
  mode = "reversal"       纯反转：20 日超跌、偏离度、RSI、布林位置
  mode = "dip_in_trend"   趋势内回踩：只在中期趋势（价格 > MA60）时做超跌反弹

三种模式的分数与依据都写进 reasons，模式与权重交给 config / sweep 脚本决定。
"""

from __future__ import annotations

from typing import Any

from .base import AgentContext, BaseAgent, Signal


def _f(features: dict[str, Any], key: str) -> float | None:
    v = features.get(key)
    if isinstance(v, (int, float)) and v == v:  # 排除 NaN
        return float(v)
    return None


class TechnicalAgent(BaseAgent):
    name = "technical"
    title = "技术分析 Agent"
    responsibility = "按 trend / reversal / dip_in_trend 三种规则体系判断指数所处位置"
    default_weight = 0.40

    def run(self, ctx: AgentContext) -> Signal:
        f = ctx.features
        mode = str(self.params.get("mode", "trend"))

        close = _f(f, "close")
        ma5, ma20, ma60 = _f(f, "ma5"), _f(f, "ma20"), _f(f, "ma60")
        trend_up = bool(close and ma20 and close > ma20 and ma5 and ma5 > ma20)
        trend_down = bool(close and ma20 and close < ma20 and ma5 and ma5 < ma20)

        if mode == "trend":
            votes = self._trend_votes(f, close, ma5, ma20, ma60, trend_up)
        else:
            votes = self._reversal_votes(f, close, ma60, mode)

        score = sum(v for _, v, _ in votes)
        reasons = [f"模式：{mode}"] + [f"[{k}] {text}" for k, _, text in votes]
        fired = {k for k, v, _ in votes if v != 0}
        coverage = min(1.0, len(fired) / 5)
        sample = 1.0 if ctx.bars >= 120 else (0.7 if ctx.bars >= 60 else 0.4)
        confidence = 0.35 + 0.65 * coverage * sample

        above_ma60 = bool(close is not None and ma60 is not None and close > ma60)

        return Signal(
            agent=self.name,
            score=score,
            confidence=confidence,
            reasons=reasons,
            evidence={
                "mode": mode,
                "above_ma60": above_ma60,
                "rules_fired": sorted(fired),
                "raw_sum": round(score, 4),
                "regime": "up" if trend_up else ("down" if trend_down else "range"),
            },
        ).clamp()

    # ---------------- 顺势 ----------------
    def _trend_votes(self, f, close, ma5, ma20, ma60, trend_up):
        rsi = _f(f, "rsi")
        hist, dif, dea = _f(f, "macd_hist"), _f(f, "macd_dif"), _f(f, "macd_dea")
        vol_ratio, ret1, ret20 = _f(f, "vol_ratio"), _f(f, "ret1"), _f(f, "ret20")
        slope = _f(f, "ma20_slope5")
        votes: list[tuple[str, float, str]] = []

        if close is not None and ma20 is not None:
            votes.append(
                ("trend", +0.25 if close > ma20 else -0.25,
                 f"收盘 {close:.2f} {'站上' if close > ma20 else '跌破'} MA20 {ma20:.2f}")
            )
        if ma20 is not None and ma60 is not None:
            votes.append(
                ("ma_long", +0.15 if ma20 > ma60 else -0.15,
                 f"MA20 {ma20:.2f} {'高于' if ma20 > ma60 else '低于'} MA60 {ma60:.2f}")
            )
        if hist is not None:
            votes.append(("macd", +0.15 if hist > 0 else -0.15, f"MACD 柱 {hist:+.3f}"))
        if dif is not None and dea is not None:
            votes.append(
                ("macd_cross", +0.10 if dif > dea else -0.10,
                 f"DIF {dif:+.2f} 与 DEA {dea:+.2f} 的相对位置")
            )
        if slope is not None:
            votes.append(
                ("slope", +0.10 if slope > 0 else -0.10, f"MA20 五日斜率 {slope:+.2%}")
            )
        if ret20 is not None:
            votes.append(("momentum", +0.10 if ret20 > 0 else -0.10, f"20 日动量 {ret20:+.2%}"))
        if rsi is not None and trend_up and rsi > 70:
            votes.append(("rsi", +0.10, f"上升趋势中 RSI {rsi:.1f} 处强势区"))
        if vol_ratio is not None and ret1 is not None and vol_ratio > 1.2:
            votes.append(
                ("volume", +0.10 if ret1 > 0 else -0.10,
                 f"放量{'上涨' if ret1 > 0 else '下跌'}（量比 {vol_ratio:.2f}）")
            )
        return votes

    # ---------------- 反转 / 趋势内回踩 ----------------
    def _reversal_votes(self, f, close, ma60, mode):
        rsi = _f(f, "rsi")
        ret20, bias20 = _f(f, "ret20"), _f(f, "bias20")
        boll_pos = _f(f, "boll_pos")
        vol_ratio, ret1 = _f(f, "vol_ratio"), _f(f, "ret1")
        votes: list[tuple[str, float, str]] = []

        # 中期趋势过滤：dip_in_trend 模式下，跌破 MA60 不做多（只留做空方向的减分）
        above_ma60 = bool(close is not None and ma60 is not None and close > ma60)
        gate = 1.0 if (mode != "dip_in_trend" or above_ma60) else 0.0

        if ret20 is not None:
            if ret20 < -0.10:
                votes.append(("oversold", +0.35 * gate,
                              f"20 日跌幅 {ret20:.1%}，显著超跌"
                              + ("" if gate else "（已跌破 MA60，不做多）")))
            elif ret20 < -0.04:
                votes.append(("oversold", +0.22 * gate, f"20 日跌幅 {ret20:.1%}，短期超跌"))
            elif ret20 > 0.08:
                votes.append(("overheat", -0.25, f"20 日涨幅 {ret20:+.1%}，短期过热"))

        if bias20 is not None:
            if bias20 < -0.04:
                votes.append(("bias", +0.20 * gate, f"偏离 MA20 {bias20:.1%}，向下偏离较大"))
            elif bias20 > 0.06:
                votes.append(("bias", -0.20, f"偏离 MA20 {bias20:+.1%}，向上偏离较大"))

        if rsi is not None:
            if rsi < 30:
                votes.append(("rsi", +0.20 * gate, f"RSI {rsi:.1f} 超卖"))
            elif rsi > 70:
                votes.append(("rsi", -0.20, f"RSI {rsi:.1f} 超买"))
            elif rsi < 45:
                votes.append(("rsi", +0.05 * gate, f"RSI {rsi:.1f} 偏弱，接近超卖"))
            elif rsi > 60:
                votes.append(("rsi", -0.05, f"RSI {rsi:.1f} 偏强，接近超买"))

        if boll_pos is not None:
            if boll_pos < 0.10:
                votes.append(("boll", +0.15 * gate, f"贴近布林下轨（位置 {boll_pos:.0%}）"))
            elif boll_pos > 0.90:
                votes.append(("boll", -0.15, f"贴近布林上轨（位置 {boll_pos:.0%}）"))

        if vol_ratio is not None and ret1 is not None:
            if vol_ratio < 0.8 and ret1 < 0:
                votes.append(("volume", +0.05 * gate, f"缩量下跌（量比 {vol_ratio:.2f}），卖压衰竭"))
            elif vol_ratio > 1.3 and ret1 < 0:
                votes.append(("volume", -0.10, f"放量下跌（量比 {vol_ratio:.2f}），恐慌抛售"))

        return votes
