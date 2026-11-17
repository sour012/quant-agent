"""新闻情绪 Agent。

两条通道：
1. 主通道：财经快讯标题/摘要的词典情感分，按时间衰减加权聚合；
2. 降级通道：近 5 日无足够新闻样本时，用「量价 + 5 日动量」构造市场情绪代理，
   并在 evidence 中显式标记 degraded=True —— 宁可说明数据不足，也不用噪声充数。
"""

from __future__ import annotations

import math
from typing import Any

from .base import AgentContext, BaseAgent, Signal


def _f(features: dict[str, Any], key: str) -> float | None:
    v = features.get(key)
    if isinstance(v, (int, float)) and v == v:
        return float(v)
    return None


class SentimentAgent(BaseAgent):
    name = "sentiment"
    title = "新闻情绪 Agent"
    responsibility = "聚合财经快讯情感倾向，衡量市场短期情绪温度"
    default_weight = 0.15

    def run(self, ctx: AgentContext) -> Signal:
        cfg = ctx.cfg
        params = cfg.agent(self.name) if cfg else {}
        min_news = int(params.get("min_news_for_full_conf", 25))
        fallback_enabled = bool(params.get("fallback_on_market", True))

        f = ctx.features
        news_score = _f(f, "news_score") or 0.0
        news_count = int(_f(f, "news_count") or 0)
        news_pos = int(_f(f, "news_pos") or 0)
        news_neg = int(_f(f, "news_neg") or 0)

        if news_count > 0:
            confidence = min(1.0, news_count / max(1, min_news))
            confidence = 0.35 + 0.65 * confidence
            reasons = [
                f"近 5 日样本 {news_count} 条快讯（偏正面 {news_pos} / 偏负面 {news_neg}）",
                f"衰减加权情绪分 {news_score:+.2f}"
                + ("，情绪偏暖" if news_score > 0.1 else "，情绪偏冷" if news_score < -0.1 else "，情绪中性"),
            ]
            return Signal(
                agent=self.name,
                score=news_score,
                confidence=confidence,
                reasons=reasons,
                evidence={"degraded": False, "news_count": news_count,
                          "news_pos": news_pos, "news_neg": news_neg},
            ).clamp()

        if not fallback_enabled:
            return Signal(
                agent=self.name, score=0.0, confidence=0.0,
                reasons=["近期无新闻样本，情绪 Agent 弃权"], evidence={"degraded": True},
            )

        # ---- 降级为市场情绪代理：量能 + 5 日动量 ----
        ret5 = _f(f, "ret5")
        vol_ratio = _f(f, "vol_ratio")
        if ret5 is None:
            return Signal(
                agent=self.name, score=0.0, confidence=0.0,
                reasons=["无新闻且行情不足，情绪 Agent 弃权"], evidence={"degraded": True},
            )
        momentum = math.tanh(ret5 * 6.0)
        vol_effect = 0.0
        if vol_ratio is not None:
            vol_effect = math.tanh((vol_ratio - 1.0) * 1.5) * (1.0 if ret5 >= 0 else -1.0)
        score = 0.65 * momentum + 0.35 * vol_effect

        reasons = [
            "近 5 日无可用新闻，降级为市场情绪代理（5 日动量 + 量比）",
            f"5 日涨跌幅 {ret5:+.2%}，量比 {vol_ratio:.2f}" if vol_ratio is not None else f"5 日涨跌幅 {ret5:+.2%}",
        ]
        return Signal(
            agent=self.name, score=score, confidence=0.35,
            reasons=reasons, evidence={"degraded": True, "proxy": True},
        ).clamp()
