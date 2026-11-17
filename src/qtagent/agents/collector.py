"""数据采集与校验 Agent。

它是流水线第一棒：确认当日拿到的行情/贸易/新闻是否完整、是否陈旧，
并给出数据质量分（= 后续所有信号的可信度上限）。
它不发表多空观点，只保证「输入是干净的」。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import AgentContext, BaseAgent, Signal

REQUIRED_MARKET_COLS = ["open", "high", "low", "close", "volume"]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value == value  # NaN != NaN


class CollectorAgent(BaseAgent):
    name = "collector"
    title = "数据采集 Agent"
    responsibility = "抓取行情/海关/新闻数据，做完整性校验与特征对齐，产出统一数据底座"
    default_weight = 0.10

    def run(self, ctx: AgentContext) -> Signal:
        cfg = ctx.cfg
        params = cfg.agent(self.name) if cfg else {}
        max_lag = int(params.get("max_lag_days", 5))
        max_missing = float(params.get("max_missing_ratio", 0.02))

        reasons: list[str] = []
        penalties = 0.0
        evidence: dict[str, Any] = {}

        # 1) 样本量
        bars = ctx.bars
        evidence["bars"] = bars
        if bars < 60:
            penalties += 0.6
            reasons.append(f"样本不足：仅 {bars} 根日线，技术指标不可靠")
        elif bars < 120:
            penalties += 0.2
            reasons.append(f"样本偏少：{bars} 根日线，长周期指标按可用窗口计算")
        else:
            reasons.append(f"样本充足：{bars} 根日线")

        # 2) 行情字段完整性
        window = ctx.market.tail(250)
        missing_ratio = float(window[REQUIRED_MARKET_COLS].isna().mean().mean()) if bars else 1.0
        evidence["missing_ratio"] = round(missing_ratio, 4)
        if missing_ratio > max_missing:
            penalties += 0.4
            reasons.append(f"行情字段缺失率 {missing_ratio:.2%} 超过阈值 {max_missing:.2%}")
        else:
            reasons.append(f"行情字段完整（缺失率 {missing_ratio:.2%}）")

        # 3) 数据新鲜度
        last_date = pd.to_datetime(ctx.market["trade_date"].iloc[-1]).date()
        lag = (pd.to_datetime(ctx.trade_date).date() - last_date).days
        evidence["last_bar"] = str(last_date)
        evidence["lag_days"] = lag
        if lag > max_lag:
            penalties += 0.5
            reasons.append(f"行情最新数据落后 {lag} 天，超出容忍度 {max_lag} 天")
        else:
            reasons.append("行情数据已更新至最新交易日")

        # 4) 辅助数据（贸易 / 新闻）
        trade_ok = _is_number(ctx.features.get("export_yoy"))
        news_cnt = int(ctx.features.get("news_count", 0) or 0)
        evidence["trade_available"] = trade_ok
        evidence["news_count_5d"] = news_cnt
        if not trade_ok:
            penalties += 0.3
            reasons.append("贸易宏观特征缺失，宏观 Agent 将不参与投票")
        if news_cnt == 0:
            reasons.append("近 5 日无新闻样本，情绪 Agent 将降级为市场情绪代理")

        confidence = max(0.0, min(1.0, 1.0 - penalties))
        return Signal(
            agent=self.name,
            score=0.0,
            confidence=confidence,
            reasons=reasons,
            evidence=evidence,
        ).clamp()
