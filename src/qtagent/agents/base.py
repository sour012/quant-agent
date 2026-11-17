"""Agent 公共契约。

所有 Agent 统一输入 AgentContext、输出 Signal：
  score       ∈ [-1, 1]，正数偏多、负数偏空
  confidence  ∈ [0, 1]，数据质量/样本量决定的可信度
  reasons     人可读的依据（用于归因报告，而不是黑盒打分）

这样做的好处：决策层可以纯数学地做加权合成，并且每一分都能追溯到具体规则。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..config import Config


@dataclass
class Signal:
    agent: str
    score: float
    confidence: float
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def clamp(self) -> "Signal":
        self.score = float(max(-1.0, min(1.0, self.score)))
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        return self

    @property
    def weighted(self) -> float:
        """经置信度折减后的有效分数。"""
        return self.score * self.confidence


@dataclass
class PortfolioState:
    equity: float = 0.0
    cash: float = 0.0
    position_ratio: float = 0.0
    last_action: str = "HOLD"
    days_in_position: int = 0


@dataclass
class AgentContext:
    trade_date: str
    market: pd.DataFrame               # 截至 trade_date（含）的日线
    trade: pd.DataFrame                # 截至当月（含）的月度贸易数据
    news: pd.DataFrame                 # 截至 trade_date 的新闻
    features: dict[str, Any] = field(default_factory=dict)
    portfolio: PortfolioState = field(default_factory=PortfolioState)
    peer_signals: dict[str, Signal] = field(default_factory=dict)
    cfg: Config | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def close(self) -> pd.Series:
        return self.market["close"]

    @property
    def returns(self) -> pd.Series:
        return self.close.pct_change()

    @property
    def last_price(self) -> float:
        return float(self.market["close"].iloc[-1])

    @property
    def bars(self) -> int:
        return int(len(self.market))


class BaseAgent(ABC):
    name: str = "base"
    title: str = "基础 Agent"
    responsibility: str = ""
    default_weight: float = 0.0

    def __init__(self, cfg_or_params: Any = None) -> None:
        # 传 Config 时参数「实时」从配置读取，改配置立即生效（方便调参/网格搜索）；
        # 传 dict 时作为固定参数，便于单元测试直接构造。
        if isinstance(cfg_or_params, dict):
            self._params_override: dict[str, Any] | None = dict(cfg_or_params)
            self._cfg_ref: Config | None = None
        else:
            self._params_override = None
            self._cfg_ref = cfg_or_params

    @property
    def params(self) -> dict[str, Any]:
        if self._params_override is not None:
            return self._params_override
        if self._cfg_ref is not None:
            return self._cfg_ref.agent(self.name)
        return {}

    def weight(self, cfg: Config | None) -> float:
        if cfg is None:
            return self.default_weight
        weights = cfg.weights
        return float(weights.get(self.name, self.default_weight))

    @abstractmethod
    def run(self, ctx: AgentContext) -> Signal:
        raise NotImplementedError

    # 供子类使用的小工具
    @staticmethod
    def neutral(reason: str, agent: str) -> Signal:
        return Signal(agent=agent, score=0.0, confidence=0.0, reasons=[reason])
