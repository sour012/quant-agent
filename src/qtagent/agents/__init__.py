"""agents 子包：五个 Agent + 统一注册表。"""

from __future__ import annotations

from typing import Any

from .base import AgentContext, BaseAgent, PortfolioState, Signal
from .collector import CollectorAgent
from .decision import Decision, DecisionAgent
from .macro import MacroAgent
from .risk import RiskAgent
from .sentiment import SentimentAgent
from .technical import TechnicalAgent

__all__ = [
    "AgentContext",
    "BaseAgent",
    "PortfolioState",
    "Signal",
    "CollectorAgent",
    "TechnicalAgent",
    "SentimentAgent",
    "MacroAgent",
    "RiskAgent",
    "DecisionAgent",
    "Decision",
    "build_agents",
]


def build_agents(cfg: Any) -> dict[str, BaseAgent]:
    """按配置实例化所有 Agent（决策 Agent 单独持有，内部有状态）。"""
    return {
        "collector": CollectorAgent(cfg),
        "technical": TechnicalAgent(cfg),
        "sentiment": SentimentAgent(cfg),
        "macro": MacroAgent(cfg),
        "risk": RiskAgent(cfg),
    }
