"""Agent 行为测试：验证「规则触发方向」和「仓位约束」这两类最容易出错的地方。"""

from __future__ import annotations

import copy

import pandas as pd

from qtagent.agents import (
    AgentContext,
    MacroAgent,
    PortfolioState,
    RiskAgent,
    Signal,
    TechnicalAgent,
)
from qtagent.agents.decision import DecisionAgent
from qtagent.config import Config


def _ctx(cfg, market, features, position=0.0):
    return AgentContext(
        trade_date="2026-01-05",
        market=market,
        trade=pd.DataFrame(),
        news=pd.DataFrame(),
        features=features,
        portfolio=PortfolioState(equity=1_000_000, cash=1_000_000, position_ratio=position),
        cfg=cfg,
    )


def test_technical_reversal_buy_on_oversold(cfg, market):
    agent = TechnicalAgent({"mode": "reversal"})
    features = {
        "close": 100.0, "ma5": 101.0, "ma20": 105.0, "ma60": 90.0,
        "ret20": -0.12, "bias20": -0.05, "rsi": 25.0, "boll_pos": 0.05,
        "vol_ratio": 0.7, "ret1": -0.005,
    }
    sig = agent.run(_ctx(cfg, market, features))
    assert sig.score > 0.4
    assert any("超跌" in r for r in sig.reasons)
    assert sig.evidence["mode"] == "reversal"


def test_technical_dip_in_trend_blocks_below_ma60(cfg, market):
    agent = TechnicalAgent({"mode": "dip_in_trend"})
    common = {"ma5": 80.0, "ma20": 82.0, "ret20": -0.12, "bias20": -0.05, "rsi": 25.0,
              "boll_pos": 0.05, "vol_ratio": 0.7, "ret1": -0.005}
    above = agent.run(_ctx(cfg, market, {**common, "close": 100.0, "ma60": 90.0}))
    below = agent.run(_ctx(cfg, market, {**common, "close": 80.0, "ma60": 95.0}))
    assert above.score > 0.4
    assert below.score <= 0.0  # 跌破 MA60 不做多


def test_risk_cap_shrinks_when_vol_doubles(cfg, market):
    agent = RiskAgent()
    calm = agent.run(_ctx(cfg, market, {"vol20": 0.10, "dd_from_peak": -0.02, "atr_pct": 0.01, "close": 4.0}))
    wild = agent.run(_ctx(cfg, market, {"vol20": 0.40, "dd_from_peak": -0.02, "atr_pct": 0.03, "close": 4.0}))
    assert calm.evidence["position_cap"] > wild.evidence["position_cap"]
    assert wild.evidence["position_cap"] <= 0.35
    assert wild.score < 0


def test_macro_abstains_without_data(cfg, market):
    sig = MacroAgent().run(_ctx(cfg, market, {}))
    assert sig.confidence == 0.0 and sig.evidence["available"] is False


def test_decision_respects_cap_and_threshold(cfg, market):
    raw = copy.deepcopy(cfg.raw)
    raw["agents"]["decision"]["confirm_days"] = 1
    raw["agents"]["decision"]["buy_threshold"] = 0.05
    test_cfg = Config(raw)

    ctx = _ctx(test_cfg, market, {"close": 4.0}, position=0.0)
    ctx.peer_signals = {
        "technical": Signal("technical", 0.8, 1.0, ["技术强"],
                            {"mode": "trend", "regime": "up"}),
        "sentiment": Signal("sentiment", 0.0, 0.0, ["无样本"]),
        "macro": Signal("macro", 0.6, 1.0, ["外需扩张"]),
        "risk": Signal("risk", -0.2, 0.9, ["波动偏高"], {"position_cap": 0.4}),
    }
    d = DecisionAgent(test_cfg.decision).run(ctx)
    assert d.action == "BUY"
    assert 0 < d.target_position <= 0.4
    assert d.contributions["technical"] > 0
    assert any("风险预算" in r for r in d.reasons)


def test_decision_hold_in_neutral_band(cfg, market):
    raw = copy.deepcopy(cfg.raw)
    raw["agents"]["decision"].update({"buy_threshold": 0.3, "sell_threshold": -0.3, "confirm_days": 1})
    test_cfg = Config(raw)
    ctx = _ctx(test_cfg, market, {"close": 4.0}, position=0.3)
    ctx.peer_signals = {
        "technical": Signal("technical", 0.1, 1.0),
        "sentiment": Signal("sentiment", 0.0, 1.0),
        "macro": Signal("macro", 0.1, 1.0),
        "risk": Signal("risk", 0.0, 1.0, [], {"position_cap": 1.0}),
    }
    d = DecisionAgent(test_cfg.decision).run(ctx)
    assert d.action == "HOLD"
    assert abs(d.target_position - 0.3) < 1e-9
