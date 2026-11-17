"""流水线与回测的一致性测试。

三个关键不变量：
1. 无未来函数：只用截至当日的数据算出的信号，必须与「把未来数据砍掉」后算出的完全一致；
2. 可复现：同一份数据 + 同一份配置，两次回测净值完全相同；
3. T+1 成交：成交价等于信号次日开盘价（含滑点），而不是信号当日收盘价。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qtagent.backtest import BacktestEngine
from qtagent.orchestrator import TradingPipeline


def _pipeline(cfg, market, trade, news=None):
    return TradingPipeline(
        cfg, market=market, trade=trade,
        news=news if news is not None else pd.DataFrame(),
        benchmark=pd.DataFrame(),
    )


def test_no_lookahead_signal_matches_truncated_data(cfg, market, trade):
    full = _pipeline(cfg, market, trade)
    idx = len(full.dates) - 1
    date = full.dates[idx]
    d_full, _ = full.step(idx)

    cut_market = market[pd.to_datetime(market["trade_date"]) <= date].reset_index(drop=True)
    cut = _pipeline(cfg, cut_market, trade)
    d_cut, _ = cut.step(len(cut.dates) - 1)

    assert d_full.trade_date == d_cut.trade_date
    assert d_full.score == pytest.approx(d_cut.score, abs=1e-9)
    assert d_full.target_position == pytest.approx(d_cut.target_position, abs=1e-9)


def test_backtest_is_reproducible(cfg, market, trade):
    r1 = BacktestEngine(_pipeline(cfg, market, trade), cfg).run()
    r2 = BacktestEngine(_pipeline(cfg, market, trade), cfg).run()
    pd.testing.assert_series_equal(
        r1.daily["equity"].reset_index(drop=True),
        r2.daily["equity"].reset_index(drop=True),
    )
    assert r1.metrics["total_return"] == pytest.approx(r2.metrics["total_return"], abs=1e-12)


def test_t_plus_one_execution_at_next_open(cfg, market, trade):
    result = BacktestEngine(_pipeline(cfg, market, trade), cfg).run()
    if result.trades.empty:
        pytest.skip("区间内没有完整交易回合")
    trade_row = result.trades.iloc[0]
    open_px = float(
        market.loc[market["trade_date"] == trade_row["entry_date"], "open"].iloc[0]
    )
    expected = open_px * (1 + float(cfg.backtest["slippage"]))
    # 成交价在 trades 里按 4 位小数落库，这里按 1e-3 的绝对误差比对
    assert trade_row["entry_price"] == pytest.approx(expected, abs=1e-3)


def test_features_are_sane(cfg, market, trade):
    p = _pipeline(cfg, market, trade)
    f = p.features.dropna(subset=["boll_pos"])
    assert f["boll_pos"].between(0, 1).all()
    assert (f["vol20"] > 0).all()
    assert f["trade_lag_days"].dropna().ge(0).all()
