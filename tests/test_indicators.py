"""指标计算的单元测试：优先覆盖「容易写反」的地方（如布林通道上下轨顺序）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qtagent import indicators as ind


def test_sma_basic():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = ind.sma(s, 3)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert out.iloc[2] == 2.0
    assert out.iloc[4] == 4.0


def test_bollinger_order():
    s = pd.Series(np.linspace(10, 20, 50))
    lower, mid, upper = ind.bollinger(s, 20, 2.0)
    tail = slice(25, None)
    assert (upper[tail] > mid[tail]).all()
    assert (mid[tail] > lower[tail]).all()


def test_rsi_bounds_and_direction():
    up = pd.Series(np.linspace(10, 30, 60))
    down = pd.Series(np.linspace(30, 10, 60))
    rsi_up, rsi_down = ind.rsi(up), ind.rsi(down)
    assert rsi_up.dropna().between(0, 100).all()
    assert rsi_up.iloc[-1] > 70
    assert rsi_down.iloc[-1] < 30


def test_atr_positive():
    df = pd.DataFrame(
        {"high": [11, 12, 13, 14, 15] * 6, "low": [9, 10, 11, 12, 13] * 6, "close": [10, 11, 12, 13, 14] * 6}
    )
    a = ind.atr(df, 14).dropna()
    assert not a.empty and (a > 0).all()


def test_max_drawdown():
    equity = pd.Series([1.0, 1.2, 0.9, 1.1])
    assert ind.max_drawdown(equity) == -0.25


def test_annualized_vol_scale():
    ret = pd.Series(np.full(60, 0.01))
    vol = ind.annualized_vol(ret, 20).dropna()
    assert (vol < 0.01).all()  # 常数收益序列波动应为 0
