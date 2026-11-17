"""技术指标计算：全部为无状态纯函数，便于单元测试与复用。"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI。"""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    # 只涨不跌时 avg_loss=0，RSI 定义上为 100；两者都为 0 视为无波动，取中性 50
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return out.fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    hist = (dif - dea) * 2  # 国内软件常用的柱体定义
    return dif, dea, hist


def bollinger(close: pd.Series, window: int = 20, k: float = 2.0):
    mid = sma(close, window)
    std = close.rolling(window, min_periods=window).std(ddof=0)
    return mid - k * std, mid, mid + k * std


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def annualized_vol(returns: pd.Series, window: int = 20, periods: int = TRADING_DAYS) -> pd.Series:
    return returns.rolling(window, min_periods=max(5, window // 2)).std(ddof=0) * np.sqrt(periods)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(3, window // 2)).mean()
    std = series.rolling(window, min_periods=max(3, window // 2)).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def hist_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """历史模拟法 VaR（返回正数，表示该置信度下的单期最大预期亏损）。"""
    if returns.dropna().empty:
        return 0.0
    q = np.nanquantile(returns.dropna(), 1 - confidence)
    return float(max(0.0, -q))
