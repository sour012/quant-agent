"""测试夹具：用确定性随机漫步构造行情，避免测试依赖网络与真实数据库。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qtagent.config import Config  # noqa: E402
from qtagent.orchestrator import TradingPipeline  # noqa: E402


def make_market(days: int = 400, seed: int = 7, drift: float = 0.0002) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, 0.012, days)
    close = 4.0 * np.exp(np.cumsum(ret))
    dates = pd.bdate_range("2023-01-02", periods=days)
    open_ = close * (1 + rng.normal(0, 0.002, days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, days)))
    return pd.DataFrame(
        {
            "index_code": "TEST",
            "trade_date": dates.strftime("%Y-%m-%d"),
            "open": open_, "high": high, "low": low, "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, days),
            "amount": rng.integers(1e8, 5e8, days),
        }
    )


def make_trade(months: int = 40, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2020-01", periods=months, freq="M")
    export = 3000e8 * (1 + rng.normal(0, 0.05, months))
    imp = 2200e8 * (1 + rng.normal(0, 0.05, months))
    return pd.DataFrame(
        {
            "trade_month": idx.astype(str),
            "region_code": "TOTAL",
            "export_usd": export,
            "import_usd": imp,
            "balance_usd": export - imp,
            "export_yoy": rng.normal(5, 12, months),
            "import_yoy": rng.normal(3, 10, months),
            "export_mom": rng.normal(0, 3, months),
            "import_mom": rng.normal(0, 3, months),
            "export_accum_usd": export * 6,
            "import_accum_usd": imp * 6,
            "source": "synthetic",
        }
    )


@pytest.fixture(scope="session")
def cfg() -> Config:
    return Config.load()


@pytest.fixture(scope="session")
def market() -> pd.DataFrame:
    return make_market()


@pytest.fixture(scope="session")
def trade() -> pd.DataFrame:
    return make_trade()


@pytest.fixture
def pipeline(cfg, market, trade) -> TradingPipeline:
    return TradingPipeline(cfg, market=market, trade=trade, news=pd.DataFrame(), benchmark=pd.DataFrame())
