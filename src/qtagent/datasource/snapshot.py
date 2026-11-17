"""离线快照数据源：从 data/snapshot 读取已冻结的 CSV。

存在意义：网络不可用、或需要复现历史结论时，用它替代在线接口，
保证「同一份快照 + 同一份配置 = 同一份回测结果」。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class SnapshotClient:
    def __init__(self, snapshot_dir: str | Path) -> None:
        self.dir = Path(snapshot_dir)

    def _read(self, name: str) -> pd.DataFrame:
        path = self.dir / name
        if not path.exists():
            raise FileNotFoundError(f"缺少快照文件: {path}")
        return pd.read_csv(path, dtype={"trade_month": str, "trade_date": str})

    def customs_monthly(self, page_size: int = 100) -> pd.DataFrame:  # noqa: ARG002
        return self._read("trade_monthly.csv")

    def index_daily(
        self, code: str = "000300", start: str = "2019-01-01", end: str = "20500101"
    ) -> pd.DataFrame:  # noqa: ARG002
        df = self._read("market_daily.csv")
        return df[df["index_code"] == code].reset_index(drop=True)

    def fx_daily(self, pair: str = "USDCNH", start: str = "2019-01-01") -> pd.DataFrame:  # noqa: ARG002
        df = self._read("fx_daily.csv")
        return df[df["pair"] == pair.upper()].reset_index(drop=True)

    def fast_news(self, pages: int = 20, page_size: int = 100) -> pd.DataFrame:  # noqa: ARG002
        df = self._read("news.csv")
        df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
        return df.dropna(subset=["publish_time"]).reset_index(drop=True)
