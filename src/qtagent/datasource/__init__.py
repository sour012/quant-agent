"""数据源工厂：按配置返回统一的取数客户端。"""

from __future__ import annotations

from ..config import Config
from .eastmoney import EastMoneyClient
from .snapshot import SnapshotClient

__all__ = ["EastMoneyClient", "SnapshotClient", "build_client"]


def build_client(cfg: Config):
    provider = (cfg.data.get("provider") or "eastmoney").lower()
    if provider == "snapshot":
        return SnapshotClient(cfg.path("data/snapshot"))
    if provider == "eastmoney":
        return EastMoneyClient()
    raise ValueError(f"未知的数据源 provider: {provider}")
