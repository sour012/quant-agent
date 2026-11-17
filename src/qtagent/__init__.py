"""量化交易 Agent 系统（qtagent）。

五 Agent 协作：数据采集 → 技术分析 / 新闻情绪 / 宏观贸易 → 风险管理 → 交易决策。
"""

from __future__ import annotations

import logging

__version__ = "0.1.0"

__all__ = ["__version__", "setup_logging"]


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
