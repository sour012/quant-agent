"""配置加载：把 YAML 转成带类型的访问器，并统一解析相对路径。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class Config:
    """项目级配置对象。

    设计意图：Agent 只依赖显式传入的参数，不读全局变量，
    这样回测、单元测试、参数搜索都能用不同配置实例并行运行。
    """

    def __init__(self, raw: dict[str, Any], root: Path | None = None) -> None:
        if not isinstance(raw, dict):
            raise ValueError("配置文件内容格式错误，应为映射结构")
        self.raw = raw
        self.root = Path(root) if root else PROJECT_ROOT

    # ---------------- 构造 ----------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        with open(p, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls(raw, root=PROJECT_ROOT)

    # ---------------- 快捷访问 ----------------
    @property
    def data(self) -> dict[str, Any]:
        return self.raw.get("data", {})

    @property
    def backtest(self) -> dict[str, Any]:
        return self.raw.get("backtest", {})

    @property
    def output(self) -> dict[str, Any]:
        return self.raw.get("output", {})

    def agent(self, name: str) -> dict[str, Any]:
        return self.raw.get("agents", {}).get(name, {})

    @property
    def decision(self) -> dict[str, Any]:
        return self.agent("decision")

    @property
    def weights(self) -> dict[str, float]:
        return dict(self.decision.get("weights", {}))

    # ---------------- 路径 ----------------
    def path(self, rel: str | Path) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (self.root / p)

    @property
    def cache_dir(self) -> Path:
        d = self.path(self.data.get("cache_dir", "data/raw"))
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def report_dir(self) -> Path:
        d = self.path(self.output.get("report_dir", "reports"))
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def db_url(self) -> str:
        """SQLite 的相对路径按项目根目录解析，MySQL 原样返回。"""
        url = self.raw.get("db", {}).get("url", "sqlite:///data/qtagent.db")
        prefix = "sqlite:///"
        if url.startswith(prefix):
            rel = url[len(prefix):]
            if not Path(rel).is_absolute():
                p = self.root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                return prefix + p.as_posix()
        return url
