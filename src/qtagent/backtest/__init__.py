"""回测子包。"""

from .engine import BacktestEngine, BacktestResult
from .metrics import compute_metrics, format_metrics
from .report import build_markdown, plot_equity

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "compute_metrics",
    "format_metrics",
    "build_markdown",
    "plot_equity",
]
