"""epoch-printer: Lightweight fixed-width metric table printer for ML training loops."""

from .core import EpochPrinter, EpochTablePrinter, MetricCol
from .helpers import (
    build_default_epoch_columns,
    ensure_time_last,
    make_epoch_columns,
    make_time_column,
    results_to_dataframe,
    style_results_dataframe,
)

__all__ = [
    "MetricCol",
    "EpochTablePrinter",
    "EpochPrinter",
    "make_time_column",
    "ensure_time_last",
    "build_default_epoch_columns",
    "make_epoch_columns",
    "results_to_dataframe",
    "style_results_dataframe",
]

__version__ = "0.1.0"
