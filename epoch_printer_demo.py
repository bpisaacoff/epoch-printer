"""
EpochPrinter demo script
========================

This file contains:
1. A self-contained implementation of the core EpochPrinter API.
2. Helper utilities for common deep-learning metric layouts.
3. Runnable demos for scripts, notebooks, and custom experiment loops.

It is designed as a starting point for turning the utility into a small PyPI
package for iterative R&D workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isinf, isnan
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class MetricCol:
    """
    Column specification for a printed metrics table.

    Parameters
    ----------
    key:
        Key expected in the row mapping.
    title:
        Header text to display.
    width:
        Fixed display width for this column.
    fmt:
        Format specifier applied with Python's format mini-language.
    align:
        Alignment character, usually ">", "<", or "^".
    transform:
        Optional callable applied to the raw value before formatting.
    """

    key: str
    title: str
    width: int = 10
    fmt: str = ".4f"
    align: str = ">"
    transform: Callable[[Any], Any] | None = None


class EpochTablePrinter:
    """
    Lightweight fixed-width printer for epoch- or step-based experiment metrics.

    The printer is intentionally simple:
    - you define columns once,
    - pass in row dictionaries as metrics are produced,
    - get stable, aligned, human-readable output in any terminal or notebook.

    This is useful when you want something more structured than ad hoc print
    statements, but much lighter than a full experiment tracking stack.
    """

    def __init__(
        self,
        columns: Sequence[MetricCol],
        sep: str = "  ",
        missing: str = "--",
    ) -> None:
        self.columns = list(columns)
        self.sep = sep
        self.missing = missing

    def header_str(self) -> str:
        parts = [f"{col.title:{col.align}{col.width}}" for col in self.columns]
        return self.sep.join(parts)

    def rule_str(self) -> str:
        return "-" * len(self.header_str())

    def format_header(self) -> str:
        return f"{self.header_str()}\n{self.rule_str()}"

    def format_row(self, row: Mapping[str, Any]) -> str:
        parts = [self._format_cell(col, row.get(col.key, None)) for col in self.columns]
        return self.sep.join(parts)

    def print_header(self) -> None:
        print(self.format_header())

    def print_row(self, row: Mapping[str, Any]) -> None:
        print(self.format_row(row))

    def _format_cell(self, col: MetricCol, value: Any) -> str:
        if value is None:
            return f"{self.missing:{col.align}{col.width}}"

        if col.transform is not None:
            value = col.transform(value)

        if isinstance(value, float):
            if isnan(value):
                return f"{'nan':{col.align}{col.width}}"
            if isinf(value):
                text = "inf" if value > 0 else "-inf"
                return f"{text:{col.align}{col.width}}"

        try:
            return f"{value:{col.align}{col.width}{col.fmt}}"
        except (ValueError, TypeError):
            text = str(value)
            if len(text) > col.width:
                text = text[: max(1, col.width - 1)] + "…"
            return f"{text:{col.align}{col.width}}"


# Optional friendlier alias for package ergonomics.
EpochPrinter = EpochTablePrinter


def make_time_column() -> MetricCol:
    """Return the default rightmost time column."""
    return MetricCol("time", "Time", width=8, fmt=".2f")


def ensure_time_last(
    columns: Sequence[MetricCol],
    time_col: MetricCol | None = None,
) -> list[MetricCol]:
    """
    Return columns with any existing ``time`` column removed and re-appended last.

    This makes composition safe when you build columns from reusable pieces.
    """
    time_col = time_col or make_time_column()
    non_time = [col for col in columns if col.key != "time"]
    return [*non_time, time_col]



def build_default_epoch_columns(include_time: bool = True) -> list[MetricCol]:
    """
    Build a default deep-learning-oriented epoch table schema.

    The default schema targets common supervised-learning workflows and gives a
    practical baseline for train/validation monitoring.
    """
    base = [
        MetricCol("epoch", "Ep", width=4, fmt="d"),
        MetricCol("lr", "LR", width=10, fmt=".2e"),
        MetricCol("train_loss", "TrainLoss", width=12, fmt=".4e"),
        MetricCol("val_loss", "ValLoss", width=12, fmt=".4e"),
        MetricCol("train_mae", "T_MAE", width=10, fmt=".3e"),
        MetricCol("val_mae", "V_MAE", width=10, fmt=".3e"),
        MetricCol("train_rmse", "T_RMSE", width=10, fmt=".3e"),
        MetricCol("val_rmse", "V_RMSE", width=10, fmt=".3e"),
        MetricCol("val_corr", "V_Corr", width=8, fmt=".3f"),
        MetricCol("val_explained_var", "V_ExpVar", width=10, fmt=".3f"),
    ]

    if include_time:
        return ensure_time_last(base)
    return base



def make_epoch_columns(
    extra_cols: Sequence[MetricCol] | None = None,
    include_time: bool = True,
    time_col: MetricCol | None = None,
) -> list[MetricCol]:
    """
    Preferred helper for notebook and script usage.

    Start from a sensible default schema, then append project-specific metrics.
    """
    cols = build_default_epoch_columns(include_time=False)
    if extra_cols:
        cols.extend(extra_cols)

    if include_time:
        cols = ensure_time_last(cols, time_col=time_col)

    return cols



def results_to_dataframe(results: Iterable[Mapping[str, Any]]):
    """Convert a sequence of row dictionaries into a pandas DataFrame."""
    import pandas as pd

    return pd.DataFrame(results)



def style_results_dataframe(df):
    """
    Apply common metric formatting for notebook display.

    This helper is intentionally conservative: it formats only known columns and
    leaves everything else unchanged.
    """
    format_map = {
        "lr": "{:.2e}",
        "train_loss": "{:.4e}",
        "val_loss": "{:.4e}",
        "train_mae": "{:.3e}",
        "val_mae": "{:.3e}",
        "train_rmse": "{:.3e}",
        "val_rmse": "{:.3e}",
        "train_mape": "{:.2%}",
        "val_mape": "{:.2%}",
        "train_corr": "{:.3f}",
        "val_corr": "{:.3f}",
        "train_explained_var": "{:.3f}",
        "val_explained_var": "{:.3f}",
        "train_acc_1e_1": "{:.2%}",
        "val_acc_1e_1": "{:.2%}",
        "train_acc_5e_2": "{:.2%}",
        "val_acc_5e_2": "{:.2%}",
        "train_acc_1e_2": "{:.2%}",
        "val_acc_1e_2": "{:.2%}",
        "time": "{:.2f}",
    }

    available_map = {k: v for k, v in format_map.items() if k in df.columns}

    return df.style.format(available_map).hide(axis="index")


# -----------------------------------------------------------------------------
# Demo helpers
# -----------------------------------------------------------------------------

def _demo_basic_training_loop() -> None:
    print("\n=== Demo 1: Basic training loop ===")

    printer = EpochPrinter(
        make_epoch_columns(
            [
                MetricCol("val_mape", "V_MAPE", width=9, fmt=".2%"),
                MetricCol("val_acc_1e_2", "V_Acc1e-2", width=10, fmt=".2%"),
            ]
        )
    )

    rows = [
        {
            "epoch": 1,
            "lr": 1.00e-3,
            "train_loss": 1.8421e-1,
            "val_loss": 2.1077e-1,
            "train_mae": 1.92e-2,
            "val_mae": 2.15e-2,
            "train_rmse": 4.51e-2,
            "val_rmse": 4.91e-2,
            "val_corr": 0.781,
            "val_explained_var": 0.612,
            "val_mape": 0.0842,
            "val_acc_1e_2": 0.431,
            "time": 12.47,
        },
        {
            "epoch": 2,
            "lr": 7.50e-4,
            "train_loss": 1.2376e-1,
            "val_loss": 1.5542e-1,
            "train_mae": 1.48e-2,
            "val_mae": 1.74e-2,
            "train_rmse": 3.73e-2,
            "val_rmse": 4.11e-2,
            "val_corr": 0.829,
            "val_explained_var": 0.691,
            "val_mape": 0.0698,
            "val_acc_1e_2": 0.507,
            "time": 11.93,
        },
    ]

    printer.print_header()
    for row in rows:
        printer.print_row(row)



def _demo_custom_research_metrics() -> None:
    print("\n=== Demo 2: Custom research metrics ===")

    columns = ensure_time_last(
        [
            MetricCol("epoch", "Ep", width=4, fmt="d"),
            MetricCol("lr", "LR", width=10, fmt=".2e"),
            MetricCol("train_loss", "TrainLoss", width=12, fmt=".4e"),
            MetricCol("val_loss", "ValLoss", width=12, fmt=".4e"),
            MetricCol("grad_norm", "GradNorm", width=9, fmt=".3f"),
            MetricCol("clip_frac", "ClipFrac", width=9, fmt=".2%"),
            MetricCol("throughput", "Samples/s", width=10, fmt=".0f"),
            MetricCol("notes", "Notes", width=14, fmt="s"),
        ]
    )

    printer = EpochPrinter(columns)
    printer.print_header()

    printer.print_row(
        {
            "epoch": 7,
            "lr": 3e-4,
            "train_loss": 9.844e-2,
            "val_loss": 1.121e-1,
            "grad_norm": 1.734,
            "clip_frac": 0.08,
            "throughput": 18240,
            "notes": "stable",
            "time": 9.88,
        }
    )
    printer.print_row(
        {
            "epoch": 8,
            "lr": 2e-4,
            "train_loss": 8.912e-2,
            "val_loss": 1.094e-1,
            "grad_norm": 2.401,
            "clip_frac": 0.11,
            "throughput": 17902,
            "notes": "long annotation that will truncate",
            "time": 10.12,
        }
    )



def _demo_missing_nan_transform() -> None:
    print("\n=== Demo 3: Missing values, transforms, and special floats ===")

    printer = EpochPrinter(
        [
            MetricCol("epoch", "Ep", width=4, fmt="d"),
            MetricCol("phase", "Phase", width=10, fmt="s"),
            MetricCol("score", "Score", width=8, fmt=".3f"),
            MetricCol("error_rate", "Err", width=8, fmt=".2%"),
            MetricCol("gpu_mem_gb", "GPU_GB", width=8, fmt=".2f", transform=lambda x: x / 1024),
            MetricCol("time", "Time", width=8, fmt=".2f"),
        ]
    )

    printer.print_header()
    printer.print_row(
        {
            "epoch": 1,
            "phase": "warmup",
            "score": float("nan"),
            "error_rate": 0.184,
            "gpu_mem_gb": 24576,
            "time": 4.2,
        }
    )
    printer.print_row(
        {
            "epoch": 2,
            "phase": "train",
            "score": float("inf"),
            "gpu_mem_gb": 28672,
            "time": 4.4,
        }
    )
    printer.print_row(
        {
            "epoch": 3,
            "phase": "eval",
            "score": 0.913,
            "error_rate": 0.071,
            "gpu_mem_gb": 20480,
            "time": 3.9,
        }
    )



def _demo_notebook_dataframe() -> None:
    print("\n=== Demo 4: Notebook / DataFrame integration ===")

    results = [
        {"epoch": 1, "lr": 1e-3, "train_loss": 0.21, "val_loss": 0.25, "time": 12.3},
        {"epoch": 2, "lr": 8e-4, "train_loss": 0.17, "val_loss": 0.20, "time": 11.8},
        {"epoch": 3, "lr": 6e-4, "train_loss": 0.14, "val_loss": 0.18, "time": 11.1},
    ]

    df = results_to_dataframe(results)
    print(df)
    print("\nIn a notebook, call: style_results_dataframe(df)")


if __name__ == "__main__":
    _demo_basic_training_loop()
    _demo_custom_research_metrics()
    _demo_missing_nan_transform()
    _demo_notebook_dataframe()
