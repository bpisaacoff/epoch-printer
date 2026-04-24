from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .core import MetricCol


def make_time_column() -> MetricCol:
    """Return the default time column, suitable as the rightmost column."""
    return MetricCol("time", "Time", width=8, fmt=".2f")


def ensure_time_last(
    columns: Sequence[MetricCol],
    time_col: MetricCol | None = None,
) -> list[MetricCol]:
    """
    Return *columns* with any existing ``time`` column moved to the end.

    Safe to call on lists that already include a time column or that do not.
    When composing columns from reusable pieces, this guarantees the time
    column always appears at the far right.

    Parameters
    ----------
    columns:
        Input column list, optionally already containing a ``time`` column.
    time_col:
        Custom time column to append. Defaults to :func:`make_time_column`.
    """
    time_col = time_col or make_time_column()
    non_time = [col for col in columns if col.key != "time"]
    return [*non_time, time_col]


def build_default_epoch_columns(include_time: bool = True) -> list[MetricCol]:
    """
    Return a default epoch table schema for common supervised-learning workflows.

    Covers epoch index, learning rate, train/val loss, MAE, RMSE, correlation,
    and explained variance. A time column is appended last when *include_time*
    is ``True``.
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
    Build an epoch column list starting from the default schema.

    Appends *extra_cols* (if any) after the default columns, then places the
    time column last when *include_time* is ``True``.

    Parameters
    ----------
    extra_cols:
        Additional project-specific columns to append.
    include_time:
        Whether to append a time column at the end.
    time_col:
        Custom time column. Defaults to :func:`make_time_column`.
    """
    cols = build_default_epoch_columns(include_time=False)
    if extra_cols:
        cols.extend(extra_cols)
    if include_time:
        cols = ensure_time_last(cols, time_col=time_col)
    return cols


def results_to_dataframe(results: Iterable[Mapping[str, Any]]):
    """
    Convert a sequence of row dicts into a pandas DataFrame.

    Requires pandas. Install with ``pip install "epoch-printer[pandas]"``.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for results_to_dataframe. "
            'Install it with: pip install "epoch-printer[pandas]"'
        ) from exc
    return pd.DataFrame(results)


def style_results_dataframe(df):
    """
    Apply standard metric formatting to a DataFrame for notebook display.

    Only formats known metric columns; all other columns are left unchanged.
    Returns a pandas ``Styler`` object with the index hidden.

    Requires pandas with Styler support (pandas >= 1.3).
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
