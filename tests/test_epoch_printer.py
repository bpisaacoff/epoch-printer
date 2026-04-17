"""Tests for epoch-printer core behavior."""

from __future__ import annotations

import math
import pytest

from epoch_printer import (
    EpochPrinter,
    EpochTablePrinter,
    MetricCol,
    build_default_epoch_columns,
    ensure_time_last,
    make_epoch_columns,
    make_time_column,
)


# ---------------------------------------------------------------------------
# MetricCol
# ---------------------------------------------------------------------------

def test_metric_col_defaults():
    col = MetricCol("loss", "Loss")
    assert col.width == 10
    assert col.fmt == ".4f"
    assert col.align == ">"
    assert col.transform is None


def test_metric_col_frozen():
    col = MetricCol("loss", "Loss")
    with pytest.raises(Exception):
        col.width = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EpochTablePrinter – header
# ---------------------------------------------------------------------------

def test_header_str_alignment():
    cols = [
        MetricCol("epoch", "Ep", width=4, fmt="d"),
        MetricCol("loss", "Loss", width=8, fmt=".4f"),
    ]
    p = EpochTablePrinter(cols)
    header = p.header_str()
    assert "  Ep" in header
    assert "    Loss" in header


def test_rule_length_matches_header():
    cols = [MetricCol("a", "Alpha", width=6), MetricCol("b", "Beta", width=5)]
    p = EpochTablePrinter(cols)
    assert len(p.rule_str()) == len(p.header_str())


def test_format_header_contains_rule():
    p = EpochTablePrinter([MetricCol("x", "X", width=5)])
    fh = p.format_header()
    assert "\n" in fh
    assert "-" in fh


# ---------------------------------------------------------------------------
# EpochTablePrinter – row formatting
# ---------------------------------------------------------------------------

def test_format_row_float():
    cols = [MetricCol("loss", "Loss", width=10, fmt=".4f")]
    p = EpochTablePrinter(cols)
    result = p.format_row({"loss": 0.1234})
    assert "0.1234" in result


def test_format_row_int():
    cols = [MetricCol("epoch", "Ep", width=4, fmt="d")]
    p = EpochTablePrinter(cols)
    result = p.format_row({"epoch": 7})
    assert "7" in result


def test_format_row_missing_value():
    cols = [MetricCol("score", "Score", width=8, fmt=".3f")]
    p = EpochTablePrinter(cols, missing="--")
    result = p.format_row({})
    assert "--" in result


def test_format_row_none_value():
    cols = [MetricCol("score", "Score", width=8, fmt=".3f")]
    p = EpochTablePrinter(cols, missing="N/A")
    result = p.format_row({"score": None})
    assert "N/A" in result


def test_format_row_nan():
    cols = [MetricCol("x", "X", width=6, fmt=".3f")]
    p = EpochTablePrinter(cols)
    result = p.format_row({"x": float("nan")})
    assert "nan" in result


def test_format_row_pos_inf():
    cols = [MetricCol("x", "X", width=6, fmt=".3f")]
    p = EpochTablePrinter(cols)
    result = p.format_row({"x": float("inf")})
    assert "inf" in result


def test_format_row_neg_inf():
    cols = [MetricCol("x", "X", width=6, fmt=".3f")]
    p = EpochTablePrinter(cols)
    result = p.format_row({"x": float("-inf")})
    assert "-inf" in result


def test_format_row_string_column():
    cols = [MetricCol("phase", "Phase", width=10, fmt="s")]
    p = EpochTablePrinter(cols)
    result = p.format_row({"phase": "warmup"})
    assert "warmup" in result


def test_format_row_string_truncation():
    cols = [MetricCol("notes", "Notes", width=6, fmt="s")]
    p = EpochTablePrinter(cols)
    result = p.format_row({"notes": "this is a very long annotation"})
    assert len(result) == 6
    assert "\u2026" in result


def test_format_row_transform():
    cols = [
        MetricCol("mem_mb", "GPU_GB", width=8, fmt=".2f", transform=lambda x: x / 1024)
    ]
    p = EpochTablePrinter(cols)
    result = p.format_row({"mem_mb": 8192})
    assert "8.00" in result


def test_format_row_transform_applied_before_special_float_check():
    cols = [MetricCol("x", "X", width=8, fmt=".3f", transform=lambda v: float("nan"))]
    p = EpochTablePrinter(cols)
    result = p.format_row({"x": 1.0})
    assert "nan" in result


def test_sep_appears_between_columns():
    cols = [MetricCol("a", "A", width=3, fmt="d"), MetricCol("b", "B", width=3, fmt="d")]
    p = EpochTablePrinter(cols, sep=" | ")
    result = p.format_row({"a": 1, "b": 2})
    assert " | " in result


# ---------------------------------------------------------------------------
# EpochPrinter alias
# ---------------------------------------------------------------------------

def test_epoch_printer_alias():
    assert EpochPrinter is EpochTablePrinter


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def test_make_time_column():
    tc = make_time_column()
    assert tc.key == "time"
    assert tc.width == 8
    assert tc.fmt == ".2f"


def test_ensure_time_last_moves_time():
    cols = [
        MetricCol("time", "Time", width=8, fmt=".2f"),
        MetricCol("loss", "Loss", width=10, fmt=".4f"),
    ]
    result = ensure_time_last(cols)
    assert result[-1].key == "time"
    assert result[0].key == "loss"


def test_ensure_time_last_no_time():
    cols = [MetricCol("epoch", "Ep", width=4, fmt="d")]
    result = ensure_time_last(cols)
    assert result[-1].key == "time"
    assert len(result) == 2


def test_ensure_time_last_custom_time_col():
    custom = MetricCol("time", "T(s)", width=6, fmt=".1f")
    cols = [MetricCol("loss", "Loss", width=10, fmt=".4f")]
    result = ensure_time_last(cols, time_col=custom)
    assert result[-1].title == "T(s)"


def test_build_default_epoch_columns_time_last():
    cols = build_default_epoch_columns(include_time=True)
    assert cols[-1].key == "time"


def test_build_default_epoch_columns_no_time():
    cols = build_default_epoch_columns(include_time=False)
    assert all(col.key != "time" for col in cols)


def test_make_epoch_columns_extra_before_time():
    extra = [MetricCol("custom", "Custom", width=8, fmt=".3f")]
    cols = make_epoch_columns(extra_cols=extra, include_time=True)
    keys = [col.key for col in cols]
    assert keys[-1] == "time"
    assert "custom" in keys
    assert keys.index("custom") < keys.index("time")


def test_make_epoch_columns_no_time():
    cols = make_epoch_columns(include_time=False)
    assert all(col.key != "time" for col in cols)


# ---------------------------------------------------------------------------
# Optional pandas helpers
# ---------------------------------------------------------------------------

def test_results_to_dataframe_no_pandas(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pandas", None)  # type: ignore[arg-type]
    from importlib import reload
    import epoch_printer.helpers as h
    reload(h)

    with pytest.raises(ImportError, match="pandas"):
        h.results_to_dataframe([{"a": 1}])


try:
    import pandas  # noqa: F401
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


@pytest.mark.skipif(not _PANDAS_AVAILABLE, reason="pandas not installed")
def test_results_to_dataframe_basic():
    from epoch_printer import results_to_dataframe

    rows = [{"epoch": 1, "loss": 0.5}, {"epoch": 2, "loss": 0.3}]
    df = results_to_dataframe(rows)
    assert list(df.columns) == ["epoch", "loss"]
    assert len(df) == 2


@pytest.mark.skipif(not _PANDAS_AVAILABLE, reason="pandas not installed")
def test_style_results_dataframe_returns_styler():
    from epoch_printer import results_to_dataframe, style_results_dataframe
    import pandas as pd

    rows = [{"epoch": 1, "lr": 1e-3, "train_loss": 0.2, "time": 10.0}]
    df = results_to_dataframe(rows)
    styler = style_results_dataframe(df)
    assert isinstance(styler, pd.io.formats.style.Styler)


@pytest.mark.skipif(not _PANDAS_AVAILABLE, reason="pandas not installed")
def test_style_results_dataframe_unknown_columns_ignored():
    from epoch_printer import results_to_dataframe, style_results_dataframe

    rows = [{"epoch": 1, "my_custom_metric": 0.99}]
    df = results_to_dataframe(rows)
    # Should not raise even though 'my_custom_metric' is not in the format map
    style_results_dataframe(df)
