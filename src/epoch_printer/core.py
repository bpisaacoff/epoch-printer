from __future__ import annotations

from dataclasses import dataclass
from math import isinf, isnan
from typing import Any, Callable, Mapping, Sequence


def _fmt_min_width(fmt: str) -> int:
    for probe in (0.0, 0):
        try:
            return len(format(probe, fmt))
        except (ValueError, TypeError):
            continue
    return 0


@dataclass(frozen=True)
class MetricCol:
    """
    Column specification for a printed metrics table.

    Parameters
    ----------
    key:
        Key expected in each row mapping.
    title:
        Header label to display. Defaults to ``key`` when omitted.
    width:
        Fixed display width for this column. Defaults to the larger of
        ``len(effective_title)`` and the minimum width implied by the format
        spec, so both the header and a zero-valued cell always fit.
    fmt:
        Format specifier applied with Python's format mini-language.
    align:
        Alignment character: ``">"``, ``"<"``, or ``"^"``.
    transform:
        Optional callable applied to the raw value before formatting.
    """

    key: str
    title: str | None = None
    width: int | None = None
    fmt: str = ".4f"
    align: str = ">"
    transform: Callable[[Any], Any] | None = None

    @property
    def effective_title(self) -> str:
        return self.key if self.title is None else self.title

    @property
    def effective_width(self) -> int:
        if self.width is not None:
            return self.width
        return max(len(self.effective_title), _fmt_min_width(self.fmt))


class EpochTablePrinter:
    """
    Lightweight fixed-width printer for epoch- or step-based experiment metrics.

    Define columns once, then pass row dictionaries as metrics are produced.
    Output is stable and aligned across all rows, making it easy to scan
    during long training runs in terminals, notebooks, or IDE consoles.

    Parameters
    ----------
    columns:
        Ordered sequence of :class:`MetricCol` definitions.
    sep:
        Separator string between columns.
    missing:
        Placeholder displayed when a key is absent from a row.
    header_rule:
        Print a dashed horizontal rule beneath the header row.
    vertical_lines:
        Print ``" | "`` separators between columns instead of ``sep``.
    row_lines:
        Print a dashed horizontal rule after every data row.
    """

    def __init__(
        self,
        columns: Sequence[MetricCol],
        sep: str = "  ",
        missing: str = "--",
        header_rule: bool = True,
        vertical_lines: bool = False,
        row_lines: bool = False,
    ) -> None:
        self.columns = list(columns)
        self.sep = sep
        self.missing = missing
        self.header_rule = header_rule
        self.vertical_lines = vertical_lines
        self.row_lines = row_lines

    @property
    def _sep(self) -> str:
        return " | " if self.vertical_lines else self.sep

    def header_str(self) -> str:
        """Return the header line as a string."""
        parts = [f"{col.effective_title:{col.align}{col.effective_width}}" for col in self.columns]
        return self._sep.join(parts)

    def rule_str(self) -> str:
        """Return a horizontal rule matching the header width."""
        return "-" * len(self.header_str())

    def format_header(self) -> str:
        """Return the header (and optional rule) as a single string."""
        if self.header_rule:
            return f"{self.header_str()}\n{self.rule_str()}"
        return self.header_str()

    def format_row(self, row: Mapping[str, Any]) -> str:
        """Format a row mapping as a fixed-width string."""
        parts = [self._format_cell(col, row.get(col.key)) for col in self.columns]
        line = self._sep.join(parts)
        if self.row_lines:
            return f"{line}\n{self.rule_str()}"
        return line

    def print_header(self) -> None:
        """Print the header and rule."""
        print(self.format_header())

    def print_row(self, row: Mapping[str, Any]) -> None:
        """Format and print one row."""
        print(self.format_row(row))

    def _format_cell(self, col: MetricCol, value: Any) -> str:
        if value is None:
            return f"{self.missing:{col.align}{col.effective_width}}"

        if col.transform is not None:
            value = col.transform(value)

        if isinstance(value, float):
            if isnan(value):
                return f"{'nan':{col.align}{col.effective_width}}"
            if isinf(value):
                text = "inf" if value > 0 else "-inf"
                return f"{text:{col.align}{col.effective_width}}"

        # Python format specs don't truncate strings; enforce width manually.
        if isinstance(value, str) and len(value) > col.effective_width:
            value = value[: max(1, col.effective_width - 1)] + "\u2026"

        try:
            return f"{value:{col.align}{col.effective_width}{col.fmt}}"
        except (ValueError, TypeError):
            text = str(value)
            if len(text) > col.effective_width:
                text = text[: max(1, col.effective_width - 1)] + "\u2026"
            return f"{text:{col.align}{col.effective_width}}"


#: Friendly alias for :class:`EpochTablePrinter`.
EpochPrinter = EpochTablePrinter
