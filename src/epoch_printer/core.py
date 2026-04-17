from __future__ import annotations

from dataclasses import dataclass
from math import isinf, isnan
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class MetricCol:
    """
    Column specification for a printed metrics table.

    Parameters
    ----------
    key:
        Key expected in each row mapping.
    title:
        Header label to display.
    width:
        Fixed display width for this column.
    fmt:
        Format specifier applied with Python's format mini-language.
    align:
        Alignment character: ``">"``, ``"<"``, or ``"^"``.
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
        """Return the header line as a string."""
        parts = [f"{col.title:{col.align}{col.width}}" for col in self.columns]
        return self.sep.join(parts)

    def rule_str(self) -> str:
        """Return a horizontal rule matching the header width."""
        return "-" * len(self.header_str())

    def format_header(self) -> str:
        """Return the header and rule as a single string."""
        return f"{self.header_str()}\n{self.rule_str()}"

    def format_row(self, row: Mapping[str, Any]) -> str:
        """Format a row mapping as a fixed-width string."""
        parts = [self._format_cell(col, row.get(col.key)) for col in self.columns]
        return self.sep.join(parts)

    def print_header(self) -> None:
        """Print the header and rule."""
        print(self.format_header())

    def print_row(self, row: Mapping[str, Any]) -> None:
        """Format and print one row."""
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

        # Python format specs don't truncate strings; enforce width manually.
        if isinstance(value, str) and len(value) > col.width:
            value = value[: max(1, col.width - 1)] + "\u2026"

        try:
            return f"{value:{col.align}{col.width}{col.fmt}}"
        except (ValueError, TypeError):
            text = str(value)
            if len(text) > col.width:
                text = text[: max(1, col.width - 1)] + "\u2026"
            return f"{text:{col.align}{col.width}}"


#: Friendly alias for :class:`EpochTablePrinter`.
EpochPrinter = EpochTablePrinter
