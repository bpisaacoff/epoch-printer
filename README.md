# epoch-printer

Lightweight fixed-width metric table printer for ML training loops.

```
  Ep          LR    TrainLoss      ValLoss      T_MAE      V_MAE     T_RMSE     V_RMSE   V_Corr   V_ExpVar      Time
-----------------------------------------------------------------------------------------------------------------
   1    1.0000e-03  1.8421e-01  2.1077e-01  1.920e-02  2.150e-02  4.510e-02  4.910e-02    0.781      0.612     12.47
   2    7.5000e-04  1.2376e-01  1.5542e-01  1.480e-02  1.740e-02  3.730e-02  4.110e-02    0.829      0.691     11.93
```

## What it is

`epoch-printer` gives you a simple, reusable way to turn dictionaries of metrics into stable, readable epoch-by-epoch output in terminals, notebooks, scripts, and IDE consoles.

Two core ideas:

- `MetricCol` describes how each metric should be displayed.
- `EpochTablePrinter` renders a consistent fixed-width table from those column definitions.

## What problem it solves

In iterative R&D, experiment logging usually falls into one of three unsatisfying patterns:

1. Raw `print()` calls become noisy and inconsistent.
2. Full experiment-tracking platforms are too heavy for quick iteration.
3. Notebook displays look fine after the run but don't help much while it is happening.

`epoch-printer` sits in the middle. It gives you:

- fixed-width aligned output that is easy to scan while training is running
- a reusable schema instead of hand-formatting each line
- graceful handling of missing metrics, `nan`, and `inf`
- custom transforms and per-metric formatting
- a path from live console printing to notebook-friendly DataFrame summaries

## When it is useful

Any time you have repeated metric updates across epochs, steps, or evaluation cycles:

- supervised learning training loops
- validation runs and fine-tuning experiments
- ablation studies and hyperparameter sweeps
- notebook-based experimentation
- IDE console workflows where readability matters

Although the name says "epoch," the same pattern works for optimization steps, evaluation checkpoints, curriculum stages, and outer-loop search iterations.

## Installation

```bash
pip install epoch-printer
```

With optional pandas support for DataFrame helpers:

```bash
pip install "epoch-printer[pandas]"
```

## Quick start

```python
from epoch_printer import EpochPrinter, make_epoch_columns

printer = EpochPrinter(make_epoch_columns())
printer.print_header()

for epoch in range(1, num_epochs + 1):
    metrics = train_one_epoch(...)   # returns a dict
    printer.print_row(metrics)
```

## Usage examples

### Standard training loop

```python
from epoch_printer import EpochPrinter, MetricCol, make_epoch_columns

printer = EpochPrinter(
    make_epoch_columns([
        MetricCol("val_mape", "V_MAPE", width=9, fmt=".2%"),
        MetricCol("val_acc_1e_2", "V_Acc1e-2", width=10, fmt=".2%"),
    ])
)

printer.print_header()

for epoch in range(1, num_epochs + 1):
    row = {"epoch": epoch, "lr": scheduler.get_lr(), "train_loss": ..., "time": ...}
    printer.print_row(row)
```

### Custom column schema

```python
from epoch_printer import EpochTablePrinter, MetricCol, ensure_time_last

columns = ensure_time_last([
    MetricCol("epoch",      "Ep",        width=4,  fmt="d"),
    MetricCol("lr",         "LR",        width=10, fmt=".2e"),
    MetricCol("train_loss", "TrainLoss", width=12, fmt=".4e"),
    MetricCol("val_loss",   "ValLoss",   width=12, fmt=".4e"),
    MetricCol("grad_norm",  "GradNorm",  width=9,  fmt=".3f"),
    MetricCol("clip_frac",  "ClipFrac",  width=9,  fmt=".2%"),
    MetricCol("throughput", "Samples/s", width=10, fmt=".0f"),
])

printer = EpochTablePrinter(columns)
printer.print_header()
printer.print_row(row)
```

### Value transforms

Apply a function to a raw value before it is formatted. Useful for unit conversions or derived statistics:

```python
# Display GPU memory in GiB even though the raw value is in MiB
MetricCol("gpu_mem_mb", "GPU_GB", width=8, fmt=".2f", transform=lambda x: x / 1024)
```

### Missing values, nan, and inf

If a key is absent from the row dict, the printer inserts a placeholder (`--` by default).
`nan` and `inf` are rendered as the strings `"nan"`, `"inf"`, and `"-inf"` rather than raising.

```python
printer.print_row({"epoch": 1})                        # missing keys show "--"
printer.print_row({"epoch": 2, "score": float("nan")}) # shows "nan"
```

### Notebook workflow

Collect rows during training, then convert to a styled DataFrame at the end:

```python
from epoch_printer import results_to_dataframe, style_results_dataframe

results = []  # accumulate row dicts during training

df = results_to_dataframe(results)
style_results_dataframe(df)   # renders as a styled HTML table in Jupyter
```

## API overview

### `MetricCol`

A frozen dataclass that defines one column.

| Field | Default | Description |
|-------|---------|-------------|
| `key` | — | Key expected in each row dict |
| `title` | `None` | Header label; defaults to `key` if omitted |
| `width` | `None` | Fixed display width; defaults to `len(title)` if omitted |
| `fmt` | `".4f"` | Python format specifier |
| `align` | `">"` | Alignment: `">"`, `"<"`, or `"^"` |
| `transform` | `None` | Optional callable applied before formatting |

### `EpochTablePrinter` / `EpochPrinter`

`EpochPrinter` is an alias for `EpochTablePrinter`.

```python
printer = EpochTablePrinter(columns, sep="  ", missing="--")
printer.print_header()
printer.print_row(row)

# String forms without side effects:
printer.format_header()
printer.format_row(row)
```

**Constructor parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `columns` | — | Ordered sequence of `MetricCol` definitions |
| `sep` | `"  "` | Separator string between columns |
| `missing` | `"--"` | Placeholder for absent keys |
| `header_rule` | `True` | Print a dashed rule beneath the header row |
| `vertical_lines` | `False` | Print `\|` separators between columns |
| `row_lines` | `False` | Print a dashed rule after every data row |

```python
# Example with all line options enabled:
printer = EpochTablePrinter(columns, vertical_lines=True, row_lines=True)
```

### Column helpers

| Function | Description |
|----------|-------------|
| `make_time_column()` | Returns the default `"time"` column |
| `ensure_time_last(cols)` | Moves/appends the time column to the end |
| `build_default_epoch_columns()` | Full default schema: epoch, LR, loss, MAE, RMSE, corr, explained var, time |
| `make_epoch_columns(extra_cols)` | Default schema plus your extra columns, time last |

### DataFrame helpers (optional pandas)

| Function | Description |
|----------|-------------|
| `results_to_dataframe(rows)` | Converts a list of row dicts to a DataFrame |
| `style_results_dataframe(df)` | Returns a pandas Styler with standard metric formatting |

These require pandas. Install with `pip install pandas` or `pip install "epoch-printer[pandas]"`. Calling them without pandas installed raises a clear `ImportError`.

## Design philosophy

- **No required dependencies** — standard library only for core functionality.
- **Portable** — works in scripts, notebooks, and IDE consoles without configuration.
- **Explicit** — every displayed metric is declared in a column schema; nothing is inferred silently.
- **Composable** — start with defaults and extend with project-specific columns.
- **Stable** — columns never shift position from row to row.
- **Small** — a focused utility, not a platform.

## License

MIT
