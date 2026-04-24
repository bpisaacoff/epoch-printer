# epoch-printer

Readable fixed-width epoch tables for ML training loops, with no required dependencies.

```
  Ep          LR    TrainLoss      ValLoss      T_MAE      V_MAE     T_RMSE     V_RMSE   V_Corr   V_ExpVar      Time
-----------------------------------------------------------------------------------------------------------------
   1    1.0000e-03  1.8421e-01  2.1077e-01  1.920e-02  2.150e-02  4.510e-02  4.910e-02    0.781      0.612     12.47
   2    7.5000e-04  1.2376e-01  1.5542e-01  1.480e-02  1.740e-02  3.730e-02  4.110e-02    0.829      0.691     11.93
   3    5.0000e-04  9.1040e-02  1.2010e-01  1.210e-02  1.480e-02  3.140e-02  3.720e-02    0.863      0.744     11.64
```

## Why use this?

Training loop output usually falls into one of three patterns:

1. `print(f"epoch {epoch}, loss {loss:.4f}")` — fine for 2 metrics, messy for 10.
2. Full experiment trackers (W&B, MLflow, TensorBoard) — great for dashboards, overkill for a terminal window.
3. Custom formatting code — rewritten per project, fragile when the metric set changes.

`epoch-printer` gives you a fixed-width table that is easy to scan while training is running, defined once per project, and portable across scripts, notebooks, and IDE consoles. It handles missing values, `nan`/`inf`, and column alignment automatically.

This is **not a replacement for experiment trackers**. Use it alongside W&B or MLflow for readable stdout while your dashboard updates in the background.

## Installation

```bash
pip install epoch-printer
```

With optional pandas support for DataFrame summaries:

```bash
pip install "epoch-printer[pandas]"
```

## 30-second example

Paste into an existing training loop:

```python
import time
from epoch_printer import EpochPrinter, make_epoch_columns

printer = EpochPrinter(make_epoch_columns())
printer.print_header()

for epoch in range(1, num_epochs + 1):
    t0 = time.time()
    train_metrics = train_one_epoch(model, loader, optimizer)
    val_metrics   = evaluate(model, val_loader)
    scheduler.step()

    printer.print_row({
        "epoch":      epoch,
        "lr":         scheduler.get_last_lr()[0],
        "train_loss": train_metrics["loss"],
        "val_loss":   val_metrics["loss"],
        "val_mae":    val_metrics["mae"],
        "time":       time.time() - t0,
    })
```

`make_epoch_columns()` provides a default schema (epoch, LR, loss, MAE, RMSE, correlation, explained variance, time). Keys absent from the row dict render as `--`.

## Common workflows

### PyTorch / JAX-style training loop

See [`examples/pytorch_training_loop.py`](examples/pytorch_training_loop.py) for a self-contained example shaped like a real PyTorch training run (no PyTorch required — model and loader are mocked).

To append project-specific columns alongside the defaults:

```python
from epoch_printer import EpochPrinter, MetricCol, make_epoch_columns

printer = EpochPrinter(
    make_epoch_columns([
        MetricCol("val_acc",   "V_Acc",   width=8, fmt=".2%"),
        MetricCol("grad_norm", "GradNorm", width=9, fmt=".3f"),
    ])
)
printer.print_header()
```

### Custom research metrics

When the default schema does not match your experiment, build your own column list:

```python
from epoch_printer import EpochTablePrinter, MetricCol, ensure_time_last

columns = ensure_time_last([
    MetricCol("epoch",      "Ep",       width=4,  fmt="d"),
    MetricCol("lr",         "LR",       width=10, fmt=".2e"),
    MetricCol("recon_loss", "Recon",    width=12, fmt=".4e"),
    MetricCol("kl_div",     "KL",       width=10, fmt=".4e"),
    MetricCol("grad_norm",  "GradNorm", width=9,  fmt=".3f"),
    MetricCol("throughput", "Samp/s",   width=8,  fmt=".0f"),
])

printer = EpochTablePrinter(columns)
printer.print_header()
```

Apply a transform to convert raw values before display (e.g. MiB → GiB):

```python
MetricCol("gpu_mem_mb", "GPU_GiB", width=8, fmt=".2f", transform=lambda x: x / 1024)
```

See [`examples/custom_research_metrics.py`](examples/custom_research_metrics.py) for more patterns including VAE training and table formatting options.

### Notebook summaries

Collect rows during training, then convert to a styled DataFrame at the end of the cell:

```python
from epoch_printer import (
    EpochPrinter, make_epoch_columns,
    results_to_dataframe, style_results_dataframe,
)

results = []
printer = EpochPrinter(make_epoch_columns())
printer.print_header()

for epoch in range(1, num_epochs + 1):
    row = {"epoch": epoch, "lr": ..., "train_loss": ..., "val_loss": ..., "time": ...}
    printer.print_row(row)
    results.append(row)

# At end of notebook cell — renders as styled HTML table in Jupyter:
style_results_dataframe(results_to_dataframe(results))
```

See [`examples/notebook_summary.py`](examples/notebook_summary.py) for a runnable version.

### Alongside W&B / TensorBoard / MLflow

`epoch-printer` only writes to stdout. Pass the same row dict to your experiment tracker:

```python
import wandb
from epoch_printer import EpochPrinter, make_epoch_columns

wandb.init(project="my-experiment")
printer = EpochPrinter(make_epoch_columns())
printer.print_header()

for epoch in range(1, num_epochs + 1):
    row = {"epoch": epoch, "lr": ..., "train_loss": ..., "val_loss": ..., "time": ...}
    wandb.log(row, step=epoch)   # to W&B
    printer.print_row(row)       # to stdout
```

See [`examples/logger_integration.py`](examples/logger_integration.py) for a version that also mocks TensorBoard and MLflow log calls.

## API overview

### `MetricCol`

Frozen dataclass defining one column.

| Field | Default | Description |
|-------|---------|-------------|
| `key` | — | Key expected in each row dict |
| `title` | `None` | Header label; defaults to `key` if omitted |
| `width` | `None` | Fixed display width; defaults to `len(title)` if omitted |
| `fmt` | `".4f"` | Python format specifier (e.g. `"d"`, `".2e"`, `".2%"`, `"s"`) |
| `align` | `">"` | Alignment: `">"` (right), `"<"` (left), or `"^"` (center) |
| `transform` | `None` | Optional callable applied to the raw value before formatting |

### `EpochTablePrinter` / `EpochPrinter`

`EpochPrinter` is an alias for `EpochTablePrinter`.

```python
printer = EpochTablePrinter(columns, sep="  ", missing="--")
printer.print_header()     # print header + rule to stdout
printer.print_row(row)     # print one formatted row

# Non-printing string forms:
printer.format_header()    # returns header string
printer.format_row(row)    # returns formatted row string
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `columns` | — | Ordered sequence of `MetricCol` definitions |
| `sep` | `"  "` | Separator between columns |
| `missing` | `"--"` | Placeholder for absent or `None` keys |
| `header_rule` | `True` | Dashed rule beneath the header |
| `vertical_lines` | `False` | `" | "` separators between columns |
| `row_lines` | `False` | Dashed rule after every data row |

### Column helpers

| Function | Description |
|----------|-------------|
| `make_time_column()` | Returns the default `"time"` column (width 8, `.2f`) |
| `ensure_time_last(cols)` | Moves or appends the time column to the end |
| `build_default_epoch_columns()` | Full default schema: epoch, LR, loss, MAE, RMSE, corr, explained var, time |
| `make_epoch_columns(extra_cols)` | Default schema plus your extra columns, time last |

### DataFrame helpers (requires pandas)

| Function | Description |
|----------|-------------|
| `results_to_dataframe(rows)` | Converts a list of row dicts to a DataFrame |
| `style_results_dataframe(df)` | Returns a pandas Styler with standard metric formatting |

Install with `pip install pandas` or `pip install "epoch-printer[pandas]"`.

## Design choices

- **No required dependencies.** Core functionality uses only the standard library.
- **Explicit schema.** Every displayed metric is declared in a column definition; nothing is inferred silently.
- **Stable column order.** Positions never shift from row to row, making it easy to scan vertically during a long run.
- **Row dicts, not keyword arguments.** The same dict you pass to `wandb.log` or `mlflow.log_metrics` works here without modification.
- **Portable.** Plain-text output works in terminals, notebooks, and IDE consoles without configuration.
- **Small.** A focused utility, not a platform.

## When not to use this

- You need **persistent experiment storage**, run comparison, or metric visualization → use W&B, MLflow, or TensorBoard.
- You have **hundreds of metrics** per row that need filtering, aggregation, or search → use a proper logging framework.
- You need **structured output** (JSON, CSV) for downstream processing → write to a file directly.
- Your runs take **under a minute** with two or three metrics → `print()` is fine.

## License

MIT · v0.1.0
