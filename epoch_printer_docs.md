# EpochPrinter

## What it is

EpochPrinter is a lightweight utility for printing aligned metric tables during iterative machine learning and deep learning experiments. It gives you a simple, reusable way to turn dictionaries of metrics into stable, readable epoch-by-epoch output in terminals, notebooks, scripts, and IDE consoles.

At its core, it has two ideas:

- `MetricCol` describes how each metric should be displayed.
- `EpochTablePrinter` renders a consistent fixed-width table from those column definitions.

It also includes a small set of helpers for common R&D workflows:

- `make_time_column()`
- `ensure_time_last()`
- `build_default_epoch_columns()`
- `make_epoch_columns()`
- `results_to_dataframe()`
- `style_results_dataframe()`

## What problem it solves

In iterative R&D, experiment logging usually falls into one of three unsatisfying patterns:

1. Raw `print()` statements become noisy and inconsistent.
2. Full experiment-tracking platforms are too heavy for quick iteration.
3. Notebook displays look fine after the run, but don’t help much while the run is happening.

EpochPrinter sits in the middle.

It gives you:

- fixed-width aligned output that is easy to scan while training is running
- a reusable schema for metrics, instead of hand-formatting each line
- graceful handling of missing metrics, `nan`, and `inf`
- support for custom transforms and formatting per metric
- a path from live console printing to notebook-friendly DataFrame summaries

## When it is useful

EpochPrinter is useful whenever you have repeated metric updates across epochs, steps, iterations, or evaluation cycles.

Typical cases:

- supervised learning training loops
- validation runs during deep learning experiments
- fine-tuning workflows
- ablation studies
- hyperparameter sweeps run from scripts
- notebook-based experimentation where you still want clean live logs
- IDE console workflows where readability matters

Although the name says “epoch,” the pattern also works for:

- optimization steps
- evaluation checkpoints
- outer-loop search iterations
- curriculum stages
- dataset passes

## Why this is a good fit for deep learning R&D

Deep learning workflows often produce a mix of metrics with different scales and meanings: losses, accuracies, correlations, learning rates, gradient diagnostics, timings, and custom research metrics. A useful printer for this setting needs to be flexible without becoming heavy.

This design works well because it is:

- **portable**: no backend service required
- **explicit**: every displayed metric is declared in a column schema
- **composable**: start with defaults and add project-specific columns
- **stable**: columns do not shift around from row to row
- **notebook-friendly**: rows can also be collected into a DataFrame

## Main API

### `MetricCol`

A dataclass that defines one printed column.

Fields:

- `key`: metric key expected in each row dictionary
- `title`: header label
- `width`: display width
- `fmt`: Python format specifier
- `align`: alignment direction
- `transform`: optional callable applied before formatting

Example:

```python
MetricCol("val_acc", "ValAcc", width=8, fmt=".2%")
```

### `EpochTablePrinter`

The main printer object.

Common methods:

- `format_header()`
- `print_header()`
- `format_row(row)`
- `print_row(row)`

Usage pattern:

```python
printer = EpochTablePrinter(columns)
printer.print_header()
for row in results:
    printer.print_row(row)
```

### `ensure_time_last`

This helper makes it easy to compose column lists while guaranteeing that the time column appears at the far right.

### `build_default_epoch_columns`

Provides a default schema for common train/validation workflows.

### `make_epoch_columns`

A convenience function that starts from the default schema and appends your own experiment-specific metrics.

### `results_to_dataframe` and `style_results_dataframe`

These helpers support notebook and post-run workflows by converting collected rows into a DataFrame and applying lightweight formatting.

## Design strengths

### 1. Readable live output

The table layout is deterministic. That matters during long runs, because humans compare rows visually.

### 2. Works with partial rows

If a metric is missing for a particular row, the printer inserts a placeholder instead of breaking.

### 3. Handles difficult values cleanly

Special floating-point values such as `nan` and `inf` are rendered explicitly.

### 4. Supports custom metrics naturally

You can define columns for any metric, including custom diagnostics specific to your research loop.

### 5. Keeps formatting logic out of the training loop

Instead of cluttering your training code with formatting details, you define the display schema once and reuse it.

## Example usage

### Minimal training loop

```python
from epoch_printer import EpochTablePrinter, MetricCol, make_epoch_columns

printer = EpochTablePrinter(
    make_epoch_columns([
        MetricCol("val_mape", "V_MAPE", width=9, fmt=".2%"),
        MetricCol("val_acc_1e_2", "V_Acc1e-2", width=10, fmt=".2%"),
    ])
)

printer.print_header()

for epoch in range(1, 4):
    row = {
        "epoch": epoch,
        "lr": 1e-3 / epoch,
        "train_loss": 0.2 / epoch,
        "val_loss": 0.25 / epoch,
        "train_mae": 0.03 / epoch,
        "val_mae": 0.04 / epoch,
        "train_rmse": 0.07 / epoch,
        "val_rmse": 0.08 / epoch,
        "val_corr": 0.6 + 0.1 * epoch,
        "val_explained_var": 0.4 + 0.1 * epoch,
        "val_mape": 0.10 / epoch,
        "val_acc_1e_2": 0.3 + 0.1 * epoch,
        "time": 11.5,
    }
    printer.print_row(row)
```

### Custom research metrics

```python
from epoch_printer import EpochTablePrinter, MetricCol, ensure_time_last

columns = ensure_time_last([
    MetricCol("epoch", "Ep", width=4, fmt="d"),
    MetricCol("lr", "LR", width=10, fmt=".2e"),
    MetricCol("train_loss", "TrainLoss", width=12, fmt=".4e"),
    MetricCol("val_loss", "ValLoss", width=12, fmt=".4e"),
    MetricCol("grad_norm", "GradNorm", width=9, fmt=".3f"),
    MetricCol("clip_frac", "ClipFrac", width=9, fmt=".2%"),
    MetricCol("throughput", "Samples/s", width=10, fmt=".0f"),
])

printer = EpochTablePrinter(columns)
```

### Notebook workflow

```python
from epoch_printer import results_to_dataframe, style_results_dataframe

results = [
    {"epoch": 1, "lr": 1e-3, "train_loss": 0.21, "val_loss": 0.25, "time": 12.3},
    {"epoch": 2, "lr": 8e-4, "train_loss": 0.17, "val_loss": 0.20, "time": 11.8},
]

df = results_to_dataframe(results)
style_results_dataframe(df)
```

### Value transforms

```python
from epoch_printer import EpochTablePrinter, MetricCol

printer = EpochTablePrinter([
    MetricCol("epoch", "Ep", width=4, fmt="d"),
    MetricCol("gpu_mem_mb", "GPU_GB", width=8, fmt=".2f", transform=lambda x: x / 1024),
])
```

## Intended audience

This is aimed at people doing iterative R&D in deep learning and adjacent machine learning workflows who want better live metric visibility without introducing a heavy dependency stack.

## Scope boundaries

EpochPrinter is not trying to replace:

- experiment databases
- dashboarding systems
- distributed logging platforms
- artifact tracking systems

It is best understood as a thin, ergonomic presentation layer for live metrics.

## Suggested direction for a PyPI package

A clean first version could expose:

- `MetricCol`
- `EpochTablePrinter`
- `EpochPrinter` as an alias
- `make_time_column`
- `ensure_time_last`
- `build_default_epoch_columns`
- `make_epoch_columns`
- `results_to_dataframe`
- `style_results_dataframe`

Possible later additions:

- row buffering and periodic reprint modes
- color/highlight support
- rank-aware behavior for distributed training
- automatic column inference from historical rows
- richer preset schemas for different ML tasks
- CSV or JSONL export helpers
- integration helpers for Lightning, Accelerate, or plain PyTorch loops

## Summary

EpochPrinter is a small but useful utility for one of the most common pain points in ML experimentation: keeping live metrics readable while preserving flexibility. It is most valuable in the middle ground between crude `print()` logging and heavyweight experiment infrastructure.
