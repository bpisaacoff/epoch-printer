"""
Custom column schema example.

Shows how to build a fully custom schema, use transforms, handle missing
values gracefully, and work with special float values (nan, inf).
"""

from epoch_printer import EpochTablePrinter, MetricCol, ensure_time_last

# --- Custom research metrics schema ---

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

printer = EpochTablePrinter(columns)
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

print()

# --- Transform and special-float handling ---

print("Transform and missing-value handling:")

mem_printer = EpochTablePrinter(
    [
        MetricCol("epoch", "Ep", width=4, fmt="d"),
        MetricCol("phase", "Phase", width=10, fmt="s"),
        MetricCol("score", "Score", width=8, fmt=".3f"),
        MetricCol("error_rate", "Err", width=8, fmt=".2%"),
        # transform: raw value in MiB -> display in GiB
        MetricCol("gpu_mem_mb", "GPU_GB", width=8, fmt=".2f", transform=lambda x: x / 1024),
        MetricCol("time", "Time", width=8, fmt=".2f"),
    ]
)

mem_printer.print_header()
mem_printer.print_row(
    {"epoch": 1, "phase": "warmup", "score": float("nan"), "error_rate": 0.184, "gpu_mem_mb": 24576, "time": 4.2}
)
mem_printer.print_row(
    {"epoch": 2, "phase": "train", "score": float("inf"), "gpu_mem_mb": 28672, "time": 4.4}
)
mem_printer.print_row(
    {"epoch": 3, "phase": "eval", "score": 0.913, "error_rate": 0.071, "gpu_mem_mb": 20480, "time": 3.9}
)

print()

# --- Optional title and width defaults ---
# title defaults to key; width defaults to len(title).

print("Auto title and width (key used as header, sized to fit):")

auto_printer = EpochTablePrinter(
    [
        MetricCol("epoch", fmt="d"),          # title="epoch", width=5
        MetricCol("train_loss", fmt=".4e"),   # title="train_loss", width=10
        MetricCol("val_loss", "Val", fmt=".4e"),  # explicit title, width=3
        MetricCol("lr", "LR", width=10, fmt=".2e"),  # explicit title and width
    ]
)

auto_printer.print_header()
auto_printer.print_row({"epoch": 1, "train_loss": 0.1842, "val_loss": 0.2108, "lr": 1e-3})
auto_printer.print_row({"epoch": 2, "train_loss": 0.1391, "val_loss": 0.1794, "lr": 5e-4})

print()

# --- Line options: vertical column separators and horizontal row rules ---

print("Vertical lines between columns and horizontal rules between rows:")

line_cols = [
    MetricCol("epoch", "Ep", width=4, fmt="d"),
    MetricCol("train_loss", "TrainLoss", width=12, fmt=".4e"),
    MetricCol("val_loss", "ValLoss", width=12, fmt=".4e"),
    MetricCol("lr", "LR", width=10, fmt=".2e"),
]

line_printer = EpochTablePrinter(line_cols, vertical_lines=True, row_lines=True)
line_printer.print_header()
line_printer.print_row({"epoch": 1, "train_loss": 0.1842, "val_loss": 0.2108, "lr": 1e-3})
line_printer.print_row({"epoch": 2, "train_loss": 0.1238, "val_loss": 0.1554, "lr": 7.5e-4})
line_printer.print_row({"epoch": 3, "train_loss": 0.0981, "val_loss": 0.1203, "lr": 5e-4})
