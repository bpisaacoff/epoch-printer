"""
Custom research metrics example.

Shows how to define your own column schema for experiments that don't fit
the default supervised-learning template: VAE training, custom loss terms,
gradient stats, hardware metrics, and table formatting options.
"""
import random

from epoch_printer import EpochTablePrinter, MetricCol, ensure_time_last


# ---------------------------------------------------------------------------
# Example 1: VAE training with reconstruction + KL loss
# ---------------------------------------------------------------------------

print("VAE training (reconstruction + KL loss):")

vae_columns = ensure_time_last([
    MetricCol("epoch",      "Ep",       width=4,  fmt="d"),
    MetricCol("lr",         "LR",       width=10, fmt=".2e"),
    MetricCol("recon_loss", "Recon",    width=12, fmt=".4e"),
    MetricCol("kl_div",     "KL",       width=10, fmt=".4e"),
    MetricCol("total_loss", "Total",    width=12, fmt=".4e"),
    MetricCol("grad_norm",  "GradNorm", width=9,  fmt=".3f"),
    MetricCol("throughput", "Samp/s",   width=8,  fmt=".0f"),
])

printer = EpochTablePrinter(vae_columns)
printer.print_header()

for epoch in range(1, 6):
    recon = 0.6 / epoch + random.gauss(0, 0.005)
    kl    = 0.05 * epoch + random.gauss(0, 0.002)
    printer.print_row({
        "epoch":      epoch,
        "lr":         1e-3 * (0.85 ** epoch),
        "recon_loss": recon,
        "kl_div":     kl,
        "total_loss": recon + kl,
        "grad_norm":  abs(random.gauss(1.2, 0.2)),
        "throughput": random.gauss(6200, 150),
        "time":       random.gauss(14.0, 0.3),
    })

print()


# ---------------------------------------------------------------------------
# Example 2: Value transforms and missing-value handling
# Raw GPU memory is in MiB; display in GiB via a transform.
# val_acc is absent in epoch 1 (warmup) and renders as "--".
# ---------------------------------------------------------------------------

print("GPU memory transform (MiB -> GiB) and missing-value handling:")

mem_columns = [
    MetricCol("epoch",      "Ep",      width=4,  fmt="d"),
    MetricCol("phase",      "Phase",   width=8,  fmt="s"),
    MetricCol("val_loss",   "ValLoss", width=10, fmt=".4f"),
    MetricCol("val_acc",    "V_Acc",   width=8,  fmt=".2%"),
    MetricCol("gpu_mem_mb", "GPU_GiB", width=8,  fmt=".2f",
              transform=lambda x: x / 1024),
    MetricCol("time",       "Time",    width=8,  fmt=".2f"),
]

mem_printer = EpochTablePrinter(mem_columns)
mem_printer.print_header()

mem_printer.print_row({
    "epoch": 1, "phase": "warmup",
    "val_loss": 0.743, "gpu_mem_mb": 22528, "time": 11.2,
    # val_acc absent -> shows "--"
})
mem_printer.print_row({
    "epoch": 2, "phase": "train",
    "val_loss": 0.581, "val_acc": 0.672, "gpu_mem_mb": 26624, "time": 13.4,
})
mem_printer.print_row({
    "epoch": 3, "phase": "train",
    "val_loss": 0.447, "val_acc": 0.741, "gpu_mem_mb": 26624, "time": 13.1,
})

print()


# ---------------------------------------------------------------------------
# Example 3: Vertical column separators and row rules
# ---------------------------------------------------------------------------

print("Vertical column separators and row rules:")

line_cols = [
    MetricCol("epoch",      "Ep",        width=4,  fmt="d"),
    MetricCol("train_loss", "TrainLoss", width=12, fmt=".4e"),
    MetricCol("val_loss",   "ValLoss",   width=12, fmt=".4e"),
    MetricCol("lr",         "LR",        width=10, fmt=".2e"),
]

line_printer = EpochTablePrinter(line_cols, vertical_lines=True, row_lines=True)
line_printer.print_header()
line_printer.print_row({"epoch": 1, "train_loss": 0.1842, "val_loss": 0.2108, "lr": 1e-3})
line_printer.print_row({"epoch": 2, "train_loss": 0.1238, "val_loss": 0.1554, "lr": 7.5e-4})
line_printer.print_row({"epoch": 3, "train_loss": 0.0981, "val_loss": 0.1203, "lr": 5e-4})
