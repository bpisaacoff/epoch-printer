"""
Experiment tracker integration example.

Shows how to use epoch-printer alongside W&B, TensorBoard, and MLflow.
epoch-printer only writes to stdout; your tracker handles persistence and
visualization. Use the same underlying training metrics for both — tracker
calls may need light filtering (e.g. excluding "epoch" from MLflow log_metrics,
which expects only numeric scalars and uses step as the epoch counter).

All tracker calls are mocked — no external packages required to run this file.
Replace the mock classes with real tracker objects to use in your project.
"""
import math
import random
import time

from epoch_printer import EpochTablePrinter, MetricCol, ensure_time_last


# ---------------------------------------------------------------------------
# Mock experiment trackers
# Replace these with your real tracker objects.
# ---------------------------------------------------------------------------

class MockWandb:
    """Stand-in for wandb. Real usage: import wandb; wandb.init(...)"""
    def log(self, metrics: dict, step: int = None):
        pass  # real: wandb.log(metrics, step=step)


class MockSummaryWriter:
    """Stand-in for torch.utils.tensorboard.SummaryWriter."""
    def add_scalars(self, tag: str, tag_scalar_dict: dict, global_step: int):
        pass  # real: writer.add_scalars(tag, tag_scalar_dict, global_step)

    def close(self):
        pass  # real: writer.close()


class MockMlflow:
    """Stand-in for mlflow. Real usage: import mlflow; mlflow.start_run()"""
    def log_metrics(self, metrics: dict, step: int = None):
        pass  # real: mlflow.log_metrics(metrics, step=step)


wandb   = MockWandb()
writer  = MockSummaryWriter()
mlflow  = MockMlflow()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

NUM_EPOCHS = 6

printer = EpochTablePrinter(
    ensure_time_last([
        MetricCol("epoch",      "Ep",        width=4,  fmt="d"),
        MetricCol("lr",         "LR",        width=10, fmt=".2e"),
        MetricCol("train_loss", "TrainLoss", width=12, fmt=".4e"),
        MetricCol("val_loss",   "ValLoss",   width=12, fmt=".4e"),
        MetricCol("val_mae",    "V_MAE",     width=10, fmt=".3e"),
        MetricCol("val_acc",    "V_Acc",     width=8,  fmt=".2%"),
    ])
)
printer.print_header()

for epoch in range(1, NUM_EPOCHS + 1):
    t0    = time.time()
    noise = random.gauss(0, 0.005)
    row   = {
        "epoch":      epoch,
        "lr":         1e-3 * math.exp(-0.15 * epoch),
        "train_loss": 0.6 * math.exp(-0.20 * epoch) + 0.05 + noise,
        "val_loss":   0.7 * math.exp(-0.18 * epoch) + 0.07 + noise,
        "val_mae":    0.08 * math.exp(-0.15 * epoch) + 0.01,
        "val_acc":    min(0.52 + 0.07 * epoch + random.gauss(0, 0.01), 0.99),
        "time":       time.time() - t0 + random.gauss(10, 0.5),
    }

    # epoch-printer: readable stdout ----------------------------------------
    printer.print_row(row)

    # W&B: log the same metrics dict; "epoch" is redundant with step but harmless
    wandb.log({k: v for k, v in row.items() if k != "epoch"}, step=epoch)

    # TensorBoard: group scalars by category --------------------------------
    writer.add_scalars("loss",     {"train": row["train_loss"], "val": row["val_loss"]}, epoch)
    writer.add_scalars("accuracy", {"val": row["val_acc"]}, epoch)

    # MLflow: log_metrics expects numeric scalars; exclude "epoch" since step covers it
    mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)

writer.close()

print()
print("Replace the mock classes above with your real tracker objects.")
print("epoch-printer output appears in stdout; the tracker handles persistence.")
