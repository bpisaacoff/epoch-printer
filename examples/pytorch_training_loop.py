"""
PyTorch-style training loop example.

Demonstrates epoch-printer in a loop shaped like a real PyTorch training run.
No PyTorch required — model, loader, optimizer, and scheduler are all mocked.
"""
import math
import random
import time

from epoch_printer import EpochTablePrinter, MetricCol, ensure_time_last


# ---------------------------------------------------------------------------
# Mock training infrastructure
# ---------------------------------------------------------------------------

def train_one_epoch(epoch, lr):
    """Simulates one training epoch. Returns a metrics dict."""
    time.sleep(0.01)
    noise = random.gauss(0, 0.005)
    loss = 0.8 * math.exp(-0.25 * epoch) + 0.05 + noise
    return {
        "loss":       max(loss, 0.01),
        "grad_norm":  abs(random.gauss(1.5, 0.3)),
        "throughput": random.gauss(8400, 200),
    }


def evaluate(epoch):
    """Simulates one validation pass. Returns a metrics dict."""
    noise = random.gauss(0, 0.007)
    val_loss = 0.9 * math.exp(-0.22 * epoch) + 0.07 + noise
    val_acc  = min(0.45 + 0.06 * epoch + random.gauss(0, 0.01), 0.99)
    return {
        "loss":     max(val_loss, 0.01),
        "accuracy": val_acc,
    }


class CosineScheduler:
    """Minimal cosine-decay LR scheduler stand-in for torch.optim.lr_scheduler."""
    def __init__(self, lr_init, num_epochs):
        self.lr_init    = lr_init
        self.num_epochs = num_epochs
        self._epoch     = 0

    def step(self):
        self._epoch += 1

    def get_last_lr(self):
        t = self._epoch / self.num_epochs
        return [self.lr_init * 0.5 * (1 + math.cos(math.pi * t))]


# ---------------------------------------------------------------------------
# Training setup
# ---------------------------------------------------------------------------

NUM_EPOCHS = 8
LR_INIT    = 1e-3

scheduler = CosineScheduler(LR_INIT, NUM_EPOCHS)

printer = EpochTablePrinter(
    ensure_time_last([
        MetricCol("epoch",      "Ep",        width=4,  fmt="d"),
        MetricCol("lr",         "LR",        width=10, fmt=".2e"),
        MetricCol("train_loss", "TrainLoss", width=12, fmt=".4e"),
        MetricCol("val_loss",   "ValLoss",   width=12, fmt=".4e"),
        MetricCol("val_acc",    "V_Acc",     width=8,  fmt=".2%"),
        MetricCol("grad_norm",  "GradNorm",  width=9,  fmt=".3f"),
        MetricCol("samp_s",     "Samp/s",    width=8,  fmt=".0f"),
    ])
)

printer.print_header()

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()

    train = train_one_epoch(epoch, scheduler.get_last_lr()[0])
    val   = evaluate(epoch)
    scheduler.step()

    printer.print_row({
        "epoch":      epoch,
        "lr":         scheduler.get_last_lr()[0],
        "train_loss": train["loss"],
        "val_loss":   val["loss"],
        "val_acc":    val["accuracy"],
        "grad_norm":  train["grad_norm"],
        "samp_s":     train["throughput"],
        "time":       time.time() - t0,
    })
