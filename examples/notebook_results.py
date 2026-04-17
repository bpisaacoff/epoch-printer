"""
Notebook / DataFrame results example.

In a notebook, collect rows during training and convert them to a styled
DataFrame at the end for a clean summary view.

Run this script directly to see the plain DataFrame output.
In a Jupyter notebook, the last line renders as a styled HTML table.
"""

from epoch_printer import (
    EpochPrinter,
    make_epoch_columns,
    results_to_dataframe,
    style_results_dataframe,
)

# Simulate collecting rows during a training run
results = []

printer = EpochPrinter(make_epoch_columns())
printer.print_header()

for epoch in range(1, 6):
    row = {
        "epoch": epoch,
        "lr": 1e-3 * (0.8 ** epoch),
        "train_loss": 0.30 / epoch,
        "val_loss": 0.35 / epoch,
        "train_mae": 0.04 / epoch,
        "val_mae": 0.05 / epoch,
        "train_rmse": 0.09 / epoch,
        "val_rmse": 0.10 / epoch,
        "val_corr": min(0.5 + 0.1 * epoch, 0.99),
        "val_explained_var": min(0.3 + 0.1 * epoch, 0.99),
        "time": 11.5 - 0.2 * epoch,
    }
    printer.print_row(row)
    results.append(row)

print()

# Convert to DataFrame for post-run inspection
df = results_to_dataframe(results)
print(df.to_string(index=False))

# In a Jupyter notebook, replace the above print with:
#   style_results_dataframe(df)
