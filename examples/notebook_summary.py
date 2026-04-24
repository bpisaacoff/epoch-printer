"""
Notebook summary example.

Prints a live epoch table during training, then converts the collected rows
to a styled DataFrame summary at the end of the notebook cell.

Run this script directly to see plain-text output.
In a Jupyter notebook, the final style_results_dataframe() call renders as
a styled HTML table.

Optional pandas dependency: pip install pandas
                        or: pip install "epoch-printer[pandas]"
"""
import math
import random

from epoch_printer import (
    EpochPrinter,
    MetricCol,
    make_epoch_columns,
    results_to_dataframe,
    style_results_dataframe,
)

NUM_EPOCHS = 10

printer = EpochPrinter(
    make_epoch_columns([
        MetricCol("val_acc", "V_Acc", width=8, fmt=".2%"),
    ])
)

results = []
printer.print_header()

for epoch in range(1, NUM_EPOCHS + 1):
    noise = random.gauss(0, 0.004)
    row = {
        "epoch":             epoch,
        "lr":                1e-3 * (0.9 ** epoch),
        "train_loss":        0.55 * math.exp(-0.18 * epoch) + 0.04 + noise,
        "val_loss":          0.62 * math.exp(-0.16 * epoch) + 0.06 + noise,
        "train_mae":         0.07 * math.exp(-0.14 * epoch) + 0.01,
        "val_mae":           0.08 * math.exp(-0.13 * epoch) + 0.012,
        "train_rmse":        0.12 * math.exp(-0.12 * epoch) + 0.02,
        "val_rmse":          0.14 * math.exp(-0.11 * epoch) + 0.025,
        "val_corr":          min(0.60 + 0.04 * epoch, 0.99),
        "val_explained_var": min(0.40 + 0.05 * epoch, 0.99),
        "val_acc":           min(0.50 + 0.05 * epoch + random.gauss(0, 0.01), 0.99),
        "time":              13.2 - 0.1 * epoch + random.gauss(0, 0.2),
    }
    printer.print_row(row)
    results.append(row)

print()

# Convert to DataFrame for post-run inspection.
# In a Jupyter notebook, replace the print block with: style_results_dataframe(df)
try:
    df = results_to_dataframe(results)
    print("DataFrame summary:")
    print()
    print(df.to_string(index=False))
    print()
    print("In a Jupyter notebook, replace the print above with:")
    print("  style_results_dataframe(df)")
except ImportError:
    print("Install pandas to see the DataFrame summary:")
    print("  pip install pandas  or  pip install 'epoch-printer[pandas]'")
