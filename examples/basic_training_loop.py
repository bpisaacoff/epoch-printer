"""
Basic training loop example.

Shows the most common usage pattern: build a column schema once, print
the header, then print one row per epoch.
"""

from epoch_printer import EpochPrinter, MetricCol, make_epoch_columns

printer = EpochPrinter(
    make_epoch_columns(
        [
            MetricCol("val_mape", "V_MAPE", width=9, fmt=".2%"),
            MetricCol("val_acc_1e_2", "V_Acc1e-2", width=10, fmt=".2%"),
        ]
    )
)

rows = [
    {
        "epoch": 1,
        "lr": 1.00e-3,
        "train_loss": 1.8421e-1,
        "val_loss": 2.1077e-1,
        "train_mae": 1.92e-2,
        "val_mae": 2.15e-2,
        "train_rmse": 4.51e-2,
        "val_rmse": 4.91e-2,
        "val_corr": 0.781,
        "val_explained_var": 0.612,
        "val_mape": 0.0842,
        "val_acc_1e_2": 0.431,
        "time": 12.47,
    },
    {
        "epoch": 2,
        "lr": 7.50e-4,
        "train_loss": 1.2376e-1,
        "val_loss": 1.5542e-1,
        "train_mae": 1.48e-2,
        "val_mae": 1.74e-2,
        "train_rmse": 3.73e-2,
        "val_rmse": 4.11e-2,
        "val_corr": 0.829,
        "val_explained_var": 0.691,
        "val_mape": 0.0698,
        "val_acc_1e_2": 0.507,
        "time": 11.93,
    },
    {
        "epoch": 3,
        "lr": 5.00e-4,
        "train_loss": 9.104e-2,
        "val_loss": 1.201e-1,
        "train_mae": 1.21e-2,
        "val_mae": 1.48e-2,
        "train_rmse": 3.14e-2,
        "val_rmse": 3.72e-2,
        "val_corr": 0.863,
        "val_explained_var": 0.744,
        "val_mape": 0.0571,
        "val_acc_1e_2": 0.562,
        "time": 11.64,
    },
]

printer.print_header()
for row in rows:
    printer.print_row(row)
