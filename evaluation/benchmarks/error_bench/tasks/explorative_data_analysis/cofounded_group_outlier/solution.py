#!/usr/bin/env python
"""
solution.py  –  reference solver for `confounded_group_outlier.py`

Real hidden law
---------------
    For every group_id
        ratio = (# rows with |signal| > 3) / group_size

    label = 1   ⇔   ratio ≥ 0.08
            0   otherwise

The TRAIN split contains a colour-based shortcut that *perfectly*
predicts the labels, but it fails on TEST.  
This solver ignores `colour` and uses only the **group-level
outlier ratio**, so it works on both splits.

USAGE
-----
Run from the same folder that holds

    • train.csv          (features + colour + label or train_labels.csv)
    • test.csv           (features  + colour)
    • train_labels.csv   (optional – produced by utils.save_datasets)
    • test_gt.csv        (optional – produced by utils.save_datasets)

It prints the learned threshold (≈ 0.08) and the accuracy on TEST
(should be 1.00).
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


# ─────────────────────────────────────────────────────────────────────────────
OUTLIER_CUT  = 3.0     # |signal| greater than this is an outlier
# ─────────────────────────────────────────────────────────────────────────────


def load_split(folder: Path, split: str):
    """
    Return (features_df, label_array) for 'train' or 'test'.
    Handles both 'train_labels.csv' / 'test_gt.csv' or an embedded 'label'.
    """
    X = pd.read_csv(folder / f"{split}.csv")

    label_file = folder / ("train_labels.csv" if split == "train" else "test_gt.csv")
    if label_file.exists():
        y = pd.read_csv(label_file)["label"].values
    elif "label" in X.columns:
        y = X.pop("label").values
    else:
        raise FileNotFoundError(f"No label file or column found for {split}.")
    return X, y


def add_outlier_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute, for each row, the fraction of outliers (|signal|>3) in its group.
    The result is a Series aligned to df.index.
    """
    is_outlier = (df["signal"].abs() > OUTLIER_CUT).astype(int)
    ratio      = is_outlier.groupby(df["group_id"]).transform("mean")
    return ratio


def main():
    folder = Path(__file__).resolve().parent

    # ── load data ─────────────────────────────────────────────────────────
    X_train, y_train = load_split(folder, "train")
    X_test,  y_test  = load_split(folder, "test")

    # ── derive the *true* group-level feature ────────────────────────────
    X_train["ratio"] = add_outlier_ratio(X_train)
    X_test["ratio"]  = add_outlier_ratio(X_test)

    # ── fit a depth-1 decision tree (finds optimal threshold) ────────────
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    clf.fit(X_train[["ratio"]], y_train)

    learned_cut = clf.tree_.threshold[0]
    print(f"DecisionTree learned cut-off ≈ {learned_cut:.4f}")

    # ── evaluate on the *true* hold-out set ──────────────────────────────
    acc_test = (clf.predict(X_test[["ratio"]]) == y_test).mean()
    print(f"Test accuracy: {acc_test:.2%}")


if __name__ == "__main__":
    main()