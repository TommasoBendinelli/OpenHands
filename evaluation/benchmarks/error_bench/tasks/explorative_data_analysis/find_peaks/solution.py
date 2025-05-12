#!/usr/bin/env python
"""
solution.py  –  reference solver for `generate_peak_dataset.py`

Real hidden law
---------------
    • **Class 0** signals contain **3 – 4 non-overlapping Gaussian peaks**.
    • **Class 1** signals contain **5 – 6 non-overlapping Gaussian peaks**.

A *depth-1* decision tree that uses only the **peak count** therefore
separates the two classes with zero error:

        label = 1   ⇔   peak_count ≥ 4.5
                0   otherwise

Feature engineering
-------------------
For every 1-D signal we

1.  Estimate the local baseline with the median value.
2.  Raise a *horizontal* threshold line  
        `threshold = median + Δ`,  with Δ = 1.8.
    (Noise is ≈ 0.2 σ while true peaks are ≥ 2 units above the baseline,
    so this line sits comfortably *between* noise and peaks.)
3.  Find all **local maxima** that stand above this line.
4.  From those maxima, keep only the highest one in every
    `min_separation = 20` samples (≈ the minimal 6 · σ support of a peak).
    The survivors are the peaks, whose count is the single feature.

USAGE
-----
Run from the folder that contains

    • train.csv          (features + optional label column)
    • test.csv           (features + optional label column)
    • train_labels.csv   (optional – produced by utils.save_datasets)
    • test_gt.csv        (optional – produced by utils.save_datasets)

It prints the learned cut-off (≈ 4.5) and the accuracy on TEST
(should be 1.00 if the data obeys the generator).
"""

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# ─────────────────────────────────────────────────────────────────────────────
DELTA              = 1.8   # vertical offset of the horizontal line
MIN_SEPARATION     = 22    # minimal distance (samples) between two peaks
# ─────────────────────────────────────────────────────────────────────────────


def load_split(folder: Path, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Return (features_df, label_array) for 'train' or 'test'.
    Works whether the label is embedded or saved separately.
    """
    X = pd.read_csv(folder / f"{split}.csv")

    y_file = folder / ("train_labels.csv" if split == "train" else "test_gt.csv")
    if y_file.exists():
        y = pd.read_csv(y_file)["label"].values
    elif "label" in X.columns:
        y = X.pop("label").values
    else:
        raise FileNotFoundError(f"No label file or 'label' column found for {split}.")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
#  Peak-count feature
# ─────────────────────────────────────────────────────────────────────────────
def _peak_count(signal: Sequence[float],
                delta: float = DELTA,
                min_sep: int = MIN_SEPARATION) -> int:
    """
    Return the number of non-overlapping peaks in *signal*.

    A peak is the highest local maximum within a *min_sep*-wide window,
    provided that it rises above  (median(signal) + delta).
    """
    sig = np.asarray(signal, dtype=float)
    baseline = np.median(sig)
    thresh   = baseline + delta

    # indices of local maxima that stand above the threshold
    maxima = np.where(
        (sig[1:-1] > sig[:-2]) &
        (sig[1:-1] > sig[2: ]) &
        (sig[1:-1] > thresh)
    )[0] + 1

    # keep only the tallest maximum in every ±min_sep window
    maxima = sorted(maxima, key=sig.__getitem__, reverse=True)
    selected: list[int] = []
    for idx in maxima:
        if all(abs(idx - j) > min_sep for j in selected):
            selected.append(idx)

    return len(selected)


def add_peak_count(df: pd.DataFrame) -> pd.Series:
    """Compute the peak count for every row in *df* (all columns are signal)."""
    # df.values is a 2-D ndarray – fast row-wise iteration
    counts = np.fromiter((_peak_count(row) for row in df.values),
                         dtype=int, count=len(df))
    return pd.Series(counts, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    folder = Path(__file__).resolve().parent

    # ── load data ─────────────────────────────────────────────────────────
    X_train, y_train = load_split(folder, "train")
    X_test,  y_test  = load_split(folder, "test")

    # ── derive the single discriminative feature ─────────────────────────
    X_train["peak_count"] = add_peak_count(X_train)
    X_test["peak_count"]  = add_peak_count(X_test)

    # ── depth-1 decision stump finds the optimal cut-off ─────────────────
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    clf.fit(X_train[["peak_count"]], y_train)
    # Accuracy on the TRAIN set is 100% 
    acc_train = clf.score(X_train[["peak_count"]], y_train)
    print(f"Train accuracy: {acc_train:.2%}")

    learned_cut = clf.tree_.threshold[0]
    print(f"DecisionTree learned cut-off ≈ {learned_cut:.1f}")

    # ── evaluate on the hold-out TEST split ──────────────────────────────
    acc_test = (clf.predict(X_test[["peak_count"]]) == y_test).mean()
    print(f"Test accuracy: {acc_test:.2%}")


if __name__ == "__main__":
    main()