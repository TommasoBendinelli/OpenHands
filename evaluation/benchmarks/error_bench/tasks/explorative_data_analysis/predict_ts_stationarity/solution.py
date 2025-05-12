#!/usr/bin/env python
"""
solution.py  –  reference solver for the `stationarity_ts` dataset

Decision rule
-------------
    label = 1   (stationary)         ⇔   var(Δx) / var(x)  ≥  0.9
            0   (non-stationary)     otherwise

The script:
1.  loads the CSV files written by utils.save_datasets
2.  derives one scalar feature per time-series row  →  variance-ratio
       ratio =  var(first-difference) / var(raw series)
3.  fits a depth-1 DecisionTree (just to learn / confirm the cut-off)
4.  evaluates accuracy on the held-out test split
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import random

# ──────────────────────────────────────────────────────────────────────────────
EPS = 1e-8                    # protect against division-by-zero
# ──────────────────────────────────────────────────────────────────────────────


def variance_ratio(df: pd.DataFrame) -> np.ndarray:
    """
    Return the row-wise ratio  var(Δx) / var(x).

    With a random-walk the raw-series variance explodes (∝ n),
    whereas the first-difference variance stays bounded (≈ σ²).
    For white-noise (stationary) both variances are of the same order,
    so the ratio is ≳ 1.  This single scalar separates the classes cleanly.
    """
    ts = df.values                                        # shape (n_samples, n_steps)
    var_series = np.var(ts, axis=1, ddof=1)
    var_diff   = np.var(np.diff(ts, axis=1), axis=1, ddof=1)
    return var_diff / (var_series + EPS)                  # (n_samples,)


def load_split(folder: Path, split: str):
    """
    Load X and y for *train* or *test*.
    utils.save_datasets either keeps the labels in the same CSV
    or writes them to  <split>_labels.csv  (for the test set).
    """
    X = pd.read_csv(folder / f"{split}.csv")
    if split == "train":            # labels in the same file
        y_path = folder / f"{split}_labels.csv"
    else:                           # labels in a separate file
        y_path = folder / f"{split}_gt.csv"
    if y_path.exists():               # labels in a separate file (e.g. test set)
        y = pd.read_csv(y_path)["label"].values

    else:                             # labels embedded in X
        y = X.pop("label").values
    return X, y


def main():
    random.seed(0)  # for reproducibility
    np.random.seed(0)
    
    folder = Path(__file__).resolve().parent      # dataset lives next to this script

    X_train, y_train = load_split(folder, "train")
    X_test,  y_test  = load_split(folder, "test")

    # ── Feature engineering ────────────────────────────────────────────────
    f_train = variance_ratio(X_train).reshape(-1, 1)
    f_test  = variance_ratio(X_test).reshape(-1, 1)

    # ── Train a single-split tree (depth = 1) ──────────────────────────────
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    clf.fit(f_train, y_train)

    # learned threshold (for curiosity)
    learned_cut = clf.tree_.threshold[0]
    print(f"DecisionTree learned cut-off ≈ {learned_cut:.3f}")

    # ── Evaluate on test set ───────────────────────────────────────────────
    y_pred = clf.predict(f_test)
    acc = (y_pred == y_test).mean()
    print(f"Test accuracy: {acc:.2f}")


if __name__ == "__main__":
    main()