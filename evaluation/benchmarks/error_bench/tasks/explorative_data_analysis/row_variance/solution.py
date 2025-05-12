#!/usr/bin/env python
"""
solve_row_variance.py
Train a 1-D Decision Tree on row variance → 100 % accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path


def row_variance(df: pd.DataFrame, n_feats: int = 8) -> np.ndarray:
    """
    Return the variance of the first `n_feats` columns for every row.
    (ddof=0 gives population variance, matching the generator script.)
    """
    return df.iloc[:, :n_feats].var(axis=1, ddof=0).values


def main():
    here = Path(__file__).resolve().parent

    X_train = pd.read_csv(here / "train.csv")
    y_train = pd.read_csv(here / "train_labels.csv")["label"].values

    X_test  = pd.read_csv(here / "test.csv")
    y_test  = pd.read_csv(here / "test_gt.csv")["label"].values

    # -------------------------------------------------------
    # Feature engineering: one column = per-row variance
    # -------------------------------------------------------
    var_train = row_variance(X_train).reshape(-1, 1)
    var_test  = row_variance(X_test).reshape(-1, 1)

    # -------------------------------------------------------
    # A single-split tree is enough (and gives an interpretable threshold)
    # -------------------------------------------------------
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    clf.fit(var_train, y_train)

    y_pred = clf.predict(var_test)
    acc = (y_pred == y_test).mean()

    print(f"Test accuracy: {acc:.2f}")          # should print 1.00
    # Optional: inspect the learned threshold
    thresh = clf.tree_.threshold[0]
    print(f"Learned variance threshold: {thresh:.4f}")


if __name__ == "__main__":
    main()