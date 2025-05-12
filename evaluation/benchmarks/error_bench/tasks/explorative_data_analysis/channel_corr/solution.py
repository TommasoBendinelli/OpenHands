#!/usr/bin/env python
"""
Solution script for the correlation-vs-independence dataset.

• Loads the CSV files produced by `save_datasets`
• Extracts a single feature—the absolute Pearson correlation between the
  two channels in every sample
• Fits a depth-1 DecisionTreeClassifier (learns the best correlation
  threshold)
• Evaluates on the test set and prints the accuracy
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def abs_channel_correlation(df: pd.DataFrame) -> np.ndarray:
    """Compute |Pearson r| between the two half-channels of each row."""
    half = df.shape[1] // 2
    ch1 = df.iloc[:, :half].values
    ch2 = df.iloc[:, half:].values

    # Center each channel per row
    ch1_c = ch1 - ch1.mean(axis=1, keepdims=True)
    ch2_c = ch2 - ch2.mean(axis=1, keepdims=True)

    # Pearson numerator & denominator
    numer = (ch1_c * ch2_c).sum(axis=1)
    denom = np.sqrt((ch1_c ** 2).sum(axis=1) * (ch2_c ** 2).sum(axis=1)) + 1e-12
    return np.abs(numer / denom)


def main() -> None:
    root = Path(__file__).resolve().parent

    # Load data
    X_train = pd.read_csv(root / "train.csv")
    y_train = pd.read_csv(root / "train_labels.csv").label.values
    X_test = pd.read_csv(root / "test.csv")
    y_test = pd.read_csv(root / "test_gt.csv").label.values

    # Feature extraction
    f_train = abs_channel_correlation(X_train).reshape(-1, 1)
    f_test = abs_channel_correlation(X_test).reshape(-1, 1)

    # Depth-1 tree ≈ learned threshold on correlation
    clf = DecisionTreeClassifier(max_depth=1, random_state=42)
    clf.fit(f_train, y_train)

    y_pred = clf.predict(f_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.2%}")


if __name__ == "__main__":
    main()