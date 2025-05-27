#!/usr/bin/env python
"""
solution.py  –  reference solver for the `row_max_abs.py` dataset

Decision rule
-------------
    label = 1   ⇔   max(|feat1…feat12|) > 4.0
            0   otherwise
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# ──────────────────────────────────────────────────────────────────────────────
N_FEATURES   = 12                              # feat1 … feat12
THRESH_GUESS = 4.0                             # expected optimal split
FEATURE_COLS = [f"feat{i+1}" for i in range(N_FEATURES)]
# ──────────────────────────────────────────────────────────────────────────────


def row_max_abs(df: pd.DataFrame) -> np.ndarray:
    """Row-wise maximum absolute value across the first 12 feature columns."""
    return df[FEATURE_COLS].abs().max(axis=1).values


def load_split(folder: Path, split: str):
    """
    Load X and y for 'train' or 'test', coping with both label-file conventions:
        • train_labels.csv   for training
        • test_gt.csv        for test
    or a 'label' column embedded in the feature CSV.
    """
    X = pd.read_csv(folder / f"{split}.csv")

    if split == "train":
        label_path = folder / "train_labels.csv"
    else:                               # "test"
        label_path = folder / "test_gt.csv"

    if label_path.exists():
        y = pd.read_csv(label_path)["label"].values
    elif "label" in X.columns:          # fallback: embedded column
        y = X.pop("label").values
    else:
        raise FileNotFoundError(f"No label information found for {split} split.")

    return X, y


def main():
    folder = Path(__file__).resolve().parent

    X_train, y_train = load_split(folder, "train")
    X_test,  y_test  = load_split(folder, "test")

    # ── feature engineering ───────────────────────────────────────────────
    f_train = row_max_abs(X_train).reshape(-1, 1)
    f_test  = row_max_abs(X_test).reshape(-1, 1)

    # ── depth-1 decision tree learns (and shows) the threshold ────────────
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    clf.fit(f_train, y_train)

    learned_cut = clf.tree_.threshold[0]
    print(f"DecisionTree learned cut-off ≈ {learned_cut:.3f}  "
          f"(generator used {THRESH_GUESS})")

    acc = (clf.predict(f_test) == y_test).mean()
    print(f"Test accuracy: {acc:.2%}")


if __name__ == "__main__":
    main()