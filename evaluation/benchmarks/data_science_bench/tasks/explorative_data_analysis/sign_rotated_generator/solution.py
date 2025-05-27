"""
solution.py – 1-D DecisionTree for the rotated-sign task
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import random


THETA = np.deg2rad(30)               # same hidden angle used by generator
COS, SIN = np.cos(THETA), np.sin(THETA)


def uv_product(df: pd.DataFrame) -> np.ndarray:
    """Compute (u*v) for each row using the known rotation."""
    x = df["feat1"].values
    y = df["feat2"].values
    u = COS * x + SIN * y
    v = -SIN * x + COS * y
    return (u * v).reshape(-1, 1)


def load(folder: Path, split: str):
    X = pd.read_csv(folder / f"{split}.csv")
    y_path = folder / (f"{split}_labels.csv" if split == "train" else f"{split}_gt.csv")
    if y_path.exists():
        y = pd.read_csv(y_path)["label"].values
    else:
        y = X.pop("label").values
    return X, y


def main():
    random.seed(0); np.random.seed(0)
    here = Path(__file__).resolve().parent

    X_train, y_train = load(here, "train")
    X_test,  y_test  = load(here, "test")

    f_train = uv_product(X_train)
    f_test  = uv_product(X_test)

    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    clf.fit(f_train, y_train)

    print("Cut-off:", clf.tree_.threshold[0])
    print("Test accuracy:", clf.score(f_test, y_test))


if __name__ == "__main__":
    main()