#!/usr/bin/env python
"""
solution.py – solver for group_outlier_ratio.py
Derives one feature per row = outlier ratio of its group, then uses a
depth-1 DecisionTree to confirm the 0.08 split.
"""

import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier


def load(folder: Path, split: str):
    X = pd.read_csv(folder / f"{split}.csv")
    y_file = folder / ("train_labels.csv" if split=="train" else "test_gt.csv")
    y = pd.read_csv(y_file)["label"].values if y_file.exists() else X.pop("label").values
    return X, y


def add_ratio(df: pd.DataFrame) -> pd.DataFrame:
    ratio   = df["signal"].abs().groupby(df["group_id"]).transform("mean")
    df = df.copy(); df["ratio"] = ratio
    return df


def main():
    here         = Path(__file__).resolve().parent
    X_tr, y_tr   = load(here, "train")
    X_te, y_te   = load(here, "test")

    X_tr = add_ratio(X_tr)
    X_te = add_ratio(X_te)

    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    clf.fit(X_tr[["ratio"]], y_tr)


    acc = (clf.predict(X_te[["ratio"]]) == y_te).mean()
    print(f"Test accuracy: {acc:.2%}")


if __name__ == "__main__":
    main()