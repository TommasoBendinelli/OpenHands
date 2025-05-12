#!/usr/bin/env python
"""
solution.py – reference solver for `group_mean_threshold.py`

Hidden law (built into the generator)
-------------------------------------
    label = 1   ⇔   group-mean('value')  >  3.0
            0   otherwise

The script:
1. loads the CSVs written by utils.save_datasets
2. derives one scalar feature per row = mean(value) of its group_id
3. fits a depth-1 DecisionTree (just to confirm the cutoff)
4. reports accuracy on the test split
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path


# ──────────────────────────────────────────────────────────────────
THRESH_GUESS = 3.0            # generator’s exact boundary
# ──────────────────────────────────────────────────────────────────


def load_split(folder: Path, split: str):
    """
    Return (X, y) where X is the feature DataFrame without the label column.
    Handles both `train_labels.csv` / `test_gt.csv` as well as an embedded
    'label' column.
    """
    X = pd.read_csv(folder / f"{split}.csv")

    label_file = (
        folder / "train_labels.csv" if split == "train" else folder / "test_gt.csv"
    )
    if label_file.exists():
        y = pd.read_csv(label_file)["label"].values
    elif "label" in X.columns:
        y = X.pop("label").values
    else:
        raise FileNotFoundError(f"No labels found for {split} split.")
    return X, y


def add_group_mean_feature(df: pd.DataFrame) -> pd.Series:
    """
    For each row, compute mean(value) of its group_id and return that Series.
    """
    return df.groupby("group_id")["value_0"].transform("mean")


def main():
    folder = Path(__file__).resolve().parent

    X_train, y_train = load_split(folder, "train")
    X_test, y_test = load_split(folder, "test")

    # 1️⃣  Derive the group-level statistic as a column
    X_train["grp_mean"] = add_group_mean_feature(X_train)
    X_test["grp_mean"] = add_group_mean_feature(X_test)

    # 2️⃣  Fit a depth-1 decision tree (essentially finds the optimal threshold)
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    clf.fit(X_train[["grp_mean"]], y_train)

    learned_cut = clf.tree_.threshold[0]
    print(f"DecisionTree learned cut-off ≈ {learned_cut:.3f} "
          f"(generator used {THRESH_GUESS})")

    # 3️⃣  Evaluate
    acc = (clf.predict(X_test[["grp_mean"]]) == y_test).mean()
    print(f"Test accuracy: {acc:.2%}")


if __name__ == "__main__":
    main()