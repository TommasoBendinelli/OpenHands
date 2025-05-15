"""
solution.py – 100 % accurate solver for the *dominant_feature* dataset
(no SciPy dependencies, just scikit‑learn).

Idea
-----
For every row we engineer a single feature

    diff = feat3 − max(feat1, feat2)

The dataset’s rule is precisely *label 1 ⇔ diff > 0*. Therefore a
DecisionTreeClassifier with **one** decision node (``max_depth=1``)
separates the classes perfectly.
"""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def diff_feature(df: pd.DataFrame) -> np.ndarray:
    """Return (n_samples, 1) array with *feat3 − max(feat1, feat2)*."""
    # Cast to NumPy for speed and to avoid column‑name issues.
    arr = df[["1", "2", "3"]].to_numpy()
    diff = arr[:, 2] - np.maximum(arr[:, 0], arr[:, 1])
    return diff.reshape(-1, 1)


def load_split(folder: Path, split: str):
    """Load feature matrix *X* and label vector *y* for a data *split*."""
    X = pd.read_csv(folder / f"{split}.csv")

    # Labels may be delivered in a companion file or appended to *X*.
    y_file = folder / (
        f"{split}_labels.csv" if split == "train" else f"{split}_gt.csv"
    )
    if y_file.exists():
        y = pd.read_csv(y_file)["label"].to_numpy()
    else:
        y = X.pop("label").to_numpy()

    return X, y


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    random.seed(0)
    np.random.seed(0)

    here = Path(__file__).resolve().parent

    # ----- data ------------------------------------------------------------- #
    X_train, y_train = load_split(here, "train")
    X_test, y_test = load_split(here, "test")

    f_train = diff_feature(X_train)
    f_test = diff_feature(X_test)

    # ----- model ------------------------------------------------------------ #
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    clf.fit(f_train, y_train)

    # ----- results ---------------------------------------------------------- #
    print(f"Learned threshold on 'diff' feature: {clf.tree_.threshold[0]:.3f}")
    print(f"Train accuracy : {clf.score(f_train, y_train):.2%}")
    print(f"Test  accuracy : {clf.score(f_test, y_test):.2%}")


if __name__ == "__main__":
    main()
