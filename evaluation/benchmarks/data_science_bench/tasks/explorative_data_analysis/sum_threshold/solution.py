import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path


def sum_first_three(series: pd.Series) -> float:
    """
    Aggregate feature: sum of feat1-feat3.

    The dataset was constructed so that this sum alone
    perfectly separates the classes, letting a depth-1
    stump reach 100 % accuracy.
    """
    return series.iloc[:3].sum()


def main() -> None:
    output_folder = Path(__file__).resolve().parent

    # ── load splits created by utils.save_datasets ────────────────────────────
    X_train = pd.read_csv(output_folder / "train.csv")
    y_train = pd.read_csv(output_folder / "train_labels.csv")["label"].values

    X_test  = pd.read_csv(output_folder / "test.csv")
    y_test  = pd.read_csv(output_folder / "test_gt.csv")["label"].values

    # ── feature extraction: one scalar per row ───────────────────────────────
    f_train = X_train.apply(sum_first_three, axis=1).values.reshape(-1, 1)
    f_test  = X_test.apply(sum_first_three, axis=1).values.reshape(-1, 1)

    # ── depth-1 Decision-Tree (i.e. a decision stump) ────────────────────────
    clf = DecisionTreeClassifier(max_depth=1, random_state=42)
    clf.fit(f_train, y_train)

    # ── evaluation ───────────────────────────────────────────────────────────
    acc = (clf.predict(f_test) == y_test).mean()
    print(f"Test accuracy: {acc:.2%}")


if __name__ == "__main__":
    main()