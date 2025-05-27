import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path


def burst_ratio(series: pd.Series, window: int = 30) -> float:
    """Compute the ratio between the maximum and mean rolling
    standard deviation of a time‑series.  A variance burst yields a
    high max/mean ratio.

    Parameters
    ----------
    series : pd.Series
        One time‑series (row) from the dataset.
    window : int, default 30
        Window size (in samples) for the rolling standard deviation.

    Returns
    -------
    float
        max(std_roll) / mean(std_roll) for the given series.
    """
    s = pd.Series(series.values, copy=False)
    roll_std = s.rolling(window).std().fillna(0)
    mean_std = roll_std.mean() + 1e-8  # avoid division‑by‑zero
    return roll_std.max() / mean_std


def main() -> None:
    output_folder = Path(__file__).resolve().parent

    # Load the training / test splits produced by utils.save_datasets
    X_train = pd.read_csv(output_folder / "train.csv")
    y_train = pd.read_csv(output_folder / "train_labels.csv")["label"].values

    X_test = pd.read_csv(output_folder / "test.csv")
    y_test = pd.read_csv(output_folder / "test_gt.csv")["label"].values

    # Feature extraction: one scalar per series
    f_train = X_train.apply(burst_ratio, axis=1).values.reshape(-1, 1)
    f_test = X_test.apply(burst_ratio, axis=1).values.reshape(-1, 1)

    # A tiny Decision‑Tree works well with this single, highly
    # informative feature.
    clf = DecisionTreeClassifier(max_depth=1, random_state=42)
    clf.fit(f_train, y_train)

    # Evaluate on the hold‑out set
    acc = (clf.predict(f_test) == y_test).mean()
    print(f"Test accuracy: {acc:.2%}")


if __name__ == "__main__":
    main()
