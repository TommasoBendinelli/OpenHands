import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path


def spike_ratio(series: pd.Series) -> float:
    """
    Feature  =  (max(x) − median(x)) / std(x)

    • Robustly removes the random DC offset by subtracting the median.
    • Keeps the sign (we only care about a *positive* spike).
    • Scales by the trace’s own standard deviation.

    For pure noise this ratio stays below ≈ 4.
    When a real spike is present it typically jumps above 8–12.
    """
    x = series.values.astype(float)

    baseline = np.median(x)                   # robust DC level
    sigma    = x.std(ddof=0) + 1e-8           # avoid divide-by-zero
    peak     = x.max()                        # signed peak

    return (peak - baseline) / sigma

def main() -> None:
    output_folder = Path(__file__).resolve().parent

    # ── load the splits written by utils.save_datasets ────────────────
    X_train = pd.read_csv(output_folder / "train.csv")
    y_train = pd.read_csv(output_folder / "train_labels.csv")["label"].values

    X_test  = pd.read_csv(output_folder / "test.csv")
    y_test  = pd.read_csv(output_folder / "test_gt.csv")["label"].values

    # ── feature extraction: a single scalar per signal ────────────────
    f_train = X_train.apply(spike_ratio, axis=1).values.reshape(-1, 1)
    f_test  = X_test.apply(spike_ratio, axis=1).values.reshape(-1, 1)

    # ── depth-1 Decision-Tree (stump) ─────────────────────────────────
    clf = DecisionTreeClassifier(max_depth=1, random_state=42)
    clf.fit(f_train, y_train)

    # ── evaluation ────────────────────────────────────────────────────
    acc = (clf.predict(f_test) == y_test).mean()
    print(f"Test accuracy: {acc:.2%}")


if __name__ == "__main__":
    main()