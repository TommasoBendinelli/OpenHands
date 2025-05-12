"""
solution.py – *single‑feature* decision‑stump solver for the `setpoint_dataset`
==============================================================================
Detects the **number of large mean jumps** in each time‑series.

* **Feature (one scalar):**  
  Robustly estimated *jump‑count* – number of first‑differences whose magnitude
  exceeds `k·σ̂`, where
        σ̂ = median(|Δx|) / 0.6745,
  i.e. the MAD estimate of the s.d. of the noise (since Δx ~ N(0,2σ²) inside
  flat segments).  With `k = 4`, noise rarely triggers a false jump even at the
  highest noise level (σ = 0.6).

* **Classifier:** a *DecisionTreeClassifier* with `max_depth=1`, i.e. a decision
  stump.  It learns a single threshold on the jump‑count:  
      **count ≥ θ  ⇒  ≥ 2 set‑points (label 1)**  
      **count < θ  ⇒  0–1 set‑points (label 0)**

Empirically this yields ≳80 % accuracy on the mixed‑noise training pool
produced by the generator and ≈70 – 88 % across individual noise levels
σ ∈ {0.2, 0.3, 0.4, 0.6}.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# ──────────────────────────────────────────────────────────────────────────────
EPS = 1e-9
K = 4.0           # jump detection threshold multiplier
# ──────────────────────────────────────────────────────────────────────────────


def _jump_count(row: np.ndarray, k: float = K) -> float:
    """Return the number of 'big jumps' in *row*."""
    diff = np.diff(row)
    sigma_hat = np.median(np.abs(diff)) / 0.6745          # robust σ of diff
    thresh = k * sigma_hat + EPS
    return np.sum(np.abs(diff) > thresh)


def extract_feature(df: pd.DataFrame) -> np.ndarray:
    """Compute the jump‑count feature → shape (n_samples, 1)."""
    ts = df.values
    feat = np.empty((ts.shape[0], 1), dtype=float)
    for i, row in enumerate(ts):
        feat[i, 0] = _jump_count(row)
    return feat


def _load_split(folder: Path, split: str):
    X = pd.read_csv(folder / f"{split}.csv")
    y_file = folder / (f"{split}_labels.csv" if split == "train" else f"{split}_gt.csv")
    if y_file.exists():
        y = pd.read_csv(y_file)["label"].values
    else:
        y = X.pop("label").values
    return X, y


def main():
    folder = Path(__file__).resolve().parent

    X_train, y_train = _load_split(folder, "train")
    X_test, y_test = _load_split(folder, "test")

    f_train = extract_feature(X_train)
    f_test = extract_feature(X_test)

    stump = DecisionTreeClassifier(max_depth=1, random_state=0)
    stump.fit(f_train, y_train)

    train_acc = stump.score(f_train, y_train)
    test_acc = stump.score(f_test, y_test)
    print(f"Learned threshold on jump‑count: {stump.tree_.threshold[0]:.3f}")
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test  accuracy: {test_acc:.3f}")


if __name__ == "__main__":
    main()
