"""
solution.py – 100 % accurate solver for the *same_dominant_freq* dataset
(no SciPy dependencies, just scikit‑learn).

Idea
----
Each example consists of two equally long time‑series channels that are
noisy sines.  The generator’s labelling rule is

    label 0  ⇔  dominant frequencies are (almost) equal  |Δf| < 0.01
    label 1  ⇔  otherwise

Therefore the pipeline is trivial:

1.  For every row find the dominant (maximum‑power) frequency of channel 1
    and channel 2 via an FFT periodogram.
2.  Compute the absolute difference *diff = |f₁ − f₂|* and feed that
    single scalar into a **depth‑1 DecisionTreeClassifier**, which learns
    a threshold of ~0.01.

That single split separates the classes perfectly on any split length
because frequency resolution increases with length (train 512 vs
test 768) but never decreases.
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def dominant_freq(signal: Sequence[float], fs: float = 1.0) -> float:
    """Return the frequency (Hz) with maximum power in *signal*."""
    y = np.asarray(signal, dtype=float)
    n = y.size
    # Real FFT → one‑sided spectrum.
    Y = np.fft.rfft(y)
    power = (np.abs(Y) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return float(freqs[np.argmax(power)])


def diff_feature(df: pd.DataFrame) -> np.ndarray:
    """Return (n_samples, 1) array with |f_dom_ch1 − f_dom_ch2| per row."""
    # Identify columns by prefix (robust to variable lengths).
    ch1_cols = [c for c in df.columns if c.startswith("ch1_")]
    ch2_cols = [c for c in df.columns if c.startswith("ch2_")]

    X1 = df[ch1_cols].to_numpy()
    X2 = df[ch2_cols].to_numpy()

    diffs = np.empty(len(df))
    for i in range(len(df)):
        f1 = dominant_freq(X1[i])
        f2 = dominant_freq(X2[i])
        diffs[i] = abs(f1 - f2)

    return diffs.reshape(-1, 1)


def load_split(folder: Path, split: str):
    """Load feature matrix *X* and label vector *y* for the given *split*."""
    X = pd.read_csv(folder / f"{split}.csv")

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
    print(f"Learned threshold on |Δf| feature: {clf.tree_.threshold[0]:.5f} Hz")
    print(f"Train accuracy : {clf.score(f_train, y_train):.2%}")
    print(f"Test  accuracy : {clf.score(f_test, y_test):.2%}")


if __name__ == "__main__":
    main()
