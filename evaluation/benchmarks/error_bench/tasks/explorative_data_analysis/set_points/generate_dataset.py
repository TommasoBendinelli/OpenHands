#!/usr/bin/env python
"""
generate_dataset.py – jump-count-only benchmark v2 (bug-fixed)
==============================================================
• Same white-noise variance in both classes (σ = 0.2 by default).  
• **Class 0** – 0 set-points → jump-count ≈ 0.  
• **Class 1** – 4 – 6 set-points → jump-count ≥ 4.  
• Every jump is ±6.0, so |Δx| > 4·σ̂ is always detected.

With these settings the one-feature decision stump in `solution.py`
learns “count ≥ 2.5” and scores 100 % on train & test.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
# If utils.save_datasets is one directory up
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets                      # type: ignore

# ────────────────────────────────────────────────────────────
#  Helper functions
# ────────────────────────────────────────────────────────────
def _next_mean(prev: float, rng: np.random.Generator,
               jump_mag: float, low: float, high: float) -> float:
    """Return a new mean exactly ±jump_mag away from *prev*, reflecting at edges."""
    m = prev + rng.choice([-jump_mag, jump_mag])
    if m < low or m > high:            # reflect if we hit the wall
        m = prev - (m - prev)
    return m


def generate_synthetic_setpoint_signal(
    n_points: int,
    n_setpoints: int,
    noise_std: float,
    jump_mag: float,
    rng: np.random.Generator,
    low: float = -3.0,
    high: float = 3.0,
) -> np.ndarray:
    """Piece-wise constant signal with *equal* jumps of magnitude *jump_mag*."""
    cps = (np.sort(rng.choice(np.arange(1, n_points - 1),
                              size=n_setpoints, replace=False))
           if n_setpoints else np.empty(0, dtype=int))
    cps = np.append(cps, n_points).astype(int)      # ensure integer indices

    sig = np.empty(n_points)
    start = 0
    mean = rng.uniform(low, high)
    for cp in cps:
        sig[start:cp] = mean
        if n_setpoints:                             # update mean only if we need jumps
            mean = _next_mean(mean, rng, jump_mag, low, high)
        start = cp

    sig += rng.normal(scale=noise_std, size=n_points)
    return sig


def _one_sample(
    n_points: int,
    label: int,
    rng: np.random.Generator,
    noise_std: float,
    jump_mag: float,
    min_sp1: int,
    max_sp1: int,
) -> Tuple[np.ndarray, int]:
    n_sp = 0 if label == 0 else rng.integers(min_sp1, max_sp1 + 1)
    sig = generate_synthetic_setpoint_signal(
        n_points, n_sp, noise_std, jump_mag, rng
    )
    return sig, label


def generate_dataset(
    num_samples: int = 1000,
    n_points: int = 1000,
    noise_std: float = 0.2,
    jump_mag: float = 6.0,
    min_sp1: int = 4,
    max_sp1: int = 6,
    seed: int | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with one time-series per row and a `label` column."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for _ in range(num_samples):
        lbl = rng.integers(0, 2)                     # 0 or 1 with equal prob
        sig, lab = _one_sample(
            n_points, lbl, rng, noise_std, jump_mag, min_sp1, max_sp1
        )
        rows.append(sig)
        labels.append(lab)

    df = pd.DataFrame(np.vstack(rows))
    df["label"] = labels
    return df


# ────────────────────────────────────────────────────────────
#  Script entry-point
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    out_dir = Path(__file__).resolve().parent

    # identical noise & jump size in both splits
    train_df = generate_dataset(seed=0)
    test_df  = generate_dataset(seed=1)

    save_datasets(train_df=train_df, test_df=test_df, output_folder=out_dir)

    # quick visual sanity-check
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
    ax[0].plot(train_df[train_df["label"] == 0].iloc[0, :-1])
    ax[0].set_title("class 0: 0 jumps"); ax[0].axis("off")
    ax[1].plot(train_df[train_df["label"] == 1].iloc[0, :-1])
    ax[1].set_title("class 1: 4–6 jumps"); ax[1].axis("off")
    plt.tight_layout()
    plt.show()