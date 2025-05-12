#!/usr/bin/env python
"""
generate_peak_dataset.py – peak‑count benchmark **non‑overlapping** v2
=====================================================================
• Equal white‑noise variance in both classes (σ = 0.2 by default).
• **Class 0** – 3 – 4 peaks that satisfy the amplitude/area constraints.
• **Class 1** – 5 – 6 peaks that satisfy the amplitude/area constraints.
• Peaks are *Gaussian‑shaped* and **non‑overlapping**: the ±3·σ support of any
  pair of peaks are disjoint.
• A peak’s amplitude must be ≥ `min_peak_height` (X) and its total area
  (≈ A·σ·√(2π)) must be ≥ `min_peak_area` (Y).

Under these settings the one‑feature decision stump that learns
"count ≥ 4.5" still perfectly separates the two classes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# If utils.save_datasets is one directory up
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets  # type: ignore

# ────────────────────────────────────────────────────────────
#  Helper functions
# ────────────────────────────────────────────────────────────


def _gaussian(x: np.ndarray, mu: float, sigma: float, A: float) -> np.ndarray:
    """Return a Gaussian peak A·exp(−0.5·((x−μ)/σ)²)."""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _sample_non_overlapping_peaks(
    n_points: int,
    n_peaks: int,
    rng: np.random.Generator,
    min_peak_height: float,
    min_peak_area: float,
    min_sigma: float,
    max_sigma: float,
    sep_factor: float = 3.0,  # multiply σ to define the exclusion radius
) -> List[tuple[int, float, float]]:
    """Return a list of (centre, sigma, amplitude) for non‑overlapping peaks."""
    peaks: List[tuple[int, float, float]] = []
    attempts = 0
    max_attempts = 10_000  # safeguard against infinite loops
    while len(peaks) < n_peaks and attempts < max_attempts:
        attempts += 1
        # amplitude first – it influences the minimal σ allowed by area
        amp = rng.uniform(min_peak_height, min_peak_height * 1.5)
        sigma_min_area = min_peak_area / (amp * np.sqrt(2 * np.pi))
        sigma = rng.uniform(max(min_sigma, sigma_min_area), max_sigma)

        # centre must leave sep_factor*σ margin inside the signal range
        left_margin = int(np.ceil(sep_factor * sigma))
        right_margin = n_points - left_margin
        if right_margin <= left_margin:
            raise ValueError("Signal too short for the requested peak parameters.")
        centre = int(rng.integers(left_margin, right_margin))

        # check overlap with existing peaks
        ok = True
        for c_prev, s_prev, _ in peaks:
            if abs(centre - c_prev) <= sep_factor * (sigma + s_prev):
                ok = False
                break
        if ok:
            peaks.append((centre, sigma, amp))

    if len(peaks) < n_peaks:
        raise RuntimeError(
            f"Could only place {len(peaks)} non‑overlapping peaks after {attempts} attempts; "
            "try reducing `n_peaks` or `max_sigma`, or shortening `sep_factor`."
        )
    return peaks


def _generate_peak_signal(
    n_points: int,
    n_peaks: int,
    noise_std: float,
    rng: np.random.Generator,
    min_peak_height: float,
    min_peak_area: float,
    min_sigma: float = 3.0,
    max_sigma: float = 15.0,
) -> np.ndarray:
    """Return a 1‑D signal with *n_peaks* **non‑overlapping** Gaussian peaks plus noise."""
    x = np.arange(n_points)
    sig = np.zeros_like(x, dtype=float)

    peaks = _sample_non_overlapping_peaks(
        n_points,
        n_peaks,
        rng,
        min_peak_height,
        min_peak_area,
        min_sigma,
        max_sigma,
    )

    for centre, sigma, amp in peaks:
        sig += _gaussian(x, mu=centre, sigma=sigma, A=amp)

    sig += rng.normal(scale=noise_std, size=n_points)

    # Add random bias to the signal
    bias = rng.uniform(-5, 5)
    sig += bias
    return sig


def _one_sample(
    n_points: int,
    label: int,
    rng: np.random.Generator,
    noise_std: float,
    min_peak_height: float,
    min_peak_area: float,
    min_pk0: int,
    max_pk0: int,
    min_pk1: int,
    max_pk1: int,
) -> Tuple[np.ndarray, int]:
    n_peaks = (
        rng.integers(min_pk0, max_pk0 + 1)
        if label == 0
        else rng.integers(min_pk1, max_pk1 + 1)
    )
    sig = _generate_peak_signal(
        n_points,
        n_peaks,
        noise_std,
        rng,
        min_peak_height,
        min_peak_area,
    )
    return sig, label


def generate_dataset(
    num_samples: int = 1000,
    n_points: int = 1000,
    noise_std: float = 0.2,
    min_peak_height: float = 2.0,  #  X – minimum allowed peak amplitude
    min_peak_area: float = 5.0,    #  Y – minimum allowed peak area
    min_pk0: int = 3,
    max_pk0: int = 4,
    min_pk1: int = 5,
    max_pk1: int = 6,
    seed: int | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with one time‑series per row and a `label` column."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for _ in range(num_samples):
        lbl = rng.integers(0, 2)  # 0 or 1 with equal prob
        sig, lab = _one_sample(
            n_points,
            lbl,
            rng,
            noise_std,
            min_peak_height,
            min_peak_area,
            min_pk0,
            max_pk0,
            min_pk1,
            max_pk1,
        )
        rows.append(sig)
        labels.append(lab)

    df = pd.DataFrame(np.vstack(rows))
    df["label"] = labels
    return df


# ────────────────────────────────────────────────────────────
#  Script entry‑point
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    out_dir = Path(__file__).resolve().parent

    # identical noise & peak specs in both splits
    train_df = generate_dataset(seed=0)
    test_df = generate_dataset(seed=1)

    save_datasets(train_df=train_df, test_df=test_df, output_folder=out_dir)

    # quick visual sanity‑check
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
    ax[0].plot(train_df[train_df["label"] == 0].iloc[0, :-1])
    ax[0].set_title("class 0: 3–4 peaks (non‑overlapping)")
    ax[0].axis("off")
    ax[1].plot(train_df[train_df["label"] == 1].iloc[0, :-1])
    ax[1].set_title("class 1: 5–6 peaks (non‑overlapping)")
    ax[1].axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / "peaks.png")
