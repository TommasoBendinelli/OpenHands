#!/usr/bin/env python
"""
periodic_dataset_generator.py  – v2
Generates *periodic* vs *aperiodic* time-series and normalises each example
to unit variance so that naïve variance/σ features carry no discriminatory
power.  A dominant spectral peak is still present only in class 0.

Classes
-------
    0   periodic   :  sine-wave + noise   (dominant frequency present)
    1   aperiodic  :  coloured noise      (no dominant frequency)

Outputs
-------
train.csv, test.csv (+ *_labels.csv / *_gt.csv) written by utils.save_datasets
plus a sanity-check plot (periodic vs aperiodic) in PNG format.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
# --------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets           # noqa: E402
# --------------------------------------------------------------------------

# ────────────────────────────────────────────────────────────────────────────
FS_DEFAULT = 1.0                         # sampling frequency
PERIODIC_FREQS = [0.05, 0.08, 0.12]      # sine frequencies (Hz)
SIGNAL_LEN_TRAIN = 256
SIGNAL_LEN_TEST  = 512
N_SAMPLES_SPLIT  = 256                   # per split
NOISE_STD        = 0.5                   # σ of base noise before scaling
# ────────────────────────────────────────────────────────────────────────────


def generate_signal(periodic: bool,
                    length: int = SIGNAL_LEN_TRAIN,
                    fs: float = FS_DEFAULT) -> np.ndarray:
    """
    Create either a pure coloured-noise series (aperiodic) or
    a sine-wave embedded in noise (periodic), then **normalise to unit σ**.
    """
    t = np.arange(length) / fs

    # base Gaussian noise
    noise = np.random.normal(0, NOISE_STD, size=length)

    if periodic:
        freq = np.random.choice(PERIODIC_FREQS)
        amp  = np.random.uniform(0.8, 1.5)              # comparable to noise
        x = amp * np.sin(2 * np.pi * freq * t) + noise
    else:
        # coloured noise: add a low-pass filter by cumulative sum
        noise = np.cumsum(noise) / np.sqrt(length)      # pink-ish
        x = noise

    # ── Crucial step: destroy variance-based heuristics ──────────────────
    std = np.std(x)
    if std > 0:
        x = x / std                                     # unit variance
    return x


def create_dataset(n_samples: int,
                   length: int,
                   fs: float = FS_DEFAULT) -> pd.DataFrame:
    """
    Build a balanced (n_samples) dataset and return as DataFrame.
    """
    data, labels = [], []

    for _ in range(n_samples // 2):
        data.append(generate_signal(True,  length, fs)); labels.append(0)
        data.append(generate_signal(False, length, fs)); labels.append(1)

    df = pd.DataFrame(np.vstack(data))
    df["label"] = labels
    return df


def main() -> None:
    np.random.seed(42)
    random.seed(42)
    out_dir = Path(__file__).resolve().parent

    # training & test splits
    train_df = create_dataset(N_SAMPLES_SPLIT, SIGNAL_LEN_TRAIN, FS_DEFAULT)
    test_df  = create_dataset(N_SAMPLES_SPLIT, SIGNAL_LEN_TEST,  FS_DEFAULT)

    save_datasets(train_df, test_df, output_folder=out_dir)

    # Save 6 training examples for sanity-check plot
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        ax.plot(train_df.iloc[i, :-1])
        ax.set_title(f"label {train_df.iloc[i, -1]}")
        ax.set_xlabel("time")
        ax.set_ylabel("amplitude")
        ax.grid()

    fig.tight_layout()
    fig.savefig(out_dir / "periodic_vs_aperiodic.png")
    plt.close(fig)
    # Save 6 test examples for sanity-check plot
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        ax.plot(test_df.iloc[i, :-1])
        ax.set_title(f"label {test_df.iloc[i, -1]}")
        ax.set_xlabel("time")
        ax.set_ylabel("amplitude")
        ax.grid()
    fig.tight_layout()
    fig.savefig(out_dir / "periodic_vs_aperiodic_test.png")
    plt.close(fig)

if __name__ == "__main__":
    main()