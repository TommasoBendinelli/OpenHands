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


def _boxed_two_panel_plot(df: pd.DataFrame,
                          path_png: Path,
                          path_pdf: Path) -> None:
    """Save a 1×2 boxed sanity-check figure (periodic vs aperiodic)."""
    fig = plt.figure(figsize=(10, 4), dpi=150)
    ax0 = plt.subplot(1, 2, 1)
    ax1 = plt.subplot(1, 2, 2)

    ax0.plot(df[df["label"] == 0].iloc[0, :-1])
    ax0.set_title("Periodic (class 0)", pad=6)
    ax1.plot(df[df["label"] == 1].iloc[0, :-1])
    ax1.set_title("Aperiodic (class 1)", pad=6)

    for ax in (ax0, ax1):
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():          # black frame
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_edgecolor("black")

    plt.tight_layout()
    fig.savefig(path_png)
    fig.savefig(path_pdf)
    plt.close(fig)
# -------------------------------------------------------------- #

def main() -> None:
    np.random.seed(42)
    random.seed(42)
    out_dir = Path(__file__).resolve().parent

    figs_dir = Path("67ff59b20231d9f95909f426") / "figs" / "datasets"
    figs_dir.mkdir(parents=True, exist_ok=True)
    # training & test splits
    train_df = create_dataset(N_SAMPLES_SPLIT, SIGNAL_LEN_TRAIN, FS_DEFAULT)
    test_df  = create_dataset(N_SAMPLES_SPLIT, SIGNAL_LEN_TEST,  FS_DEFAULT)

    save_datasets(train_df, test_df, output_folder=out_dir)

    # boxed two-panel plots (train & test)
    _boxed_two_panel_plot(
        train_df,
        figs_dir / "periodic_vs_aperiodic_train.png",
        figs_dir / "periodic_vs_aperiodic_train.pdf",
    )
    _boxed_two_panel_plot(
        test_df,
        figs_dir / "periodic_vs_aperiodic_test.png",
        figs_dir / "periodic_vs_aperiodic_test.pdf",
    )
    print("✓ CSVs + boxed figures written to", figs_dir)


if __name__ == "__main__":
    main()