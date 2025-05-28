#!/usr/bin/env python
"""
generate_ts_dataset.py – final version
All rows are *demeaned and scaled to unit variance*, so no simple magnitude
feature can leak the label.
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

sys.path.append(str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
from utils import save_datasets


def _boxed_two_panel_plot(df, path_png, path_pdf, titles):
    fig = plt.figure(figsize=(10, 4), dpi=150)
    ax0 = plt.subplot(1, 2, 1)
    ax1 = plt.subplot(1, 2, 2)

    # one example per class
    ax0.plot(df[df['label'] == 1].iloc[0, :-1])  # stationary → label 1
    ax1.plot(df[df['label'] == 0].iloc[0, :-1])  # RW → label 0

    ax0.set_title(titles[0], pad=6)
    ax1.set_title(titles[1], pad=6)

    for ax in (ax0, ax1):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():  # black frame
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_edgecolor('black')

    plt.tight_layout()
    fig.savefig(path_png)
    fig.savefig(path_pdf)
    plt.close(fig)

    # destination folder (same pattern as the other datasets)
    figs_dir = Path(
        os.environ.get('FIGS_DIR', '67ff59b20231d9f95909f426/figs/datasets')
    )

    # train split
    _boxed_two_panel_plot(
        train_df,
        figs_dir / 'stationary_vs_nonstationary_train.png',
        figs_dir / 'stationary_vs_nonstationary_train.pdf',
        titles=('stationary AR(1) (class 1)', 'random walk (class 0)'),
    )
    # test split
    _boxed_two_panel_plot(
        test_df,
        figs_dir / 'stationary_vs_nonstationary_test.png',
        figs_dir / 'stationary_vs_nonstationary_test.pdf',
        titles=('stationary AR(1) (class 1)', 'random walk (class 0)'),
    )
    print('✓ boxed figures written to', figs_dir)


# ──────────────────────────────────────────────────────────────────────────────
def adfuller_stationarity_test(series, significance: float = 0.05) -> bool:
    """Print and return the ADF decision (helper, unchanged)."""
    stat, p, _, _, crit_vals, _ = adfuller(series.values)
    print(f'ADF statistic: {stat:.3f}   p-value: {p:.3g}')
    if p <= significance:
        print('→ stationary')
        return True
    print('→ non-stationary')
    return False


# ──────────────────────────────────────────────────────────────────────────────
def _standardise(ts: np.ndarray) -> np.ndarray:
    """Return (ts − mean) / std  – always mean 0, std 1."""
    ts = ts - ts.mean()
    return ts / ts.std(ddof=1)


def generate_synthetic_ts_signal(
    n_steps: int,
    stationary: bool,
    std: float = 1.0,
    phi: float = 0.6,  # AR(1) coefficient
) -> np.ndarray:
    """
    • stationary      →  AR(1)  x_t = φ x_{t−1} + ε_t
    • non-stationary  →  random walk  x_t = x_{t−1} + ε_t
    ε_t ~ N(0, σ²).  Finally, standardise the whole series.
    """
    ε = np.random.normal(scale=std, size=n_steps)

    if stationary:
        x = np.empty(n_steps)
        x[0] = ε[0] / (1 - phi)
        for t in range(1, n_steps):
            x[t] = phi * x[t - 1] + ε[t]
    else:
        x = ε.cumsum()

    return _standardise(x)  # ← **critical anti-leak step**


def generate_dataset(
    n_samples: int = 300,
    n_steps: int = 1_000,
    std: float = 1.0,
) -> pd.DataFrame:
    """Return DataFrame with n_steps columns + ‘label’."""
    rows, labels = [], []
    for _ in range(n_samples):
        is_stat = bool(np.random.randint(0, 2))
        rows.append(generate_synthetic_ts_signal(n_steps, is_stat, std))
        labels.append(1 if is_stat else 0)
    df = pd.DataFrame(np.vstack(rows))
    df['label'] = labels
    return df


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    output_dir = Path(__file__).resolve().parent

    # build train / test splits (same σ for all → no row-mean leak)
    train_df = generate_dataset(n_samples=600, std=2.0)
    test_df = generate_dataset(n_samples=300, std=2.0)

    print('Train preview:')
    print(train_df.head())

    save_datasets(train_df=train_df, test_df=test_df, output_folder=output_dir)

    # optional: quick sanity check that means are ~0
    print('\nRow-wise means should be ~0 (first 5 rows):')
    print(train_df.iloc[:5, :-1].mean(axis=1).round(6).values)

    # quick ADF demo
    print('\nADF on first training series:')
    adfuller_stationarity_test(train_df.iloc[0, :-1])

    # Plotting
    _boxed_two_panel_plot(
        train_df,
        output_dir / 'stationary_vs_nonstationary_train.png',
        output_dir / 'stationary_vs_nonstationary_train.pdf',
        titles=('Stationary AR(1) (class 1)', 'Random walk (class 0)'),
    )
