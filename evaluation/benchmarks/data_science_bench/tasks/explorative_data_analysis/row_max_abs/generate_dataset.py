#!/usr/bin/env python
"""
row_max_abs.py  –  Binary classification:
label 1  ⇔  max(|feat1…feat12|) > 4.0
"""

import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets  # noqa: E402

# -------------------------------------------------------------------
THRESH = 4.0  # decision boundary
N_FEATS = 12
N_ROWS = 2_000
# -------------------------------------------------------------------


def combined_maxabs_plot(
    train_df: pd.DataFrame, test_df: pd.DataFrame, title: str, out_png: Path
):
    """
    Compare the distribution of max-absolute feature values in train vs test,
    stacked by label and offset left/right (train left, test right).
    """
    # per-row metric
    tr_max = train_df.filter(like='feat').abs().max(axis=1)
    te_max = test_df.filter(like='feat').abs().max(axis=1)

    # labels
    tr_lbl = train_df['label']
    te_lbl = test_df['label']

    # common bin edges
    bins = np.linspace(
        min(tr_max.min(), te_max.min()), max(tr_max.max(), te_max.max()), 30
    )
    centers = (bins[:-1] + bins[1:]) / 2

    # histogram counts
    tr0, _ = np.histogram(tr_max[tr_lbl == 0], bins=bins)
    tr1, _ = np.histogram(tr_max[tr_lbl == 1], bins=bins)
    te0, _ = np.histogram(te_max[te_lbl == 0], bins=bins)
    te1, _ = np.histogram(te_max[te_lbl == 1], bins=bins)

    width = (bins[1] - bins[0]) * 0.2

    plt.figure(figsize=(10, 4))
    # train (offset left)
    plt.bar(centers - 1.5 * width, tr0, width=width, alpha=0.6, label='train (label=0)')
    plt.bar(
        centers - 1.5 * width,
        tr1,
        width=width,
        bottom=tr0,
        alpha=0.6,
        hatch='//',
        label='train (label=1)',
    )
    # test (offset right)
    plt.bar(centers + 1.5 * width, te0, width=width, alpha=0.6, label='test (label=0)')
    plt.bar(
        centers + 1.5 * width,
        te1,
        width=width,
        bottom=te0,
        alpha=0.6,
        hatch='\\\\',
        label='test (label=1)',
    )

    plt.xlabel('')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    return plt


def generate_sample(has_outlier: bool) -> np.ndarray:
    """Return one row that is definitely on the correct side of THRESH."""
    if has_outlier:
        # baseline noise
        x = np.random.normal(0, 1, size=N_FEATS)
        # pick a random column and plant an extreme spike
        idx = np.random.randint(N_FEATS)
        spike = np.random.uniform(5.0, 8.0) * np.random.choice([-1, 1])
        x[idx] = spike
        return x

    # ---- class 0: guarantee *no* value crosses the threshold ------------
    # Draw until all |x| ≤ 3.5  (rarely needs more than 1–2 tries)
    while True:
        x = np.random.normal(0, 1, size=N_FEATS)
        if np.abs(x).max() <= 3.5:
            return x


def create_dataset(n_rows=N_ROWS, out_csv='maxabs_tabular.csv') -> pd.DataFrame:
    rows, labels = [], []
    for _ in range(n_rows // 2):
        rows.append(generate_sample(has_outlier=True))
        labels.append(1)
        rows.append(generate_sample(has_outlier=False))
        labels.append(0)
    cols = [f'feat{i + 1}' for i in range(N_FEATS)]
    df = pd.DataFrame(rows, columns=cols)
    df['label'] = labels
    return df


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)

    out_dir = Path(__file__).resolve().parent
    train = create_dataset()
    test = create_dataset()

    save_datasets(train, test, out_dir)
    plot = combined_maxabs_plot(
        train,
        test,
        '',
        out_dir / 'maxabs_distribution_combined.png',
    )

    # (optional) clone to your common figs folder
    figs_dir = Path(
        os.environ.get('FIGS_DIR', '67ff59b20231d9f95909f426/figs/datasets')
    )
    plot.savefig(figs_dir / 'maxabs_combined.png', dpi=150)
    plot.savefig(figs_dir / 'maxabs_combined.pdf', dpi=150)
