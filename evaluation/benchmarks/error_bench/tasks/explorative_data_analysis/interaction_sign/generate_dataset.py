#!/usr/bin/env python
"""
interaction_sign.py – Tabular binary classification:
label 1 ↔ feat1 * feat2 > 0  (same-sign quadrants)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets   # noqa: E402


def same_sign(row: np.ndarray) -> int:
    """Return 1 if feat1 and feat2 share the sign."""
    return int(row[0] * row[1] > 0)


def generate_sample(same: bool, n_feats: int = 5):
    """
    Draw (feat1, feat2) from N(0,1) until sign condition holds.
    Add extra noisy features 3..n.
    """
    while True:
        x, y = np.random.normal(0, 1, size=2)
        if same == (x * y > 0):
            break
    others = np.random.normal(0, 1, size=n_feats - 2)
    return np.concatenate([[x, y], others])


def create_dataset(n_samples=1_500, n_feats=5,
                   output_folder: Path | str = 'sign_dataset.csv'):
    data, labels = [], []
    for _ in range(n_samples // 2):
        data.append(generate_sample(True, n_feats));  labels.append(1)
        data.append(generate_sample(False, n_feats)); labels.append(0)

    cols = [f'feat{i+1}' for i in range(n_feats)]
    df = pd.DataFrame(data, columns=cols); df['label'] = labels
    return df


if __name__ == '__main__':
    out_dir = Path(__file__).resolve().parent
    train_df = create_dataset(output_folder=out_dir)
    test_df  = create_dataset(n_samples=1_500, n_feats=5,
                              output_folder=out_dir)

    save_datasets(train_df, test_df, out_dir)

    # sanity scatter
    plt.figure(figsize=(5, 5))
    same = train_df['label'] == 1
    plt.scatter(train_df.loc[same, 'feat1'], train_df.loc[same, 'feat2'],
                alpha=0.4, label='label 1 (same sign)')
    plt.scatter(train_df.loc[~same, 'feat1'], train_df.loc[~same, 'feat2'],
                alpha=0.4, label='label 0 (different sign)')
    plt.axhline(0, c='k', lw=0.7); plt.axvline(0, c='k', lw=0.7)
    plt.title('Quadrant separation by sign of feat1×feat2'); plt.legend()
    plt.tight_layout(); plt.show()
    plt.savefig(out_dir / 'sign_dataset_example.png')