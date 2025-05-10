#!/usr/bin/env python
"""
feature_compare.py – Tabular binary classification:
label 1 ↔ feat1 > feat2
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets   # noqa: E402


def first_gt_second(row: np.ndarray) -> int:
    """Return 1 if feat1 > feat2 else 0."""
    return int(row[0] > row[1])


def generate_sample(gt: bool, n_feats: int = 5):
    """
    Draw (feat1, feat2) so that their inequality is guaranteed,
    plus extra noisy features 3..n.
    """
    gap = np.random.uniform(0.2, 1.0)
    if gt:               # want feat1 > feat2
        feat1, feat2 = np.random.normal(0, 1), np.random.normal(0, 1) - gap
    else:                # want feat1 ≤ feat2
        feat1, feat2 = np.random.normal(0, 1) - gap, np.random.normal(0, 1)
    others = np.random.normal(0, 1, size=n_feats - 2)
    return np.concatenate([[feat1, feat2], others])


def create_dataset(n_samples=1_000, n_feats=5,
                   output_folder: Path | str = 'compare_dataset.csv'):
    data, labels = [], []
    for _ in range(n_samples // 2):
        data.append(generate_sample(True, n_feats));  labels.append(1)
        data.append(generate_sample(False, n_feats)); labels.append(0)

    cols = [f'feat{i+1}' for i in range(n_feats)]
    df = pd.DataFrame(data, columns=cols)
    df['label'] = labels
    return df


if __name__ == '__main__':
    out_dir = Path(__file__).resolve().parent
    train_df = create_dataset(output_folder=out_dir)
    test_df  = create_dataset(n_samples=1_000, n_feats=5, output_folder=out_dir)

    save_datasets(train_df, test_df, out_dir)

    # sanity plot: feat1 vs feat2 coloured by label
    plt.figure(figsize=(5, 5))
    plt.scatter(train_df.loc[train_df['label'] == 0, 'feat1'],
                train_df.loc[train_df['label'] == 0, 'feat2'],
                alpha=0.4, label='label 0')
    plt.scatter(train_df.loc[train_df['label'] == 1, 'feat1'],
                train_df.loc[train_df['label'] == 1, 'feat2'],
                alpha=0.4, label='label 1')
    lim = plt.gca().get_xlim()
    plt.plot(lim, lim, 'k--', label='feat1 = feat2')
    plt.xlabel('feat1'); plt.ylabel('feat2')
    plt.title('feat1 > feat2 decides the class'); plt.legend()
    plt.tight_layout(); plt.show()
    plt.savefig(out_dir / 'compare_dataset_example.png')