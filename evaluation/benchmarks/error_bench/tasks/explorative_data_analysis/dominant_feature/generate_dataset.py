#!/usr/bin/env python
"""
dominant_feature.py – Tabular binary classification:
label 1 ↔ feat3 is maximal among feat1..feat3
"""

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets  # noqa: E402


def feature3_is_max(row: np.ndarray) -> int:
    """Return 1 if feat3 is strictly greater than feat1 and feat2."""
    return int(row[2] > max(row[0], row[1]))


def generate_sample(f3max: bool, n_feats: int = 6):
    """
    Draw first three features so that feat3 is (or isn't) the max.
    Extras 4..n are noise.
    """
    base = np.random.normal(0, 1, size=3)
    gap = np.random.uniform(0.5, 1.5)
    if f3max:
        base[2] = max(base[0], base[1]) + gap
    else:
        base[2] = min(base[0], base[1]) - gap
    others = np.random.normal(0, 1, size=n_feats - 3)
    return np.concatenate([base, others])


def create_dataset(n_samples=1_200, n_feats=6, output_folder='dominant_dataset.csv'):
    data, labels = [], []
    for _ in range(n_samples // 2):
        data.append(generate_sample(True, n_feats))
        labels.append(1)
        data.append(generate_sample(False, n_feats))
        labels.append(0)

    cols = [f'{i+1}' for i in range(n_feats)]
    df = pd.DataFrame(data, columns=cols)
    df['label'] = labels
    return df


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    out_dir = Path(__file__).resolve().parent
    train_df = create_dataset(output_folder=out_dir)
    test_df = create_dataset(n_samples=1_200, n_feats=6, output_folder=out_dir)

    save_datasets(train_df, test_df, out_dir)

    # sanity plot – 3-D scatter projected to 2-D
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    idx0 = train_df['label'] == 0
    ax.scatter(
        train_df.loc[idx0, '1'],
        train_df.loc[idx0, '2'],
        train_df.loc[idx0, '3'],
        alpha=0.4,
        label='label 0',
    )
    ax.scatter(
        train_df.loc[~idx0, '1'],
        train_df.loc[~idx0, '2'],
        train_df.loc[~idx0, '3'],
        alpha=0.4,
        label='label 1',
    )
    ax.set_xlabel('1')
    ax.set_ylabel('2')
    ax.set_zlabel('3')
    ax.set_title('3 dominant vs not')
    ax.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(out_dir / 'dominant_dataset_example.png')
