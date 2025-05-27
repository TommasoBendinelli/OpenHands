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

def combined_sanity_plot(train_df: pd.DataFrame,
                         test_df:  pd.DataFrame,
                         title:    str,
                         out_png:  Path):
    # compute Δ for train/test
    diff_tr = train_df["3"] - train_df[["1", "2"]].max(axis=1)
    diff_te = test_df ["3"] - test_df [["1", "2"]].max(axis=1)

    # common bins
    bins = np.linspace(
        min(diff_tr.min(), diff_te.min()),
        max(diff_tr.max(), diff_te.max()),
        30
    )
    centers = (bins[:-1] + bins[1:]) / 2
    # hist counts
    tr0, _ = np.histogram(diff_tr[train_df["label"]==0], bins=bins)
    tr1, _ = np.histogram(diff_tr[train_df["label"]==1], bins=bins)
    te0, _ = np.histogram(diff_te [test_df["label"] ==0], bins=bins)
    te1, _ = np.histogram(diff_te [test_df["label"] ==1], bins=bins)

    width = (bins[1] - bins[0]) * 0.2

    plt.figure(figsize=(10, 4))
    # train (−offset)
    plt.bar(centers - 1.5*width, tr0, width=width, alpha=0.6, label="train (label=0)")
    plt.bar(centers - 1.5*width, tr1, width=width,
            bottom=tr0, alpha=0.6, hatch="//", label="train (label=1)")
    # test (+offset)
    plt.bar(centers + 1.5*width, te0, width=width, alpha=0.6, label="test (label=0)")
    plt.bar(centers + 1.5*width, te1, width=width,
            bottom=te0, alpha=0.6, hatch="\\\\", label="test (label=1)")

    plt.xlabel("Feature3 − max(Feature1,Feature2)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return plt


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    out_dir = Path(__file__).resolve().parent
    train_df = create_dataset(output_folder=out_dir)
    test_df = create_dataset(n_samples=1_200, n_feats=6, output_folder=out_dir)

    save_datasets(train_df, test_df, out_dir)

    plot = combined_sanity_plot(
        train_df,
        test_df,
        "",
        out_dir / "delta_distribution_combined.png"
    )
    plot.savefig(
        out_dir / "delta_distribution_combined.png",
        dpi=150
    )

    # Save also in the common folder
    plot.savefig(
        "67ff59b20231d9f95909f426/figs/datasets/dominant_feature.png",
        dpi=150
    )
    plot.savefig(
        "67ff59b20231d9f95909f426/figs/datasets/dominant_feature.pdf",
        dpi=150
    )