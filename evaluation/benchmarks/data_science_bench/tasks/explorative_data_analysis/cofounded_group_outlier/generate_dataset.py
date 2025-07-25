#!/usr/bin/env python
"""
confounded_group_outlier.py
Binary classification with a TRAIN-ONLY shortcut feature.

Real rule  (group-level):
    label = 1  ⇔  outlier ratio  ≥ 0.08   where outlier = |signal| > 3

Spurious shortcut (TRAIN only):
    colour == "red"  ⇔  label 1
    colour == "blue" ⇔  label 0

A model that relies on 'colour' scores ~100 % on train but ~50 % on test.
"""

import os
import random
import string
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets  # your helper

# ─────────────────────────────────────────────
ROWS_PER_GRP = 40
OUTLIER_CUT = 3.0
TRUE_THRESH_RATIO = 0.08
DISTRACTOR_COLS = 2
# ─────────────────────────────────────────────


def unique_id(used: set) -> str:
    while True:
        gid = f'{random.choice(string.ascii_uppercase[:6])}-{random.randint(0, 99):02d}'
        if gid not in used:
            used.add(gid)
            return gid


def make_group(label: int, colour: str, gid: str) -> list[list]:
    """
    Build one group, injecting outliers if label==1.
    """
    base = np.random.normal(0, 1, ROWS_PER_GRP)
    # Resample if there are outliers in the base
    while np.abs(base).max() > OUTLIER_CUT:
        base = np.random.normal(0, 1, ROWS_PER_GRP)

    if label == 1:
        k = np.random.randint(int(0.10 * ROWS_PER_GRP), int(0.15 * ROWS_PER_GRP) + 1)
        idx = np.random.choice(ROWS_PER_GRP, k, replace=False)
        base[idx] = np.random.normal(0, 1, k) + np.random.choice([-6, 6], k)
    rows = []
    for val in base:
        rows.append([gid, colour, val, *np.random.normal(0, 1, DISTRACTOR_COLS), label])

    return rows


def build_split(n_groups: int, confound: bool) -> pd.DataFrame:
    """
    If confound=True  →  colour correlates 100 % with the label.
    If confound=False →  colour is 50/50 regardless of label.
    """
    rows, used = [], set()
    for g in range(n_groups):
        gid = unique_id(used)
        label = g % 2  # balanced 0/1
        if confound:  # TRAIN  – inject shortcut
            colour = 'red' if label == 1 else 'blue'
        else:  # TEST   – no correlation
            colour = random.choice(['red', 'blue'])
        rows += make_group(label, colour, gid)

    cols = (
        ['group_id', 'colour', 'signal']
        + [f'noise{i + 1}' for i in range(DISTRACTOR_COLS)]
        + ['label']
    )

    return pd.DataFrame(rows, columns=cols)


def sanity_plot(df: pd.DataFrame, title: str, out_png: Path):
    out_ratio = (
        df.assign(out=lambda d: d.signal.abs() > OUTLIER_CUT)
        .groupby('group_id')['out']
        .mean()
    )
    grp_lab = df.groupby('group_id')['label'].first()
    plt.hist(out_ratio[grp_lab == 0], bins=30, alpha=0.6, label='label 0')
    plt.hist(out_ratio[grp_lab == 1], bins=30, alpha=0.6, label='label 1')
    plt.axvline(TRUE_THRESH_RATIO, ls='--', c='k')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    np.random.seed(42)
    random.seed(42)
    out_dir = Path(__file__).resolve().parent

    train = build_split(n_groups=120, confound=True)  # shortcut present
    test = build_split(n_groups=60, confound=False)  # shortcut absent

    save_datasets(train, test, out_dir)

    # Create a barplot of counting the instances of outliers per group

    is_outlier = (train['signal'].abs() > OUTLIER_CUT).astype(int)
    ratio = is_outlier.groupby(train['group_id']).transform('mean')
    train['ratio'] = ratio

    is_outlier = (test['signal'].abs() > OUTLIER_CUT).astype(int)
    ratio = is_outlier.groupby(test['group_id']).transform('mean')
    test['ratio'] = ratio

    # Round to two decimal places
    train['ratio'] = np.round(train['ratio'], 2)
    test['ratio'] = np.round(test['ratio'], 2)

    # Keep only one row per group
    train = train.drop_duplicates(subset=['group_id'])
    test = test.drop_duplicates(subset=['group_id'])

    # Create a barplot where the x_axis is the unique ratio value of each group and the y_axis is the number of instances of that ratio
    plt.figure(figsize=(10, 4))
    # Compute full range of ratio values (to two decimal places)
    x_min = min(train['ratio'].min(), test['ratio'].min())
    x_max = max(train['ratio'].max(), test['ratio'].max())
    x_vals = np.round(np.arange(x_min, x_max + 0.005, 0.01), 2)

    # Total counts
    train_counts = (
        train['ratio'].value_counts().sort_index().reindex(x_vals, fill_value=0)
    )
    test_counts = (
        test['ratio'].value_counts().sort_index().reindex(x_vals, fill_value=0)
    )

    # Counts for label == 1
    train_label1 = (
        train.loc[train['label'] == 1, 'ratio']
        .value_counts()
        .sort_index()
        .reindex(x_vals, fill_value=0)
    )
    test_label1 = (
        test.loc[test['label'] == 1, 'ratio']
        .value_counts()
        .sort_index()
        .reindex(x_vals, fill_value=0)
    )

    # Compute counts for label == 0 by subtraction
    train_label0 = train_counts - train_label1
    test_label0 = test_counts - test_label1

    # Plot
    plt.figure(figsize=(10, 4))
    width = 0.004  # half the bin width

    # --- TRAIN bars ---
    # Label 0 in solid fill
    plt.bar(
        x_vals - width,
        train_label0,
        width=width * 2,
        alpha=0.6,
        label='train (label=0)',
    )
    # Label 1 on top, with hatching
    plt.bar(
        x_vals - width,
        train_label1,
        width=width * 2,
        bottom=train_label0,
        alpha=0.6,
        hatch='//',
        label='train (label=1)',
    )

    # --- TEST bars ---
    plt.bar(
        x_vals + width,
        test_label0,
        width=width * 2,
        alpha=0.6,
        label='test (label=0)',
    )
    plt.bar(
        x_vals + width,
        test_label1,
        width=width * 2,
        bottom=test_label0,
        alpha=0.6,
        hatch='\\\\',
        label='test (label=1)',
    )

    # Ticks: show every 5th to avoid overlap
    tick_step = 5
    plt.xticks(x_vals[::tick_step], rotation=90)

    plt.xlabel('Ratio')
    plt.ylabel('Count')
    plt.title('')
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_dir / 'ratio_distribution.png')
    print('Files written: train.csv, test.csv + PNG sanity plots')

    # Save also in FIGS_DIR if provided
    figs_dir = Path(
        os.environ.get('FIGS_DIR', '67ff59b20231d9f95909f426/figs/datasets')
    )
    plt.savefig(figs_dir / 'ratio_distribution.png')
    plt.savefig(figs_dir / 'ratio_distribution.pdf')


if __name__ == '__main__':
    main()
