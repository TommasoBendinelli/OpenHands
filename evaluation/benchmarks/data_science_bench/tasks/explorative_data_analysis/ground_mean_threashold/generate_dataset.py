#!/usr/bin/env python
"""
generate_data.py  – collision-free dataset generator (patched version)

Hidden rule
-----------
A group is assigned `label = 1` **iff**

        mean(value_0)  >  3.0

Changes vs. the original generator
----------------------------------
* **Only some rows in a positive-label group carry signal**
  – 60 % of the rows are drawn from N(6, 1), the rest from N(0, 1).
  – This keeps the group-mean above 3 but removes any reliable per-row shortcut
    such as “sum of columns” or “row standard deviation”.

* **Distractor columns are pure noise** (N(0, 1) in *every* row),
  so they hold zero direct information about the label.

With these tweaks, you must aggregate by `group_id` (or equivalent) to solve
the task perfectly; single-row heuristics will top out well below 100 %.
"""

import os
import random
import string
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# so that `from utils import save_datasets` works when this file lives in
#   …/my_solution_dir/generate_data.py
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets  # ← helper provided with the task

# ──────────────────────────────────────────────────────────────────────
N_GROUPS = 120
ROWS_PER_GRP = 20
HIGH_ROW_FRAC = 0.60  # fraction of “high-μ” rows in a positive group
LOW_MU = 0.0
HIGH_MU = 6.0
THRESH = 3.0
DISTRACTOR_COLS = 3
# ──────────────────────────────────────────────────────────────────────


def unique_id(used: set) -> str:
    """Return an ID like 'C-17' that hasn’t been used before."""
    while True:
        gid = f'{random.choice(string.ascii_uppercase[:6])}-{random.randint(0, 99):02d}'
        if gid not in used:
            used.add(gid)
            return gid


def generate_dataset() -> pd.DataFrame:
    rows, used = [], set()

    for g in range(N_GROUPS):
        gid = unique_id(used)
        label = g % 2  # 0,1,0,1,…

        # How many rows in this group will use the “high” mean?
        n_high = int(round(ROWS_PER_GRP * HIGH_ROW_FRAC)) if label == 1 else 0

        for r in range(ROWS_PER_GRP):
            if r < n_high:
                mu = HIGH_MU  # contributes to pushing the group mean up
            else:
                mu = LOW_MU

            value_0 = np.random.normal(mu, 1.0)
            distractors = np.random.normal(0.0, 1.0, DISTRACTOR_COLS)  # pure noise
            rows.append([gid, value_0, *distractors, label])

    cols = (
        ['group_id', 'value_0']
        + [f'value_{i + 1}' for i in range(DISTRACTOR_COLS)]
        + ['label']
    )
    return pd.DataFrame(rows, columns=cols)


def combined_group_mean_plot(
    train_df: pd.DataFrame, test_df: pd.DataFrame, threshold: float, out_png: Path
):
    """
    Compare the distribution of group means (value_0) in train and test,
    split and stacked by the true group label.
    """
    # --- compute per-group means and labels ---------------------------
    tr_mean = train_df.groupby('group_id')['value_0'].mean()
    tr_lbl = train_df.groupby('group_id')['label'].first()
    te_mean = test_df.groupby('group_id')['value_0'].mean()
    te_lbl = test_df.groupby('group_id')['label'].first()
    # --- common bin edges --------------------------------------------
    bins = np.linspace(
        min(tr_mean.min(), te_mean.min()), max(tr_mean.max(), te_mean.max()), 30
    )
    centers = (bins[:-1] + bins[1:]) / 2
    # --- histogram counts --------------------------------------------
    tr0, _ = np.histogram(tr_mean[tr_lbl == 0], bins=bins)
    tr1, _ = np.histogram(tr_mean[tr_lbl == 1], bins=bins)
    te0, _ = np.histogram(te_mean[te_lbl == 0], bins=bins)
    te1, _ = np.histogram(te_mean[te_lbl == 1], bins=bins)

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
    # test  (offset right)
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

    # decision boundary
    plt.xlabel('')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    return plt


def main():
    np.random.seed(42)
    random.seed(42)

    out_dir = Path(__file__).resolve().parent
    train = generate_dataset()
    test = generate_dataset()
    save_datasets(train, test, out_dir)
    # ── sanity check: group means separate cleanly at THRESH ──────────
    train.groupby('group_id')['value_0'].mean()
    train.groupby('group_id')['label'].first()
    plt = combined_group_mean_plot(
        train_df=train,
        test_df=test,
        threshold=THRESH,
        out_png=out_dir / 'group_mean_example.png',
    )

    overleaf_path = Path(
        os.environ.get('FIGS_DIR', '67ff59b20231d9f95909f426/figs/datasets')
    )
    plt.savefig(overleaf_path / 'group_mean_example.png', dpi=150)
    plt.savefig(overleaf_path / 'group_mean_example.pdf', dpi=150)
    # plt.hist(grp_mean[grp_label == 0], bins=30, alpha=0.6, label='label 0')
    # plt.hist(grp_mean[grp_label == 1], bins=30, alpha=0.6, label='label 1')
    # plt.axvline(THRESH, ls='--', c='k', label=f'threshold {THRESH}')
    # plt.xlabel('group mean of value_0')
    # plt.ylabel('# groups')
    # plt.title('Patched generator  ⇒  perfect separation *only* by grouping')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(out_dir / 'group_mean_example.png')
    # plt.show()


if __name__ == '__main__':
    main()
