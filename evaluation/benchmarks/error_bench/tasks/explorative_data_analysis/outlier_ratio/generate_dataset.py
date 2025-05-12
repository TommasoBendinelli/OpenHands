#!/usr/bin/env python
"""
group_outlier_ratio.py  –  anomaly-flavoured group-by task
label 1 ⟺ at least 8 % of the group’s rows are extreme outliers (|signal|>3)
"""

import random, string, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets            # your helper

# ──────────────────────────────────────────
N_GROUPS, ROWS_PER_GRP = 100, 50
THRESH_RATIO           = 0.08              # 8 %
OUTLIER_CUTOFF         = 3.0               # |signal| threshold
DISTRACTOR_COLS        = 2                 # extra N(0,1) columns
# ──────────────────────────────────────────


def unique_id(used: set) -> str:
    """Return a never-before-seen ID like 'E-42'."""
    while True:
        gid = f"{random.choice(string.ascii_uppercase[:8])}-{random.randint(0,99):02d}"
        if gid not in used:
            used.add(gid); return gid


def make_group(label: int, gid: str):
    """
    Create one group.  label 0 = normal, label 1 = anomalous.
    In anomalous groups ~10–15 % of rows are ±N(6,1) spikes.
    """
    base   = np.random.normal(0, 1, ROWS_PER_GRP)
    if label == 1:
        k          = np.random.randint(int(.10*ROWS_PER_GRP), int(.15*ROWS_PER_GRP)+1)
        idx        = np.random.choice(ROWS_PER_GRP, k, replace=False)
        base[idx]  = np.random.normal(0, 1, k) + np.random.choice([-6, 6], k)
    rows = []
    for val in base:
        distr = np.random.normal(0, 1, DISTRACTOR_COLS)
        rows.append([gid, val, *distr, label])
    return rows


def generate_dataset() -> pd.DataFrame:
    rows, used = [], set()
    for g in range(N_GROUPS):
        gid   = unique_id(used)
        label = g % 2                 # balanced 0,1
        rows += make_group(label, gid)

    cols = (["group_id", "signal"] +
            [f"noise{i+1}" for i in range(DISTRACTOR_COLS)] +
            ["label"])
    return pd.DataFrame(rows, columns=cols)


def main():
    np.random.seed(42)
    random.seed(42)
    out = Path(__file__).resolve().parent
    train, test = generate_dataset(), generate_dataset()
    save_datasets(train, test, out)

    # sanity plot: distribution of outlier ratios
    def outlier_ratio(df):
        return df.assign(is_out=lambda d: d.signal.abs() > OUTLIER_CUTOFF) \
                 .groupby("group_id")["is_out"].mean()

    r      = outlier_ratio(train)
    y_grp  = train.groupby("group_id")["label"].first()
    plt.hist(r[y_grp==0], bins=30, alpha=.6, label="label 0")
    plt.hist(r[y_grp==1], bins=30, alpha=.6, label="label 1")
    plt.axvline(THRESH_RATIO, ls="--", c="k", label=f"threshold {THRESH_RATIO}")
    plt.xlabel("outlier ratio  |signal|>3"); plt.ylabel("# groups")
    plt.title("Anomalous vs normal groups – clear split at 0.08")
    plt.legend(); plt.tight_layout(); plt.show()
    plt.savefig(out / "outlier_ratio_example.png")


if __name__ == "__main__":
    main()