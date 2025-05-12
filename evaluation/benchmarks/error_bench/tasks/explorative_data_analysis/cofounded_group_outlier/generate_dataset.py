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

import random, string, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets          # your helper
import random
# ─────────────────────────────────────────────
ROWS_PER_GRP      = 40
OUTLIER_CUT       = 3.0
TRUE_THRESH_RATIO = 0.08
DISTRACTOR_COLS   = 2
# ─────────────────────────────────────────────


def unique_id(used: set) -> str:
    while True:
        gid = f"{random.choice(string.ascii_uppercase[:6])}-{random.randint(0,99):02d}"
        if gid not in used:
            used.add(gid); return gid


def make_group(label: int, colour: str, gid: str) -> list[list]:
    """
    Build one group, injecting outliers if label==1.
    """
    base = np.random.normal(0, 1, ROWS_PER_GRP)
    if label == 1:
        k       = np.random.randint(int(.10*ROWS_PER_GRP), int(.15*ROWS_PER_GRP)+1)
        idx     = np.random.choice(ROWS_PER_GRP, k, replace=False)
        base[idx] = np.random.normal(0,1,k) + np.random.choice([-6,6],k)
    rows = []
    for val in base:
        rows.append([gid, colour, val, *np.random.normal(0,1,DISTRACTOR_COLS), label])
    return rows


def build_split(n_groups: int, confound: bool) -> pd.DataFrame:
    """
    If confound=True  →  colour correlates 99 % with the label.
    If confound=False →  colour is 50/50 regardless of label.
    """
    rows, used = [], set()
    for g in range(n_groups):
        gid   = unique_id(used)
        label = g % 2                     # balanced 0/1
        if confound:                      # TRAIN  – inject shortcut
            colour = "red"  if label == 1 else "blue"
        else:                             # TEST   – no correlation
            colour = random.choice(["red", "blue"])
        rows += make_group(label, colour, gid)

    cols = (["group_id", "colour", "signal"] +
            [f"noise{i+1}" for i in range(DISTRACTOR_COLS)] +
            ["label"])
    return pd.DataFrame(rows, columns=cols)


def sanity_plot(df: pd.DataFrame, title: str, out_png: Path):
    out_ratio = (df.assign(out=lambda d: d.signal.abs() > OUTLIER_CUT)
                   .groupby("group_id")["out"].mean())
    grp_lab   = df.groupby("group_id")["label"].first()
    plt.hist(out_ratio[grp_lab==0], bins=30, alpha=.6, label="label 0")
    plt.hist(out_ratio[grp_lab==1], bins=30, alpha=.6, label="label 1")
    plt.axvline(TRUE_THRESH_RATIO, ls="--", c="k")
    plt.legend(); plt.title(title); plt.tight_layout(); plt.savefig(out_png); plt.close()


def main():
    np.random.seed(42)
    random.seed(42)
    out_dir = Path(__file__).resolve().parent

    train = build_split(n_groups=120, confound=True)    # shortcut present
    test  = build_split(n_groups=60,  confound=False)   # shortcut absent

    save_datasets(train, test, out_dir)

    sanity_plot(train, "TRAIN – colour IS a shortcut", out_dir/"train_plot.png")
    sanity_plot(test,  "TEST  – colour NOT a shortcut", out_dir/"test_plot.png")

    print("Files written: train.csv, test.csv + PNG sanity plots")


if __name__ == "__main__":
    main()