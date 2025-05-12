#!/usr/bin/env python
"""
row_max_abs.py  –  Binary classification:
label 1  ⇔  max(|feat1…feat12|) > 4.0
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets   # noqa: E402

# -------------------------------------------------------------------
THRESH   = 4.0      # decision boundary
N_FEATS  = 12
N_ROWS   = 2_000
# -------------------------------------------------------------------


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


def create_dataset(n_rows=N_ROWS, out_csv="maxabs_tabular.csv") -> pd.DataFrame:
    rows, labels = [], []
    for _ in range(n_rows // 2):
        rows.append(generate_sample(has_outlier=True));  labels.append(1)
        rows.append(generate_sample(has_outlier=False)); labels.append(0)
    cols = [f"feat{i+1}" for i in range(N_FEATS)]
    df = pd.DataFrame(rows, columns=cols)
    df["label"] = labels
    return df


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    out_dir = Path(__file__).resolve().parent
    train = create_dataset()
    test  = create_dataset()

    save_datasets(train, test, out_dir)

    # ------------------- sanity plot -------------------------------------
    row_max = train.filter(regex="^feat").abs().max(axis=1)
    plt.hist(row_max[train.label == 0], bins=40, alpha=0.6, label="label 0")
    plt.hist(row_max[train.label == 1], bins=40, alpha=0.6, label="label 1")
    plt.axvline(THRESH, ls="--", color="k", label=f"threshold = {THRESH}")
    plt.title("Row-wise max(|x|) cleanly separates the classes")
    plt.xlabel("max(|feature|)"); plt.legend()
    plt.tight_layout(); plt.show()
    plt.savefig(out_dir / "maxabs_example.png")