#!/usr/bin/env python
"""
sign_rotated_generator.py – variant of `interaction_sign`
with a safety margin around the decision boundary
-------------------------------------------------------

label 1  ⇔  u * v  >  0   AND  |u*v| ≥ MARGIN
label 0  ⇔  u * v  <  0   AND  |u*v| ≥ MARGIN

where
    u =  cosθ · feat1 + sinθ · feat2
    v = −sinθ · feat1 + cosθ · feat2
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import random

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets                     # noqa: E402

# ───────────────────────────────────────────────────────────────────────────
THETA  = np.deg2rad(30)          # hidden rotation angle
COS, SIN = np.cos(THETA), np.sin(THETA)

N_FEATS   = 7                    # 2 signal dims + 5 distractors
N_SAMPLES = 2_000
MARGIN    = 0.10                 # min |u*v| distance from the boundary
NOISE_STD = 0.5                  # σ for distractor features
# ───────────────────────────────────────────────────────────────────────────


def rotate(x: float, y: float) -> tuple[float, float]:
    """(feat1, feat2) → (u, v)"""
    u = COS * x + SIN * y
    v = -SIN * x + COS * y
    return u, v


def generate_sample(label_one: bool) -> np.ndarray:
    """
    Draw (feat1, feat2) until both:
        • u*v has the desired sign         (same / opposite)
        • |u*v| ≥ MARGIN                   (away from the boundary)
    Then append pure-noise features 3..N.
    """
    while True:
        x, y = np.random.normal(0, 1, size=2)
        u, v = rotate(x, y)
        prod = u * v

        # enforce sign *and* margin
        if label_one:
            if prod >  MARGIN:
                break
        else:
            if prod < -MARGIN:
                break

    noise = np.random.normal(0, NOISE_STD, size=N_FEATS - 2)
    return np.concatenate([[x, y], noise])


def create_dataset(n_samples: int, n_feats: int) -> pd.DataFrame:
    data, labels = [], []
    for _ in range(n_samples // 2):
        data.append(generate_sample(True));  labels.append(1)
        data.append(generate_sample(False)); labels.append(0)

    cols = [f"feat{i+1}" for i in range(n_feats)]
    df = pd.DataFrame(data, columns=cols)
    df["label"] = labels
    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parent
    np.random.seed(42)
    random.seed(42)

    train_df = create_dataset(N_SAMPLES, N_FEATS)
    test_df  = create_dataset(N_SAMPLES, N_FEATS)   # test uses same margin

    save_datasets(train_df, test_df, out)

    # visual sanity-check
    plt.figure(figsize=(5, 5))
    same = train_df["label"] == 1
    plt.scatter(train_df.loc[same,  "feat1"],
                train_df.loc[same,  "feat2"], alpha=0.4, label="label 1")
    plt.scatter(train_df.loc[~same, "feat1"],
                train_df.loc[~same, "feat2"], alpha=0.4, label="label 0")
    plt.axhline(0, c='k', lw=0.7); plt.axvline(0, c='k', lw=0.7)
    plt.title("Rotated quadrants with margin |u·v| ≥ {:.2f}".format(MARGIN))
    plt.legend(); plt.tight_layout()
    plt.savefig(out / "sign_rotated_example.png")