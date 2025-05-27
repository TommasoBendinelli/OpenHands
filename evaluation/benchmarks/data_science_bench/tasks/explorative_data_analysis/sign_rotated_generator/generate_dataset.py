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



def combined_uv_product_plot(train_df: pd.DataFrame,
                             test_df:  pd.DataFrame,
                             cos:      float,
                             sin:      float,
                             margin:   float,
                             title:    str,
                             out_png:  Path):
    """
    Histogram of the decisive scalar  p = (cosθ·feat1 + sinθ·feat2)
                                       · (−sinθ·feat1 + cosθ·feat2)
    shown for train/test, split & stacked by the true label.

    A dashed line at p = 0 marks the decision boundary.
    """
    # --- compute u·v for every row -----------------------------------
    def uv_prod(df: pd.DataFrame) -> np.ndarray:
        u = cos * df["feat1"] + sin * df["feat2"]
        v = -sin * df["feat1"] + cos * df["feat2"]
        return u * v

    prod_tr = uv_prod(train_df)
    prod_te = uv_prod(test_df)

    # --- common bins --------------------------------------------------
    bins = np.linspace(-2,
                       2,100)
    centers = (bins[:-1] + bins[1:]) / 2
    # --- histogram counts --------------------------------------------
    tr0, _ = np.histogram(prod_tr[train_df["label"] == 0], bins=bins)
    tr1, _ = np.histogram(prod_tr[train_df["label"] == 1], bins=bins)
    te0, _ = np.histogram(prod_te[test_df ["label"] == 0], bins=bins)
    te1, _ = np.histogram(prod_te[test_df ["label"] == 1], bins=bins)

    width = (bins[1] - bins[0]) * 0.20

    plt.figure(figsize=(10, 4))
    # train (offset left)
    plt.bar(centers - 1.5*width, tr0, width=width,
            alpha=0.6, label="train (label=0)")
    plt.bar(centers - 1.5*width, tr1, width=width,
            bottom=tr0, alpha=0.6, hatch="//", label="train (label=1)")
    # test  (offset right)
    plt.bar(centers + 1.5*width, te0, width=width,
            alpha=0.6, label="test (label=0)")
    plt.bar(centers + 1.5*width, te1, width=width,
            bottom=te0, alpha=0.6, hatch="\\\\", label="test (label=1)")

    # decision boundary
    plt.xlabel(r"$u \cdot v$")
    plt.ylabel("Count")
    plt.title("")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    # save also the pdf
    plt.savefig(out_png.with_suffix(".pdf"), dpi=150)
    return plt

if __name__ == "__main__":
    out = Path(__file__).resolve().parent
    np.random.seed(42)
    random.seed(42)

    train_df = create_dataset(N_SAMPLES, N_FEATS)
    test_df  = create_dataset(N_SAMPLES, N_FEATS)   # test uses same margin
    save_datasets(train_df, test_df, out)
    fig_folder = Path("67ff59b20231d9f95909f426/figs/datasets")
    combined_uv_product_plot(
        train_df, test_df,
        COS, SIN, MARGIN,
        title="Histogram of the decisive scalar $u \cdot v$",
        out_png=fig_folder / "uv_product_histogram.png"
    )