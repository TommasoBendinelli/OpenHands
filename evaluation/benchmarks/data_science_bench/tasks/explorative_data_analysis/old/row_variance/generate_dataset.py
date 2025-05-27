"""
row_variance.py  –  perfectly separable version
label 1  ⇔  variance(first 8 features) > 0.7
"""

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets  # noqa: E402

N_FEATS = 10
N_VARS = 8  # how many cols the rule looks at


def generate_sample(high_var: bool) -> np.ndarray:
    """Create one row that is definitely on the correct side of THRESH."""
    x = np.random.normal(0, 1, size=N_FEATS)

    if high_var:
        # inflate a random half of the first 8 features  → large variance
        mask = np.random.choice([0, 1], size=N_VARS, p=[0.5, 0.5]).astype(bool)
        x[:N_VARS][mask] *= np.random.uniform(2.0, 5.0)

        x[N_VARS:] *= np.random.uniform(0.2, 0.4)  # ↓ overall L2-norm
    else:
        x[:N_VARS] *= np.random.uniform(0.2, 0.4)

        x[N_VARS:] *= np.random.uniform(2.0, 5.0)  # ↑ overall L2-norm

    return x


def create_dataset(n_rows=2_000, out_csv='variance_tabular.csv') -> pd.DataFrame:
    rows, labels = [], []
    for _ in range(n_rows // 2):
        rows.append(generate_sample(high_var=True))
        labels.append(1)
        rows.append(generate_sample(high_var=False))
        labels.append(0)

    cols = [f'feat{i+1}' for i in range(N_FEATS)]
    df = pd.DataFrame(rows, columns=cols)
    df['label'] = labels
    return df



def combined_variance_plot(train_df: pd.DataFrame,
                           test_df:  pd.DataFrame,
                           n_vars:   int,
                           title:    str,
                           out_png:  Path):
    """
    Compare row-variance distributions (first `n_vars` features) in train & test,
    stacked by label and offset left/right (train = left bars, test = right).
    """
    # per-row variance (population definition, ddof=0 to match generator)
    
    tr_var = np.linalg.norm(train_df.iloc[:,:n_vars],axis=1) -  np.linalg.norm(train_df.iloc[:,n_vars:],axis=1)
    te_var = np.linalg.norm(test_df.iloc[:,:n_vars],axis=1) - np.linalg.norm(test_df.iloc[:,n_vars:], axis=1)
    tr_lbl = train_df["label"]
    te_lbl = test_df ["label"]

    # common bin edges
    
    bins = np.linspace(min(tr_var.min(), te_var.min()),
                       max(tr_var.max(), te_var.max()),
                       30)
    centers = (bins[:-1] + bins[1:]) / 2

    tr0, _ = np.histogram(tr_var[tr_lbl == 0], bins=bins)
    tr1, _ = np.histogram(tr_var[tr_lbl == 1], bins=bins)
    te0, _ = np.histogram(te_var[te_lbl == 0], bins=bins)
    te1, _ = np.histogram(te_var[te_lbl == 1], bins=bins)

    width = (bins[1] - bins[0]) * 0.2

    plt.figure(figsize=(10, 4))
    # train (offset left)
    plt.bar(centers - 1.5*width, tr0, width=width,
            alpha=0.6, label="train (label=0)")
    plt.bar(centers - 1.5*width, tr1, width=width,
            bottom=tr0, alpha=0.6, hatch="//", label="train (label=1)")
    # test (offset right)
    plt.bar(centers + 1.5*width, te0, width=width,
            alpha=0.6, label="test (label=0)")
    plt.bar(centers + 1.5*width, te1, width=width,
            bottom=te0, alpha=0.6, hatch="\\\\", label="test (label=1)")

    # plt.axvline(thresh, ls="--", c="k", label=f"threshold = {thresh}")
    plt.xlabel(f"row variance of first {n_vars} features")
    plt.ylabel("# rows")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    return plt


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)

    out_dir = Path(__file__).resolve().parent

    train = create_dataset()
    test = create_dataset()
    save_datasets(train, test, out_dir)

    plot = combined_variance_plot(
        train,
        test,
        N_VARS,
        "Row-variance separation: train vs test",
        out_dir / "variance_distribution_combined.png",
    )

    # (optional) copy alongside your other sanity figs
    plot.savefig("67ff59b20231d9f95909f426/figs/datasets/variance_combined.png", dpi=150)
    plot.savefig("67ff59b20231d9f95909f426/figs/datasets/variance_combined.pdf", dpi=150)
