
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets   # noqa: E402
import random

def is_high_sum(row: np.ndarray, thresh: float = 1.5) -> int:
    """Return 1 if the sum of first three columns > thresh, else 0."""
    return int(row[:3].sum() > thresh)


def generate_sample(high: bool, n_feats: int = 6, thresh: float = 1.5):
    """
    Draw features from N(0,1); adjust first-three so the sum
    lands above/below the threshold deterministically.
    """
    x = np.random.normal(0, 0.3, size=n_feats)       # mostly small values
    shift = np.random.uniform(0.3, 0.6, size=3)
    if high:
        x[:3] += shift + thresh / 3                  # push sum upward
    else:
        x[:3] -= shift                               # pull sum downward
    return x


def create_dataset(n_samples=1_000, n_feats=6, thresh=1.5,
                   output_folder: Path | str = 'sum_dataset.csv'):
    data, labels = [], []
    for _ in range(n_samples // 2):
        data.append(generate_sample(True, n_feats, thresh))
        labels.append(1)

        data.append(generate_sample(False, n_feats, thresh))
        labels.append(0)

    cols = [f'feat{i+1}' for i in range(n_feats)]
    df = pd.DataFrame(data, columns=cols)
    df['label'] = labels
    return df

def combined_sum_plot(train_df: pd.DataFrame,
                      test_df:  pd.DataFrame,
                      out_png:  Path):
    """
    Histogram of   s = feat1 + feat2 + feat3
    for train/test, split & stacked by the true label.
    A dashed line at s = thresh marks the decision boundary.
    """
    # --- compute sums -------------------------------------------------
    s_tr = train_df[["feat1", "feat2", "feat3"]].sum(axis=1)
    s_te = test_df[["feat1", "feat2", "feat3"]].sum(axis=1)

    # --- common bins --------------------------------------------------
    bins = np.linspace(min(s_tr.min(), s_te.min()),
                       max(s_tr.max(), s_te.max()),
                       40)
    centers = (bins[:-1] + bins[1:]) / 2

    tr0, _ = np.histogram(s_tr[train_df["label"] == 0], bins=bins)
    tr1, _ = np.histogram(s_tr[train_df["label"] == 1], bins=bins)
    te0, _ = np.histogram(s_te[test_df  ["label"] == 0], bins=bins)
    te1, _ = np.histogram(s_te[test_df  ["label"] == 1], bins=bins)

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
    plt.xlabel("")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.savefig(out_png.with_suffix('.pdf'), dpi=150)
    return plt


if __name__ == '__main__':
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    output_folder = Path(__file__).resolve().parent
    out_dir = Path(__file__).resolve().parent
    train_df = create_dataset(output_folder=out_dir)
    test_df  = create_dataset(n_samples=1_000, n_feats=6, thresh=1.5,
                              output_folder=out_dir)
    
    overleaf_dir = Path("67ff59b20231d9f95909f426/figs/datasets")

    plot = combined_sum_plot(
        train_df,
        test_df,
        out_png=overleaf_dir / "sum_distribution_combined.png"
    )

    # (optional) stash extra copies with your other figs
    plot.savefig("67ff59b20231d9f95909f426/figs/datasets/sum_combined.png",  dpi=150)
    plot.savefig("67ff59b20231d9f95909f426/figs/datasets/sum_combined.pdf", dpi=150)