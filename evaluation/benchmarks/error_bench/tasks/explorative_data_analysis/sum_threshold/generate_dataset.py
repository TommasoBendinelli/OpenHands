
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


if __name__ == '__main__':
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    output_folder = Path(__file__).resolve().parent
    out_dir = Path(__file__).resolve().parent
    train_df = create_dataset(output_folder=out_dir)
    test_df  = create_dataset(n_samples=1_000, n_feats=6, thresh=1.5,
                              output_folder=out_dir)
    save_datasets(train_df, test_df, out_dir)

    # sanity plot: sum(feat1..3) vs. label
    plt.figure(figsize=(6, 4))
    sums = train_df[['feat1', 'feat2', 'feat3']].sum(axis=1)
    plt.hist(sums[train_df['label'] == 0], bins=30, alpha=0.6, label='label 0')
    plt.hist(sums[train_df['label'] == 1], bins=30, alpha=0.6, label='label 1')
    plt.axvline(1.5, ls='--', label='threshold')
    plt.title('Sum of first 3 features splits the classes'); plt.legend()
    plt.tight_layout(); plt.show()
    plt.savefig(out_dir / 'sum_dataset_example.png')