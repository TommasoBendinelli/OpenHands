#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets   # noqa: E402
import random

def diverging(x_flat: np.ndarray, length: int, slope_thresh: float = 0.01) -> int:
    """
    Fit a line to |ch1 - ch2|; if slope > thresh ⇒ divergent (label 1) else 0.
    """
    ch1, ch2 = x_flat[:length], x_flat[length:]
    gap = np.abs(ch1 - ch2)
    t = np.arange(length)
    m = np.polyfit(t, gap, 1)[0]
    return 1 if m > slope_thresh else 0


def generate_sample(corr: bool, length: int = 300, noise: float = 0.4):
    """
    Two-channel series: either highly correlated or independent.
    A per-channel DC-offset is added so that the label is no longer
    predictable from the difference of means.
    """
    base = np.cumsum(np.random.normal(0, 0.2, size=length))          # smooth walk

    if corr:
        ch1 = base + np.random.normal(0, noise, size=length)
        ch2 = base + np.random.normal(0, noise, size=length)
    else:
        ch1 = np.random.normal(0, 1.0, size=length)
        ch2 = np.random.normal(0, 1.0, size=length)

    # NEW: independent offsets destroy any systematic mean difference
    offset1 = np.random.normal(0, 3.0)      # same distribution for both labels
    offset2 = np.random.normal(0, 3.0)

    ch1 += offset1
    ch2 += offset2

    return np.concatenate([ch1, ch2])

def create_dataset(n_samples=200, length=400, noise=0.3, output_folder = 'diverge_dataset.csv'):
    data, labels = [], []
    for _ in range(n_samples // 2):
        data.append(generate_sample(False, length,  noise=0.2))
        labels.append(0)
        data.append(generate_sample(True, length,  noise=0.2))
        labels.append(1)

    cols = [f'a_{t}' for t in range(length)] + [f'b_{t}' for t in range(length)]
    df = pd.DataFrame(data, columns=cols)
    df['label'] = labels
    return df


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    out_dir = Path(__file__).resolve().parent
    train_df = create_dataset(output_folder=out_dir, noise=0.3)
    test_df  = create_dataset(n_samples=200, output_folder=out_dir, noise=0.2)

    save_datasets(train_df, test_df, out_dir)

    # sanity check
    plt.figure(figsize=(10, 4))
    for i, ttl in zip([0, 1], ['Bounded gap (label 0)', 'Divergent gap (label 1)']):
        plt.subplot(1, 2, i + 1)
        plt.plot(train_df.iloc[i, :train_df.shape[1]//2], label='ch 1')
        plt.plot(train_df.iloc[i, train_df.shape[1]//2:-1], label='ch 2')
        plt.title(ttl); plt.legend()
    plt.tight_layout(); plt.show()
    plt.savefig(out_dir / 'diverge_dataset_example.png')