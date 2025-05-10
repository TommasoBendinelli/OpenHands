#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets   # noqa: E402


def diverging(x_flat: np.ndarray, length: int, slope_thresh: float = 0.01) -> int:
    """
    Fit a line to |ch1 - ch2|; if slope > thresh ⇒ divergent (label 1) else 0.
    """
    ch1, ch2 = x_flat[:length], x_flat[length:]
    gap = np.abs(ch1 - ch2)
    t = np.arange(length)
    m = np.polyfit(t, gap, 1)[0]
    return 1 if m > slope_thresh else 0


def generate_sample(diverge: bool, length: int = 400, noise: float = 0.3):
    """
    Correlated random walk; optionally add an increasing offset to ch2.
    """
    base = np.cumsum(np.random.normal(0, 0.2, size=length))
    ch1 = base + np.random.normal(0, noise, size=length)
    if diverge:
        drift = np.linspace(0, np.random.uniform(5, 10), length)
        ch2 = base + drift + np.random.normal(0, noise, size=length)
    else:
        ch2 = base + np.random.normal(0, noise, size=length)
    return np.concatenate([ch1, ch2])


def create_dataset(n_samples=200, length=400, output_folder: Path | str = 'diverge_dataset.csv'):
    data, labels = [], []
    for _ in range(n_samples // 2):
        data.append(generate_sample(False, length))
        labels.append(0)
        data.append(generate_sample(True, length))
        labels.append(1)

    cols = [f'ch1_{t}' for t in range(length)] + [f'ch2_{t}' for t in range(length)]
    df = pd.DataFrame(data, columns=cols)
    df['label'] = labels
    return df


if __name__ == '__main__':
    out_dir = Path(__file__).resolve().parent
    train_df = create_dataset(output_folder=out_dir)
    test_df  = create_dataset(n_samples=200, length=600, output_folder=out_dir)

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