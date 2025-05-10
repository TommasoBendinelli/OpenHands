#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets   # noqa: E402


def has_spike(x: np.ndarray, z_thresh: float = 4.0) -> int:
    """
    Simple spike detector: flag if any point's z-score exceeds z_thresh.
    """
    z = (x - x.mean()) / x.std(ddof=0)
    return int((np.abs(z) > z_thresh).any())


def generate_signal(spike: bool, length: int = 300, noise_std: float = 0.5):
    """
    White noise; optionally embed one tall, narrow impulse.
    """
    x = np.random.normal(0, noise_std, size=length)
    if spike:
        loc = np.random.randint(length // 10, 9 * length // 10)
        x[loc] += np.random.uniform(6, 12) * noise_std
    return x


def create_dataset(n_samples=200, length=300, output_folder: Path | str = 'spike_dataset.csv'):
    data, labels = [], []
    for _ in range(n_samples // 2):
        data.append(generate_signal(False, length))
        labels.append(0)

        data.append(generate_signal(True, length))
        labels.append(1)

    df = pd.DataFrame(np.vstack(data))
    df['label'] = labels
    return df


if __name__ == '__main__':
    out_dir = Path(__file__).resolve().parent
    train_df = create_dataset(output_folder=out_dir)
    test_df = create_dataset(n_samples=200, length=500, output_folder=out_dir)

    save_datasets(train_df, test_df, out_dir)

    # quick visual check
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_df.iloc[0, :-1]); plt.title('No spike (label 0)')
    plt.subplot(1, 2, 2)
    plt.plot(train_df.iloc[1, :-1]); plt.title('Spike present (label 1)')
    plt.tight_layout(); plt.show()
    plt.savefig(out_dir / 'spike_dataset_example.png')