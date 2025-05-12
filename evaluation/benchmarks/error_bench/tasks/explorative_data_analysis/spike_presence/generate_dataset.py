#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets   # noqa: E402
import random

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

    # Add a bias term to avoid a simple maximum threshold to catch the spike
    bias = np.random.uniform(-5, 5)
    x += bias
    return x


def create_dataset(n_samples=200, length=300, noise_std=0.5, output_folder: Path | str = 'spike_dataset.csv'):
    data, labels = [], []
    for _ in range(n_samples // 2):
        data.append(generate_signal(False, length, noise_std))
        labels.append(0)

        data.append(generate_signal(True, length, noise_std))
        labels.append(1)

    df = pd.DataFrame(np.vstack(data))
    df['label'] = labels
    return df


if __name__ == '__main__':
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    out_dir = Path(__file__).resolve().parent
    train_df = create_dataset(output_folder=out_dir, length=500, noise_std=0.2)
    test_df = create_dataset(n_samples=200, length=500, output_folder=out_dir, noise_std=0.2)

    save_datasets(train_df, test_df, out_dir)

    # Plot 6 examples of training data  
    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.plot(train_df.iloc[i, :-1])
        plt.title(f"Label: {train_df.iloc[i, -1]}")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")

    # Save the figure
    plt.savefig(out_dir / 'sum_dataset_example.png')

    # Plot 6 examples of test data
    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.plot(test_df.iloc[i, :-1])
        plt.title(f"Label: {test_df.iloc[i, -1]}")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(out_dir / 'test_dataset_example.png')