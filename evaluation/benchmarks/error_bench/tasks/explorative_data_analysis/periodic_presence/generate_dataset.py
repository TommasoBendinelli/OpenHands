#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import periodogram

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets   # noqa: E402


def has_dominant_freq(x: np.ndarray, fs: float = 1.0, ratio: float = 5.0) -> int:
    """
    Use periodogram to decide if a dominant peak exists.

    :return: 0 if dominant frequency present, 1 otherwise
    """
    freqs, pxx = periodogram(x, fs=fs)
    peak = pxx.max()
    mean_pow = pxx.mean()
    return 0 if peak > ratio * mean_pow else 1


def generate_signal(periodic: bool, length: int = 256, fs: float = 1.0):
    """
    Pure noise or noise + sine wave.
    """
    t = np.arange(length) / fs
    noise = np.random.normal(0, 0.5, size=length)
    if periodic:
        freq = np.random.choice([0.05, 0.08, 0.12])
        amp = np.random.uniform(1, 2)
        return amp * np.sin(2 * np.pi * freq * t) + noise
    return noise


def create_dataset(
    n_samples: int = 256,
    length: int = 256,
    fs: float = 1.0,
    output_folder: Path | str = 'periodic_dataset.csv',
):
    data, labels = [], []
    for _ in range(n_samples // 2):
        x = generate_signal(True, length, fs)
        data.append(x)
        labels.append(0)

        x = generate_signal(False, length, fs)
        data.append(x)
        labels.append(1)

    df = pd.DataFrame(np.vstack(data))
    df['label'] = labels
    return df


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    train_df = create_dataset(output_folder=out_dir)
    test_df = create_dataset(n_samples=256, length=512, fs=1.0, output_folder=out_dir)

    save_datasets(train_df, test_df, output_folder=out_dir)

    # sanity-check plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_df.iloc[0, :-1])
    plt.title("Periodic (label 0)")
    plt.subplot(1, 2, 2)
    plt.plot(train_df.iloc[1, :-1])
    plt.title("Aperiodic (label 1)")
    plt.tight_layout()
    plt.show()
    plt.savefig(out_dir / "periodic_dataset_example.png")