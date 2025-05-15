#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import periodogram
import random
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets   # noqa: E402


def same_dominant_freq(x_flat: np.ndarray, length: int, fs: float = 1.0, tol: float = 0.01) -> int:
    """
    Compare dominant frequency of two channels via periodogram.
    """
    ch1, ch2 = x_flat[:length], x_flat[length:]
    f1 = periodogram(ch1, fs=fs)[0][np.argmax(periodogram(ch1, fs=fs)[1])]
    f2 = periodogram(ch2, fs=fs)[0][np.argmax(periodogram(ch2, fs=fs)[1])]
    return 0 if abs(f1 - f2) < tol else 1


def generate_sample(common_freq: bool, length: int = 512, fs: float = 1.0):
    """
    Two sine-wave channels: same or different freq + noise.
    """
    t = np.arange(length) / fs
    freq1 = np.random.choice([0.04, 0.06, 0.08])
    if common_freq:
        freq2 = freq1
    else:
        freq2 = np.random.choice([f for f in [0.04, 0.06, 0.08] if f != freq1])

    ch1 = 2 * np.sin(2 * np.pi * freq1 * t + np.random.uniform(0, 2 * np.pi))
    ch2 = 2 * np.sin(2 * np.pi * freq2 * t + np.random.uniform(0, 2 * np.pi))
    noise = np.random.normal(0, 0.5, size=(2, length))
    return np.concatenate([ch1 + noise[0], ch2 + noise[1]])


def create_dataset(n_samples=256, length=512, output_folder = 'freq_dataset.csv'):
    data, labels = [], []
    for _ in range(n_samples // 2):
        data.append(generate_sample(True, length))
        labels.append(0)
        data.append(generate_sample(False, length))
        labels.append(1)

    cols = [f'a_{t}' for t in range(length)] + [f'b_{t}' for t in range(length)]
    df = pd.DataFrame(data, columns=cols)
    df['label'] = labels
    return df


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    out_dir = Path(__file__).resolve().parent
    train_df = create_dataset(output_folder=out_dir)
    test_df  = create_dataset(n_samples=256, length=768, output_folder=out_dir)

    save_datasets(train_df, test_df, out_dir)

    plt.figure(figsize=(10, 4))
    for i, title in zip([0, 1], ['Common freq (label 0)', 'Different freq (label 1)']):
        plt.subplot(1, 2, i + 1)
        plt.plot(train_df.iloc[i, :train_df.shape[1]//2], label='ch 1')
        plt.plot(train_df.iloc[i, train_df.shape[1]//2:-1], label='ch 2')
        plt.title(title); plt.legend()
    plt.tight_layout(); plt.show()
    plt.savefig(out_dir / 'freq_dataset_example.png')