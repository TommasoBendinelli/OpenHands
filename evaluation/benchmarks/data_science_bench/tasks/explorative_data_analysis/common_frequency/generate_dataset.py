#!/usr/bin/env python
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import periodogram

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets  # noqa: E402


def same_dominant_freq(
    x_flat: np.ndarray, length: int, fs: float = 1.0, tol: float = 0.01
) -> int:
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


def create_dataset(n_samples=256, length=512, output_folder='freq_dataset.csv'):
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
    test_df = create_dataset(n_samples=256, length=768, output_folder=out_dir)

    save_datasets(train_df, test_df, out_dir)

    length = train_df.shape[1] // 2
    plt.figure(figsize=(10, 4))
    for i, title in zip(
        [0, 1], ['Same Frequency (class 0)', 'Different Frequency (class 1)']
    ):
        # Channel A on top row
        ax1 = plt.subplot(2, 2, i + 1)
        ax1.plot(train_df.iloc[i, :length], label='channel A')
        ax1.set_title(f'{title} – Channel A')
        ax1.legend()
        ax1.set_xticks(range(0, length, 100))

        # Channel B on bottom row
        ax2 = plt.subplot(2, 2, i + 3)
        ax2.plot(train_df.iloc[i, length:-1], label='channel B')
        ax2.set_title(f'{title} – Channel B')
        ax2.legend()
        ax2.set_xticks(range(0, length, 100))

    plt.tight_layout()

    # Also save here if FIGS_DIR is provided
    figs_dir = Path(
        os.environ.get('FIGS_DIR', '67ff59b20231d9f95909f426/figs/datasets')
    )
    plt.savefig(figs_dir / 'common_frequency.png', dpi=150)
    plt.savefig(figs_dir / 'common_frequency.pdf', dpi=150)

    print('Files written: train.csv, test.csv + PNG sanity plots')
