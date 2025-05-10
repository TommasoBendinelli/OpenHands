#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets   # noqa: E402


def phase_aligned(x_flat: np.ndarray, length: int, max_lag: int = 10) -> int:
    """
    Cross-correlation: if the peak is at |lag| ≤ max_lag ⇒ label 0, else 1.
    """
    ch1, ch2 = x_flat[:length], x_flat[length:]
    xc = correlate(ch1 - ch1.mean(), ch2 - ch2.mean(), mode="full")
    lags = np.arange(-length + 1, length)
    peak_lag = lags[np.argmax(xc)]
    return 0 if abs(peak_lag) <= max_lag else 1


def generate_sample(aligned: bool, length: int = 512, fs: float = 1.0):
    """
    Two noisy sinusoids: either small or large phase offset.
    """
    t = np.arange(length) / fs
    freq = np.random.choice([0.04, 0.06, 0.08])
    phase1 = np.random.uniform(0, 2 * np.pi)
    if aligned:
        phase2 = phase1 + np.random.uniform(-0.1, 0.1)          # ≤ ~6 deg
    else:
        phase2 = phase1 + np.random.uniform(np.pi / 2, np.pi)   # ≥ 90 deg
    ch1 = 2 * np.sin(2 * np.pi * freq * t + phase1)
    ch2 = 2 * np.sin(2 * np.pi * freq * t + phase2)
    noise = np.random.normal(0, 0.4, size=(2, length))
    return np.concatenate([ch1 + noise[0], ch2 + noise[1]])


def create_dataset(n_samples=256, length=512, output_folder: Path | str = 'phase_dataset.csv'):
    data, labels = [], []
    for _ in range(n_samples // 2):
        data.append(generate_sample(True, length))
        labels.append(0)
        data.append(generate_sample(False, length))
        labels.append(1)

    cols = [f'ch1_{t}' for t in range(length)] + [f'ch2_{t}' for t in range(length)]
    df = pd.DataFrame(data, columns=cols)
    df['label'] = labels
    return df


if __name__ == '__main__':
    out_dir = Path(__file__).resolve().parent
    train_df = create_dataset(output_folder=out_dir)
    test_df  = create_dataset(n_samples=256, length=768, output_folder=out_dir)

    save_datasets(train_df, test_df, out_dir)

    # sanity check
    plt.figure(figsize=(10, 4))
    for i, ttl in zip([0, 1], ['Aligned (label 0)', 'Shifted (label 1)']):
        plt.subplot(1, 2, i + 1)
        plt.plot(train_df.iloc[i, :train_df.shape[1]//2], label='ch 1')
        plt.plot(train_df.iloc[i, train_df.shape[1]//2:-1], label='ch 2')
        plt.title(ttl); plt.legend()
    plt.tight_layout(); plt.show()
    plt.savefig(out_dir / 'phase_dataset_example.png')