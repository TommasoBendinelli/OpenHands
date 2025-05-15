#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets   # noqa: E402


def spikes_synchronous(x_flat: np.ndarray, length: int, z: float = 4.0, tol: int = 2) -> int:
    """
    Detect if spikes (>z-score) occur at ~same indices (within tol samples).
    """
    ch1, ch2 = x_flat[:length], x_flat[length:]
    z1 = np.abs((ch1 - ch1.mean()) / ch1.std(ddof=0))
    z2 = np.abs((ch2 - ch2.mean()) / ch2.std(ddof=0))
    idx1, idx2 = np.where(z1 > z)[0], np.where(z2 > z)[0]
    for i in idx1:
        if any(abs(i - j) <= tol for j in idx2):
            return 0  # at least one coincident spike ⇒ label 0
    return 1


def generate_sample(sync: bool,
                    length: int = 400,
                    n_spikes: int = 3,
                    noise: float = 0.4):
    # ── NEW: random baseline per channel ─────────────────────────────
    baseline1 = np.random.uniform(-5, 5)
    baseline2 = np.random.uniform(-5, 5)

    ch1 = np.random.normal(0, noise, size=length) + baseline1
    ch2 = np.random.normal(0, noise, size=length) + baseline2

    spike_locs = np.random.choice(np.arange(20, length - 20),
                                  size=n_spikes, replace=False)

    for loc in spike_locs:
        amp = np.random.uniform(8, 12)
        # ch-1 always gets the spike at *loc*
        ch1[loc] += amp

        if sync:
            # synchronous: ch-2 spike at the same index  (+ small jitter)
            ch2[loc] += amp + np.random.normal(0, 0.3)
        else:
            # independent: ch-2 spike somewhere else
            other_loc = (loc + np.random.randint(5, 30)) % length
            ch2[other_loc] += amp
    return np.concatenate([ch1, ch2])

def create_dataset(n_samples=200, length=400, output_folder= 'syncspike_dataset.csv'):
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

    out_dir = Path(__file__).resolve().parent
    train_df = create_dataset(output_folder=out_dir)
    test_df  = create_dataset(n_samples=200, length=600, output_folder=out_dir)

    save_datasets(train_df, test_df, out_dir)

    plt.figure(figsize=(10, 4))
    for i, title in zip([0, 1], ['Synchronous (label 0)', 'Independent (label 1)']):
        plt.subplot(1, 2, i + 1)
        plt.plot(train_df.iloc[i, :train_df.shape[1]//2], label='ch 1')
        plt.plot(train_df.iloc[i, train_df.shape[1]//2:-1], label='ch 2')
        plt.title(title); plt.legend()
    plt.tight_layout(); plt.show()
    plt.savefig(out_dir / 'syncspike_dataset_example.png')