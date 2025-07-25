#!/usr/bin/env python
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets  # noqa: E402


# --------------------------------------------------------------------------- #
#                           DATA-SET GENERATION                               #
# --------------------------------------------------------------------------- #
def generate_sample(corr: bool, length: int = 300, noise: float = 0.4):
    """
    Create a two-channel time series.

    * If ``corr`` is True, both channels share a common smooth random walk
      (the “base”) plus independent noise.
    * If ``corr`` is False, the channels are independent white-noise walks.

    NEW RANDOMISATIONS
    ------------------
    • Per-channel *scale*  factor  N(0, 1.5)  — may be negative (sign flips)
    • Per-channel *offset* factor N(0, 3.0)

    These transformations *destroy* any signal in the point-wise mean/abs-diff
    but leave the Pearson correlation invariant up to sign, so a classifier
    that relies on |corr| still works whereas the spurious distance-based
    solutions collapse to random chance.
    """
    base = np.cumsum(np.random.normal(0, 0.2, size=length))  # smooth walk

    if corr:
        ch1 = base + np.random.normal(0, noise, size=length)
        ch2 = base + np.random.normal(0, noise, size=length)
    else:
        ch1 = np.random.normal(0, 1.0, size=length)
        ch2 = np.random.normal(0, 1.0, size=length)

    # Independent random scales (some negative → sign flips)
    scale1 = np.random.normal(0, 1.5)
    scale2 = np.random.normal(0, 1.5)

    # Independent random offsets
    offset1 = np.random.normal(0, 3.0)
    offset2 = np.random.normal(0, 3.0)

    ch1 = ch1 * scale1 + offset1
    ch2 = ch2 * scale2 + offset2

    return np.concatenate([ch1, ch2])


def create_dataset(
    n_samples: int = 200, length: int = 300, output_folder='corr_dataset.csv'
):
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


# --------------------------------------------------------------------------- #
#                               SCRIPT ENTRY                                  #
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    out_dir = Path(__file__).resolve().parent
    train_df = create_dataset(output_folder=out_dir)
    test_df = create_dataset(n_samples=200, length=500, output_folder=out_dir)
    save_datasets(train_df, test_df, out_dir)

    # quick sanity-check plot
    plt.figure(figsize=(10, 4))
    length = train_df.shape[1] // 2
    for i, title in zip([0, 1], ['Correlated (class 0)', 'Uncorrelated (class 1)']):
        # Channel A on the top row
        ax1 = plt.subplot(2, 2, i + 1)
        ax1.plot(train_df.iloc[i, :length], label='channel A')
        ax1.set_title(f'{title} – Channel A')
        ax1.legend()
        ax1.set_xticks(range(0, length, 50))

        # Channel B right below it
        ax2 = plt.subplot(2, 2, i + 3)
        ax2.plot(train_df.iloc[i, length:-1], label='channel B')
        ax2.set_title(f'{title} – Channel B')
        ax2.legend()
        ax2.set_xticks(range(0, length, 50))

    plt.tight_layout()
    plt.show()
    plt.savefig(out_dir / 'corr_dataset_example.png')

    # Also save here if a folder is provided via FIGS_DIR
    figs_dir = Path(
        os.environ.get('FIGS_DIR', '67ff59b20231d9f95909f426/figs/datasets')
    )
    plt.savefig(figs_dir / 'channel_corr_hard.png')
    plt.savefig(figs_dir / 'channel_corr_hard.pdf')
