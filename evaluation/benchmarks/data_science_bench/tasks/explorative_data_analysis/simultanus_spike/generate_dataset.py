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


def spikes_synchronous(
    x_flat: np.ndarray, length: int, z: float = 4.0, tol: int = 2
) -> int:
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


def generate_sample(
    sync: bool, length: int = 400, n_spikes: int = 3, noise: float = 0.4
):
    # ── NEW: random baseline per channel ─────────────────────────────
    baseline1 = np.random.uniform(-5, 5)
    baseline2 = np.random.uniform(-5, 5)

    ch1 = np.random.normal(0, noise, size=length) + baseline1
    ch2 = np.random.normal(0, noise, size=length) + baseline2

    spike_locs = np.random.choice(
        np.arange(20, length - 20), size=n_spikes, replace=False
    )

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


def create_dataset(n_samples=200, length=400, output_folder='syncspike_dataset.csv'):
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


def _boxed_two_panel_plot(df: pd.DataFrame, path_png: Path, path_pdf: Path) -> None:
    """Save one synchronous and one independent example in boxed style."""
    fig = plt.figure(figsize=(10, 4), dpi=150)
    ax0, ax1 = plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)

    # synchronous → label 0
    ex_sync = df[df['label'] == 0].iloc[0]
    length = ex_sync.shape[0] // 2
    ax0.plot(ex_sync.iloc[:length], label='ch 1')
    ax0.plot(ex_sync.iloc[length:-1], label='ch 2')
    ax0.set_title('Synchronous spikes\n(class 0)', pad=6)
    ax0.legend(loc='upper right', fontsize=8)

    # independent → label 1
    ex_ind = df[df['label'] == 1].iloc[0]
    ax1.plot(ex_ind.iloc[:length], label='ch 1')
    ax1.plot(ex_ind.iloc[length:-1], label='ch 2')
    ax1.set_title('Independent spikes\n(class 1)', pad=6)
    ax1.legend(loc='upper right', fontsize=8)

    for ax in (ax0, ax1):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_edgecolor('black')

    plt.tight_layout()
    fig.savefig(path_png)
    fig.savefig(path_pdf)
    plt.close(fig)


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)

    out_dir = Path(__file__).resolve().parent

    out_dir = Path(__file__).resolve().parent
    train_df = create_dataset(output_folder=out_dir)
    test_df = create_dataset(n_samples=200, length=600, output_folder=out_dir)

    save_datasets(train_df, test_df, out_dir)

    figs_dir = Path(
        os.environ.get('FIGS_DIR', '67ff59b20231d9f95909f426/figs/datasets')
    )
    # boxed sanity plots
    plt.figure(figsize=(10, 4))
    length = train_df.shape[1] // 2  # samples per channel
    for col_idx, (lbl, title) in enumerate(
        [(0, 'Synchronous spikes (class 0)'), (1, 'Independent spikes (class 1)')]
    ):
        # fetch the first example of this label
        row = train_df[train_df['label'] == lbl].iloc[0]

        # Channel A – top row
        axA = plt.subplot(2, 2, col_idx + 1)
        axA.plot(row.iloc[:length], label='channel A')
        axA.set_title(f'{title} – Channel A')
        axA.legend()
        axA.set_xticks(range(0, length, 100))

        # Channel B – bottom row
        axB = plt.subplot(2, 2, col_idx + 3)
        axB.plot(row.iloc[length:-1], label='channel B')
        axB.set_title(f'{title} – Channel B')
        axB.legend()
        axB.set_xticks(range(0, length, 100))

    plt.tight_layout()
    plt.show()

    plt.savefig(out_dir / 'syncspike_dataset_example.png')
    plt.savefig(figs_dir / 'syncspike_dataset_example.png')
    plt.savefig(figs_dir / 'syncspike_dataset_example.pdf')
