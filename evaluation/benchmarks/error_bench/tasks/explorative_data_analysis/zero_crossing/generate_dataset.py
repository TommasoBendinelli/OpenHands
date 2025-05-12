# Add to the system path evaluation/benchmarks/error_bench/tasks/explorative_data_analysis
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
import random
from typing import Tuple
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets
import click 

# ---------------------------------------------------------------------
#  Core helpers
# ---------------------------------------------------------------------
def band_power(signal, fs, band):
    """
    Estimate average power in `band` (Hz) using Welch’s PSD.
    """
    f, Pxx = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    idx = np.logical_and(f >= band[0], f <= band[1])
    return np.trapz(Pxx[idx], f[idx])


def generate_synthetic_freq_band_signal(
    n_points: int,
    fs: int = 100,
    target_band_hz: Tuple[int, int] = (0, 4),
    noise_std: float = 0.5,
    max_noise_fraction: float = 0.99,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Create a 1-D sine wave whose carrier lies uniformly in `target_band_hz`
    and add *additive* Gaussian noise without ever flipping the sign
    of any individual sample.

    Parameters
    ----------
    n_points : int
        Number of samples.
    fs : int, optional
        Sampling frequency (Hz).  Default is 100 Hz.
    target_band_hz : (low, high), optional
        Frequency band (Hz) from which to draw the carrier uniformly.
    noise_std : float, optional
        Standard deviation of the Gaussian noise *before* clipping.
    max_noise_fraction : float, optional
        Maximum fraction of |sample| that the noise is allowed to reach
        in the **opposite** direction.  
        Example: if a sample is 0.43 and `max_noise_fraction` = 0.99,
        the lowest the noisy sample can go is
        0.43 − 0.99·0.43 ≈ +0.0043 (still positive).
    rng : np.random.Generator | None
        Optional NumPy random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_points,) containing the noisy signal.
    """
    rng = np.random.default_rng() if rng is None else rng

    # --- deterministic part -------------------------------------------------
    t = np.arange(n_points) / fs
    freq = rng.uniform(*target_band_hz)        # carrier frequency
    # Random initial phase
    phase = rng.uniform(0, 2 * np.pi)
    clean = np.sin(2 * np.pi * freq * t + phase)

    # --- additive noise -----------------------------------------------------
    noise = rng.normal(scale=noise_std, size=n_points)

    # Amount we’re allowed to move *towards* zero without crossing it
    guard_band = max_noise_fraction * np.abs(clean)

    # For positive samples, limit negative noise; for negatives, limit positive noise
    neg_limit = -guard_band        # lower bound for positive samples
    pos_limit =  guard_band        # upper bound for negative samples

    # Vectorised clipping
    noise = np.where(
        clean > 0,                           # positive samples
        np.maximum(noise, neg_limit),        # clip at neg_limit from below
        np.where(
            clean < 0,                       # negative samples
            np.minimum(noise, pos_limit),    # clip at pos_limit from above
            noise                            # samples exactly zero — leave unchanged
        )
    )


    return clean + noise


def generate_dataset(
    num_samples: int = 300,
    n_points: int = 1000,
    fs: int = 100,
    target_band_hz: tuple[int, int] = (5, 10),
    noise_std: float = 0.5,
):
    """
    Return a DataFrame where each row is a signal sample and
    `label` indicates band presence (1 = band present, 0 = absent).
    """
    data, labels = [], []
    for _ in range(num_samples):
        present = bool(np.random.choice([0, 1]))
        if present:
            target_band_hz = (3, 8)
        else:
            target_band_hz = (10, 20)
        sig = generate_synthetic_freq_band_signal(
            n_points,
            fs=fs,
            target_band_hz=target_band_hz,
            noise_std=noise_std,
        )
        data.append(sig)
        labels.append(int(present))

    df = pd.DataFrame(np.vstack(data))
    df['label'] = labels
    return df


# ---------------------------------------------------------------------
#  Example CLI orchestration (mirrors your original script layout)
# ---------------------------------------------------------------------

@click.command()
@click.option('--noise', default=300, help='Number of samples to generate.')
@click.option('--nans', default=1000, help='Number of points per sample.')
def main(noise, nans):
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    output_folder = Path(__file__).resolve().parent

    # Pick two training distributions to increase variety
    train_df = generate_dataset(noise_std=noise)

    # Corrupt with 

    test_df = generate_dataset(noise_std=0.0)

    save_datasets(train_df=train_df, test_df=test_df, output_folder=output_folder)
    # Save CSVs
    train_df_labels = train_df['label'].astype(int)
    # train_df.to_csv(output_folder / 'train_labels.csv', index=False)
    # train_df = train_df.drop(columns=['label'])
    # train_df.to_csv(output_folder / 'train.csv', index=False)
    # y_test = test_df['label']
    # y_test.to_csv(output_folder / 'test_gt.csv', index=False)
    # X_test = test_df.drop(columns=['label'])
    # X_test.to_csv(output_folder / 'test.csv', index=False)
    # Nice-to-have plots (optional)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sig_pos = train_df[train_df_labels == 1].iloc[0, :-1]
    plt.plot(sig_pos)
    plt.title('Signal WITH target band (label 1)')

    plt.subplot(1, 2, 2)
    sig_neg = train_df[train_df_labels == 0].iloc[0, :-1]
    plt.plot(sig_neg)
    plt.title('Signal WITHOUT target band (label 0)')

    plt.tight_layout()
    plt.show()
    plt.savefig(output_folder / 'freq_band_examples.png')


if __name__ == '__main__':
    main()