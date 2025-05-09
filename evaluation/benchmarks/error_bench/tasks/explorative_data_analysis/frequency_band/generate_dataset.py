# Add to the system path evaluation/benchmarks/error_bench/tasks/explorative_data_analysis
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets


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
    target_band_hz: tuple[int, int] = (0, 4),
    noise_std: float = 0.5,
):
    """
    Create a synthetic 1-D time-series signal either
    • containing (band_present=True) or
    • not containing (band_present=False)
    power in `target_band_hz`.
    """
    t = np.arange(n_points) / fs

    # Choose carrier frequencies
    # Pick the frequency in the middle of the band
    freq = np.random.uniform(target_band_hz[0], target_band_hz[1])
    # else:
    #     # Pick a random freq outside the band (0–Nyquist)
    #     nyquist = fs / 2
    #     valid = False
    #     while not valid:
    #         freq = np.random.uniform(0.5, nyquist - 0.5)
    #         valid = freq < target_band_hz[0] or freq > target_band_hz[1]

    # Build the sine plus noise
    signal = np.sin(2 * np.pi * freq * t)
    signal += np.random.normal(scale=noise_std, size=n_points)

    return signal


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
            target_band_hz = (0, 2)
        else:
            target_band_hz = (20, 50)
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
if __name__ == '__main__':
    output_folder = Path(__file__).resolve().parent

    # Pick two training distributions to increase variety
    train_df_1 = generate_dataset(noise_std=0.2)
    train_df_2 = generate_dataset(noise_std=0.1)

    test_df = generate_dataset(noise_std=0.5)
    train_df = pd.concat([train_df_1, train_df_2], ignore_index=True)
    print(train_df.head())

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
