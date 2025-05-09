import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks_cwt, ricker

sys.path.append(str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
from utils import save_datasets


def number_cwt_peaks(x, n):
    """
    Number of different peaks in x using continuous wavelet transform.

    :param x: 1D numpy array, the time series
    :param n: int, maximum width to consider
    :return: int, number of detected peaks
    """
    widths = np.arange(1, n + 1)
    return len(find_peaks_cwt(x, widths, wavelet=ricker))


def generate_signal(n_peaks, length=100, width=0.02):
    """
    Generate a synthetic signal containing n_peaks Gaussian bumps.

    :param n_peaks: int, number of peaks to embed
    :param length: int, number of time points
    :param width: float, standard deviation of each Gaussian bump
    :return: 1D numpy array of shape (length,)
    """
    t = np.linspace(0, 1, length)
    x = np.zeros_like(t)

    if n_peaks == 1:
        # Make the peak smaller
        width = 0.05
    elif n_peaks > 1:
        # Make the peak larger
        width = 0.005
    centers = np.linspace(0.1, 0.9, n_peaks)
    for c in centers:
        x += width * np.exp(-((t - c) ** 2) / (2 * width**2))

    return x


def create_dataset(
    max_peaks=5,
    samples_per_class=50,
    length=100,
    width=0.02,
    output_folder='time_series_dataset.csv',
    separate_x_and_y=False,
):
    """
    Build a CSV dataset of signals labeled by their true number of peaks.
    label 0 -> 1 peaks
    label 1 -> >1 peaks

    :param max_peaks: int, highest peak-count class
    :param samples_per_class: int, how many samples in each class
    :param length: int, length of each time series
    :param width: float, width of each Gaussian peak
    :param output_path: str, filename for saving the CSV
    :return: pandas.DataFrame of shape (max_peaks * samples_per_class, length + 1)
    """
    data = []
    labels = []

    # Generate signals with 0 and >0 peaks
    # Both classes should be represented equally
    for _ in range(samples_per_class):
        x = generate_signal(1, length, width)
        data.append(x)
        labels.append(0)

        # Create number of peaks from 2 to max_peaks randomly
        k = np.random.randint(2, max_peaks + 1)
        x = generate_signal(k, length, width)
        data.append(x)
        labels.append(1)

    df = pd.DataFrame(np.vstack(data))
    df['label'] = labels

    return df


if __name__ == '__main__':
    # Generate and save the dataset
    output_folder = Path(__file__).resolve().parent

    train_df = create_dataset(output_folder=output_folder)
    test_df = create_dataset(
        max_peaks=10,
        samples_per_class=50,
        length=100,
        width=0.02,
        output_folder=output_folder,
        separate_x_and_y=True,
    )
    print(train_df.head())
    save_datasets(train_df=train_df, test_df=test_df, output_folder=output_folder)

    # Quick visual sanity check
    # Plot signal from both classses

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    class0_signal = train_df[train_df['label'] == 0].iloc[0, :-1]
    plt.plot(class0_signal)
    plt.title('Signal 1 peaks (label 0)')

    plt.subplot(1, 2, 2)
    class1_signal = train_df[train_df['label'] == 1].iloc[0, :-1]
    plt.plot(class1_signal)
    plt.title('Signal >1 peaks (label 1)')

    plt.tight_layout()
    plt.show()
    plt.savefig(output_folder / 'dataset_sanity_check.png')
