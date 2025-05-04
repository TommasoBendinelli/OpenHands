from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks_cwt, ricker


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
    centers = np.linspace(0.1, 0.9, n_peaks)
    for c in centers:
        x += np.exp(-((t - c) ** 2) / (2 * width**2))
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

    :param max_peaks: int, highest peak-count class
    :param samples_per_class: int, how many samples in each class
    :param length: int, length of each time series
    :param width: float, width of each Gaussian peak
    :param output_path: str, filename for saving the CSV
    :return: pandas.DataFrame of shape (max_peaks * samples_per_class, length + 1)
    """
    data = []
    labels = []
    for k in range(1, max_peaks + 1):
        for _ in range(samples_per_class):
            x = generate_signal(k, length, width)
            data.append(x)
            labels.append(k)
    df = pd.DataFrame(np.vstack(data))
    df['label'] = labels
    if separate_x_and_y:
        labels = df.pop('label')
        X = df
        # Save test_gt.csv, test.csv
        X.to_csv(output_folder / 'test.csv', index=False)
        labels.to_csv(output_folder / 'test_gt.csv', index=False)

    else:
        df.to_csv(output_folder / 'train.csv', index=False)
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

    import matplotlib.pyplot as plt

    # Iterate from len(df) to 0
    for i in range(len(train_df) - 1, 1, -1):
        plt.plot(train_df.iloc[i, :-1])
        plt.title(f'Number of Peaks: {train_df.iloc[i, -1]}')

    # # Save the plot
    plt.savefig('sample_signals.png')
