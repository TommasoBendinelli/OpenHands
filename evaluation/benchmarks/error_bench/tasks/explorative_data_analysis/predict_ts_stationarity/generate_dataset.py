from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adfuller_stationarity_test(series, significance=0.05):
    """
    Adjusted Augmented Dickey-Fuller test for stationarity of time series.
    """
    result = adfuller(series.values)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    if result[1] <= significance:
        print('The time series is stationary.')
        return True
    else:
        print('The time series is non-stationary.')
        return False


def generate_synthetic_ts_signal(
    num_time_series_values, stationarity=True, mean=0, std=1
):
    """
    Generate random stationary and non-stationary time series data.
    """
    # Generate stationary time series data
    if stationarity:
        ts_data = np.random.normal(loc=mean, scale=std, size=num_time_series_values)
    else:
        ts_data = np.cumsum(
            np.random.normal(loc=mean, scale=std, size=num_time_series_values)
        )

    return ts_data


def generate_dataset(num_samples=300, num_time_series_values=1000, mean=0, std=1):
    """
    Generate a dataset of time series data with labels (0,1) indicating stationarity. One sample is one time series.
    num_samples: int, number of time series samples to generate
    num_time_series_values: int, number of time series values in each sample. needs to be sufficiently large
    return: pandas.DataFrame, dataset with time series data and labels
    """
    data = []
    labels = []
    for i in range(num_samples):
        # Generate random time series data
        stationarity = np.random.choice([True, False])
        ts_data = generate_synthetic_ts_signal(
            num_time_series_values, stationarity, mean=mean, std=std
        )
        data.append(ts_data)
        labels.append(1 if stationarity else 0)
    # Create DataFrame
    df = pd.DataFrame(np.vstack(data))
    df['label'] = labels
    return df


if __name__ == '__main__':
    output_folder = Path(__file__).resolve().parent

    # Generate synthetic time series data
    train_df_1 = generate_dataset(mean=0, std=1)
    train_df_2 = generate_dataset(mean=2, std=3)
    test_df = generate_dataset(mean=3, std=15)
    train_df = pd.concat([train_df_1, train_df_2], ignore_index=True)

    print(train_df.head())
    # Save the dataset to a CSV file
    train_df.to_csv(output_folder / 'train.csv', index=False)
    y_test = test_df['label']
    y_test.to_csv(output_folder / 'test_gt.csv', index=False)

    X_test = test_df.drop(columns=['label'])

    X_test.to_csv(output_folder / 'test.csv', index=False)

    # Plot one stationary and one non-stationary time series
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    stationary_ts = train_df[train_df['label'] == 1].iloc[0, :-1]
    plt.plot(stationary_ts)
    plt.title('Stationary Time Series')
    plt.subplot(1, 2, 2)
    non_stationary_ts = train_df[train_df['label'] == 0].iloc[0, :-1]
    plt.plot(train_df.iloc[1, :-1])
    plt.title('Non-Stationary Time Series')
    plt.show()
    plt.savefig(output_folder / 'stationary_non_stationary_ts.png')

    # Do the same plot for the test set
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    stationary_ts = test_df[test_df['label'] == 1].iloc[0, :-1]
    plt.plot(stationary_ts)
    plt.title('Stationary Time Series')
    plt.subplot(1, 2, 2)
    non_stationary_ts = test_df[test_df['label'] == 0].iloc[0, :-1]
    plt.plot(test_df.iloc[1, :-1])
    plt.title('Non-Stationary Time Series')
    plt.show()
    plt.savefig(output_folder / 'stationary_non_stationary_ts_test.png')

    # Test stationarity of the generated time series
    print('Testing stationarity of the first time series:')
    adfuller_stationarity_test(train_df.iloc[0, :-1])
