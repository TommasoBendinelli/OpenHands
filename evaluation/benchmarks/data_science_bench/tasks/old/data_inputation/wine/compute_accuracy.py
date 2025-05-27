from pathlib import Path

import pandas as pd


def main():
    # Find the csv in the same folder as this script
    csv_gt = list(Path(__file__).parent.glob('*.csv'))[0]
    # Read the csv file
    df = pd.read_csv(csv_gt)

    # Read also the cleaned.csv file in /workspace/cleaned.csv
    cleaned_csv = '/workspace/cleaned.csv'
    df_cleaned = pd.read_csv(cleaned_csv)

    #
    # Measure Mean Absolute Error (MAE)
    mean_absolute_error = ((df_cleaned - df) ** 2).mean()
    # Normalize the MAE by dividing by the mean of the ground truth
    mean_absolute_error = mean_absolute_error / df.std()
    # Take the final mean across all columns
    mean_absolute_error = mean_absolute_error.mean()
    print(f'Median Absolute Error: {mean_absolute_error}')


if __name__ == '__main__':
    main()
