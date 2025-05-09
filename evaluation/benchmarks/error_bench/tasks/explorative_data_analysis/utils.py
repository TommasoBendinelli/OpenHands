from pathlib import Path

import pandas as pd


def save_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame, output_folder: Path):
    """
    Save training and test DataFrames to CSV files in `output_folder`:
      - train.csv, train_labels.csv
      - test.csv, test_gt.csv
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    # Training set
    train_labels = train_df['label'].astype(int)
    train_df.drop(columns=['label']).to_csv(output_folder / 'train.csv', index=False)
    train_labels.to_csv(output_folder / 'train_labels.csv', index=False)

    # Test set
    test_labels = test_df['label'].astype(int)
    test_df.drop(columns=['label']).to_csv(output_folder / 'test.csv', index=False)
    test_labels.to_csv(output_folder / 'test_gt.csv', index=False)
