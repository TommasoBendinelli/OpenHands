# Add to the system path evaluation/benchmarks/error_bench/tasks/explorative_data_analysis
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets


def generate_dataset():
    # Open train.csv
    df = pd.read_csv(
        'evaluation/benchmarks/error_bench/tasks/explorative_data_analysis/house_price/train.csv'
    )

    # Create a split (80% train, 20% test)
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    # Save the datasets

    return train_df, test_df


# ---------------------------------------------------------------------
#  Example CLI orchestration (mirrors your original script layout)

if __name__ == '__main__':
    output_folder = Path(__file__).resolve().parent

    train_df, test_df = generate_dataset()
    train_df['label'] = train_df['SalePrice'].astype(int)
    test_df['label'] = test_df['SalePrice'].astype(int)
    # Drop the SalePrice column
    train_df.drop(columns=['SalePrice'], inplace=True)
    test_df.drop(columns=['SalePrice'], inplace=True)
    # Convert the label
    save_datasets(train_df, test_df, output_folder)
