from pathlib import Path

import pandas as pd


def main():
    data_dir = 'evaluation/benchmarks/error_bench/tasks/anomaly_detections/vifd'
    df = pd.read_csv(Path(data_dir) / 'carclaims.csv')
    df['FraudFound'] = df['FraudFound'].map({'Yes': 1, 'No': 0})
    # y = df['FraudFound'].map({"Yes":1, "No":0})
    # y = y.to_numpy()

    def split_on_uppercase(s):
        return ''.join(' ' + i if i.isupper() else i for i in s).lower().strip()

    columns = [split_on_uppercase(c) for c in df.columns]

    df.columns = columns
    train_df = df.sample(frac=0.5, random_state=42)
    test_df = df.drop(train_df.index)
    test_df.to_csv(data_dir + '/test_gt.csv', index=False)

    # Remove the class column from the test set
    test_df.drop(columns=['fraud found'], inplace=True)
    # Save the training and test set to csv files
    train_df.to_csv(data_dir + '/train.csv', index=False)
    test_df.to_csv(data_dir + '/test.csv', index=False)


if __name__ == '__main__':
    main()
