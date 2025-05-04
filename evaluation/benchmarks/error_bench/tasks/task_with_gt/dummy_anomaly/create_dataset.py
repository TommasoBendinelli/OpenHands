import numpy as np
import pandas as pd


def main():
    # Create a dummy dataset

    arr = np.random.rand(100000, 10)  # 100 samples, 10 features
    columns = [f'feature_{i}' for i in range(arr.shape[1])]
    df = pd.DataFrame(arr, columns=columns)

    # Make the entry at position 1004 an error
    df.iloc[1004, 0] = 1000

    # Save the dataset to a CSV file
    df.to_csv('dataset.csv', index=False)


if __name__ == '__main__':
    main()
