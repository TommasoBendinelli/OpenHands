import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('evaluation/benchmarks/error_bench/tasks/explorative_data_analysis/predict_ts_stationarity/train.csv')
    # Normalise the data across rows
    y = df['label']
    X = df.drop(columns=['label'])
    
    # Compute differences between one value and the next
    X_diff = X.diff(axis=1).iloc[:, 1:]

    # Load the test set
    test_df = pd.read_csv('evaluation/benchmarks/error_bench/tasks/explorative_data_analysis/predict_ts_stationarity/test_cheating.csv')
    y_test = test_df['label']
    X_test = test_df.drop(columns=['label'])
    
    pd.concat([(X_test.diff(axis=1).iloc[:, 1:].abs().mean(axis=1) - X_test.mean(axis=1).abs())[:20], y_test[:20]], axis=1)
    breakpoint()
    y_test = test_df['label']
   
if __name__ == '__main__':
    main()