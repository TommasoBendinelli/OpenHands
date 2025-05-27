import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

# df is your DataFrame, each column a separate time series
def count_zero_crossings(col: pd.Series) -> int:
    # A zero-crossing occurs whenever consecutive samples have opposite signs
    # (we treat exact zeros as belonging to neither sign)
    signs = np.sign(col.values)           # −1, 0, or +1
    mask  = signs != 0                    # drop the exact-zero samples
    signs = signs[mask]                   # keep only non-zeros
    # np.diff detects a sign change; abs(…) == 2 flags −1→+1 or +1→−1
    return (np.abs(np.diff(signs)) == 2).sum()


def main():
    output_folder = Path(__file__).resolve().parent
    df = pd.read_csv(output_folder / Path('train.csv'))
    df_labels = pd.read_csv(output_folder / Path('train_labels.csv'))
    df_test = pd.read_csv(output_folder / Path('test.csv'))
    df_test_labels = pd.read_csv(output_folder / Path('test_gt.csv'))
   
    x = df.apply(count_zero_crossings, axis=1)
    x_test = df_test.apply(count_zero_crossings, axis=1)
    y = df_labels['label'].values
    y_test = df_test_labels['label'].values
    # Train a Decision Tree Classifier
    clf = DecisionTreeClassifier(max_depth=1, random_state=42)
    clf.fit(x.values.reshape(-1, 1), y)
    # Predict on the test set
    y_pred = clf.predict(x_test.values.reshape(-1, 1))
    # Compute accuracy
    accuracy = (y_test == y_pred).mean()
    print(f"Test accuracy: {accuracy:.2f}")


  


if __name__ == '__main__':
    main()