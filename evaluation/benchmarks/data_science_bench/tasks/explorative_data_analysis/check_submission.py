from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
RUN_COUNTER_FILE = Path('run_counter.txt')
RUN_COUNTER_LIMIT = 20  # safeguard against accidental infinite loops

def _update_run_counter(
    path: Path = RUN_COUNTER_FILE, limit: int = RUN_COUNTER_LIMIT
) -> int:
    """Increment the run counter and enforce an upper bound.

    Raises
    ------
    RuntimeError
        If *limit* executions has been reached.
    """
    current = int(path.read_text().strip() or '0') if path.exists() else 0

    if current >= limit:
        raise RuntimeError(
            f'Script has been run {current} times; limit of {limit} reached. '
            'Please submit your best solution and exit.'
        )
    current += 1
    path.write_text(str(current))
    return current

def parse_submission_file(path: Path, df: pd.DataFrame) -> pd.Series:
    y_score = pd.read_csv(path, header=None)

    # If the first cell is a non-digit string, treat the first row as a header
    first_val = y_score.iloc[0, 0]
    if isinstance(first_val, str):
        y_score.columns = y_score.iloc[0]
        y_score = y_score.iloc[1:]

    # Drop an extra row if present (e.g., a duplicated header)
    if len(y_score) > len(df):
        y_score = y_score.iloc[1:]

    # Flatten to a 1-D vector
    if y_score.shape[1] > 1:
        y_score = y_score['label'].to_numpy().flatten()
    else:
        y_score = y_score.to_numpy().flatten()

    return y_score



def do_submission():
    csv_gt = list(Path(__file__).parent.glob('*.csv'))[0]
    df = pd.read_csv(csv_gt)
    # Check if the files exist
    if not Path('/workspace/train_engineered_feature.csv').exists():
        raise FileNotFoundError(
            '/workspace/train_engineered_feature.csv not found. Please check the file path.'
        )
    if not Path('/workspace/test_engineered_feature.csv').exists():
        raise FileNotFoundError(
            '/workspace/test_engineered_feature.csv not found. Please check the file path.'
        )
    # Get the y values from the training data
    y_training_df = pd.read_csv("/workspace/train_labels.csv")

    feature_train = parse_submission_file('/workspace/train_engineered_feature.csv', y_training_df)
    feature_test = parse_submission_file('/workspace/test_engineered_feature.csv', df)
    # Get the y values from the training data
    y_training = y_training_df.values.flatten()
    y_test = df.values.flatten()
    # y_training = y_training.iloc[:, 1].values.flatten()
    assert len(y_training) == len(feature_train), "Length of y_training and y_score_train do not match"
    assert len(df) == len(feature_test), "Length of y_training and y_score_test do not match"

    # Assert that the y_score_train and y_score_test are not all digits 
    assert not all([x.isdigit() for x in feature_train if isinstance(x,str)]), "y_score_train contains all digits, it should contain a single engineered and intepretable feature that will be then thresholded"
    assert not all([x.isdigit() for x in feature_test if isinstance(x,str)]), "y_score_test contains all digits, it should contain a single engineered and intepretable feature that will be then thresholded"

    try:
        feature_train = feature_train.astype(float)
        feature_test = feature_test.astype(float)
    except ValueError:
        raise ValueError(
            'y_score_train or y_score_test contains non-floating values. Please ensure that the predictions are floating.'
        )
    # features = np.concatenate((feature_train, feature_test), axis=0)
    # y = np.concatenate((y_training, y_test), axis=0)
    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(feature_train.reshape(-1, 1), y_training)

    pred_train = clf.predict(feature_train.reshape(-1, 1))
    pred_test = clf.predict(feature_test.reshape(-1, 1))

    # Compute accuracy
    accuracy_test = (y_test == pred_test).mean()
    accuracy_train = (y_training == pred_train).mean()
    return accuracy_test, accuracy_train, feature_train, feature_test

def main():
    accuracy_test, accuracy_train, feature_train, feature_test = do_submission()
    print(f"Submission is valid.")


if __name__ == '__main__':
    main()
