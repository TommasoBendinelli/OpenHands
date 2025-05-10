from pathlib import Path

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


def optimal_threshold_interval(
    X, y, *,
    metric="accuracy",
    pos_if_ge=True,
    rtol=1e-12, atol=1e-12
):
    """
    Find the *interval* of thresholds that maximise the chosen metric.

    Parameters
    ----------
    X : array-like, shape (n_samples,)
    y : array-like, 0/1 of the same length
    metric : "accuracy", "youden", "f1" or callable
    pos_if_ge : bool, default True
        If True, predict 1 when X >= τ, else 0.
        If False, predict 1 when X <  τ, else 0.
    rtol, atol : float
        Relative / absolute tolerance used when comparing floating-point
        metric values (via np.isclose).

    Returns
    -------
    low, high, best_score
        *low* and *high* delimit the closed interval of thresholds that achieve
        *best_score*.  When the optimum is unique, `low == high`.
    """
    X = np.asarray(X).ravel()
    y = np.asarray(y).ravel().astype(bool)

    # --- 1. all candidate thresholds ------------------------
    thresholds = np.unique(X)

    # --- 2. predictions for *all* τ at once -----------------
    pred = (X[:, None] >= thresholds) if pos_if_ge else (X[:, None] < thresholds)

    # --- 3. confusion-matrix counts -------------------------
    tp = ( pred &  y[:, None]).sum(axis=0)
    fp = ( pred & ~y[:, None]).sum(axis=0)
    fn = (~pred &  y[:, None]).sum(axis=0)
    tn = (~pred & ~y[:, None]).sum(axis=0)

    # --- 4. metric for every τ ------------------------------
    if metric == "accuracy":
        score = (tp + tn) / len(y)
    elif metric == "youden":
        tpr = tp / (tp + fn + 1e-12)
        tnr = tn / (tn + fp + 1e-12)
        score = tpr + tnr - 1
    elif metric == "f1":
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        score = 2 * prec * rec / (prec + rec + 1e-12)
    elif callable(metric):
        score = np.array([metric(*vals) for vals in zip(tp, fp, fn, tn)])
    else:
        raise ValueError("Unknown metric.")

    # --- 5. locate the interval with best score -------------
    best = score.max()
    mask = np.isclose(score, best, rtol=rtol, atol=atol)   # handle FP ties
    low, high = thresholds[mask][[0, -1]]

    return low, high, best

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
    
    thr_train = optimal_threshold_interval(feature_train, y_training)
    # Compute the optimal threshold also on the test set
    thr_test = optimal_threshold_interval(feature_test, y_test)

    # Check if there is an intersection between the two intervals. If so, sample a value from the intersection. 
    if thr_train[0] <= thr_test[1] and thr_test[0] <= thr_train[1]:
        # There is an intersection
        low = max(thr_train[0], thr_test[0])
        high = min(thr_train[1], thr_test[1])
        thr_value = np.random.uniform(low, high)
    else:
        # Get the value from the train set closest to the test set
        if abs(thr_train[0] - thr_test[0]) < abs(thr_train[1] - thr_test[1]):
            thr_value = thr_train[0]
        else:
            thr_value = thr_train[1]


    # Compute the accuracy on the test set
    y_pred = (feature_test >= thr_value).astype(int)
    y_true = y_test.astype(int)

    # Compute accuracy
    accuracy = (y_true == y_pred).mean()
    return accuracy, feature_train, feature_test

def main():
    current = _update_run_counter()
    accuracy, feature_train, feature_test = do_submission()

    # Compute accuracy
    print(f'Accuracy on test set {current}: {accuracy:.4f}')
    # Save also the y_score to a file
    np.save(f'feature_train_{current}.npy', feature_train)
    np.save(f'feature_test_{current}.npy', feature_test)
    # Save the accuracy to a file (if already exists, append)
    if Path('/mnt/accuracy.txt').exists():
        with open('/mnt/accuracy.txt', 'a') as f:
            f.write(f'Accuracy on test set {current}: {accuracy:.4f} \n')
    else:
        with open('/mnt/accuracy.txt', 'w') as f:
            f.write(f'Accuracy on test set {current}: {accuracy:.4f} \n')

    if accuracy ==1:
        print('Congratulations! You have reached the accuracy threshold of 1.0.')
        print('Please return to the user')


if __name__ == '__main__':
    main()
