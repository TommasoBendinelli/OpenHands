from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics

RUN_COUNTER_FILE = Path('run_counter.txt')
RUN_COUNTER_LIMIT = 10**12  # safeguard against accidental infinite loops


def _update_run_counter(
    path: Path = RUN_COUNTER_FILE, limit: int = RUN_COUNTER_LIMIT
) -> None:
    """Increment the run counter and enforce an upper bound.

    Raises
    ------
    RuntimeError
        If *limit* executions has been reached.
    """
    current = int(path.read_text().strip() or '0') if path.exists() else 0
    current += 1
    if current >= limit:
        raise RuntimeError(
            f'Script has been run {current} times; limit of {limit} reached. '
            'Please submit your best solution and exit.'
        )
    path.write_text(str(current))


def main():
    """
    Calculates evaluation metrics for tabular anomaly detection.
    Adapted from  https://github.com/xuhongzuo/DeepOD/blob/main/deepod/metrics/_anomaly_detection.py
    Args:

        y_true (np.array, required):
            Data label, 0 indicates normal timestamp, and 1 is anomaly.

        y_score (np.array, required):
            Predicted anomaly scores, higher score indicates higher likelihoods to be anomaly.

    Returns:
        tuple: A tuple containing:

        - auc_roc (float):
            The score of area under the ROC curve.

        - auc_pr (float):
            The score of area under the precision-recall curve.

        - f1 (float):
            The score of F1-score.

        - precision (float):
            The score of precision.

        - recall (float):
            The score of recall.

    """

    _update_run_counter()
    # Find the csv in the same folder as this script
    csv_gt = list(Path(__file__).parent.glob('*.csv'))[0]
    # Read the csv file
    df = pd.read_csv(csv_gt)

    # Read also the cleaned.csv file in /workspace/cleaned.csv
    y_score = pd.read_csv('/workspace/solution.csv', header=None).values.flatten()
    if len(y_score) > len(df):
        y_score = y_score[1:]  # remove the first element, which is the header
    y_true = df.iloc[:, -1].values

    # F1@k, using real percentage to calculate F1-score
    n_test = len(y_true)
    new_index = np.random.permutation(
        n_test
    )  # shuffle y to prevent bias of ordering (argpartition may discard entries with same value)
    y_true = y_true[new_index]
    y_score = y_score[new_index]

    # ratio = 100.0 * len(np.where(y_true == 0)[0]) / len(y_true)
    # thresh = np.percentile(y_score, ratio)
    # y_pred = (y_score >= thresh).astype(int)
    top_k = len(np.where(y_true == 1)[0])
    indices = np.argpartition(y_score, -top_k)[-top_k:]
    y_pred = np.zeros_like(y_true)
    y_pred[indices] = 1

    y_true = y_true.astype(int)
    p, r, f1, support = metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )

    roc_auc_score = metrics.roc_auc_score(y_true, y_score)

    print(f'ROC AUC: {roc_auc_score:.4f}')


if __name__ == '__main__':
    main()
