from pathlib import Path

import pandas as pd

RUN_COUNTER_FILE = Path('run_counter.txt')
RUN_COUNTER_LIMIT = 10**12  # safeguard against accidental infinite loops


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
    if isinstance(first_val, str) and not first_val.isdigit():
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


def main():
    current = _update_run_counter()
    csv_gt = list(Path(__file__).parent.glob('*.csv'))[0]
    df = pd.read_csv(csv_gt)
    y_score = parse_submission_file('/workspace/solution.csv', df)
    # y_score = pd.read_csv('/workspace/solution.csv', header=None).values.flatten()
    # if len(y_score) > len(df):
    #     y_score = y_score[1:]  # remove the first element, which is the header

    y_score = y_score.astype(int)
    y_true = df.iloc[:, -1].values

    # Compute accuracy
    accuracy = (y_true == y_score).mean()
    print(f'Accuracy on test set {current}: {accuracy:.4f}')

    # Save the accuracy to a file (if already exists, append)
    if Path('/mnt/accuracy.txt').exists():
        with open('/mnt/accuracy.txt', 'a') as f:
            f.write(f'Accuracy on test set {current}: {accuracy:.4f} \n')
    else:
        with open('/mnt/accuracy.txt', 'w') as f:
            f.write(f'Accuracy on test set {current}: {accuracy:.4f} \n')


if __name__ == '__main__':
    main()
