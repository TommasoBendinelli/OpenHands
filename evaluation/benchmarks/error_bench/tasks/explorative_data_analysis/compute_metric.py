from pathlib import Path

import pandas as pd

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
    _update_run_counter()
    csv_gt = list(Path(__file__).parent.glob('*.csv'))[0]
    df = pd.read_csv(csv_gt)

    y_score = pd.read_csv('/workspace/solution.csv', header=None).values.flatten()
    if len(y_score) > len(df):
        y_score = y_score[1:]  # remove the first element, which is the header
    y_true = df.iloc[:, -1].values

    # Compute accuracy
    accuracy = (y_true == y_score).mean()
    print(f'Accuracy: {accuracy:.4f}')


if __name__ == '__main__':
    main()
