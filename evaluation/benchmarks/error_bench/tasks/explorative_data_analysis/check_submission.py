from pathlib import Path

import pandas as pd

RUN_COUNTER_FILE = Path('run_counter.txt')
RUN_COUNTER_LIMIT = 10**12  # safeguard against accidental infinite loops


def parse_submission_file(path: Path, df: pd.DataFrame) -> pd.Series:
    y_score = pd.read_csv(path, header=None)

    # Check if the first value is a non-digit string → treat the first row as a header
    first_val = y_score.iloc[0, 0]
    if isinstance(first_val, str) and not first_val.isdigit():  # ← E721 fixed
        y_score.columns = y_score.iloc[0]
        y_score = y_score.iloc[1:]

    # If an extra row slipped in (e.g., duplicated header), drop it
    if len(y_score) > len(df):
        y_score = y_score.iloc[1:]

    # Flatten dataframe to a 1-D vector
    if y_score.shape[1] > 1:
        y_score = y_score['label'].to_numpy().flatten()
    else:
        y_score = y_score.to_numpy().flatten()

    return y_score


def main():
    csv_gt = list(Path(__file__).parent.glob('*.csv'))[0]
    df = pd.read_csv(csv_gt)
    y_score = parse_submission_file('/workspace/solution.csv', df)
    # y_score = pd.read_csv('/workspace/solution.csv', header=None) #.values.flatten()
    # # Check if there is a digit at level 0
    # if not y_score.iloc[0, 0].isdigit():
    #     # Set this as header
    #     y_score.columns = y_score.iloc[0]
    #     y_score = y_score[1:]
    # if len(y_score) > len(df):
    #     y_score = y_score[1:]  # remove the first element, which is the header
    # # Check whether there are multiple columns
    # if y_score.shape[1] > 1:
    #     # Keep the column with "label" in the name
    #     y_score = y_score['label'].values.flatten()
    # else:
    #     y_score = y_score.values.flatten()

    y_true = df.iloc[:, -1].values

    # Compute accuracy
    if len(y_true) != len(y_score):
        raise ValueError(
            f'Length of y_true ({len(y_true)}) and y_score ({len(y_score)}) do not match.'
        )

    try:
        y_score = y_score.astype(int)
    except ValueError:
        raise ValueError(
            'y_score contains non-integer values. Please ensure that the predictions are integers.'
        )
    print('Submission is valid.')


if __name__ == '__main__':
    main()
