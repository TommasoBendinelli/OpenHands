#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets   # noqa: E402


def has_variance_burst(x: np.ndarray, window: int = 30, thresh: float = 3.0) -> int:
    """
    Detect a high-variance episode via rolling std.

    :return: 0 if no burst, 1 if burst
    """
    roll_std = pd.Series(x).rolling(window).std().fillna(0).values
    burst = (roll_std > thresh * roll_std.mean()).any()
    return int(burst)


def generate_signal(burst: bool, length: int = 300, base_std: float = 0.5):
    """
    White noise with or without a middle-segment variance spike.
    """
    x = np.random.normal(0, base_std, size=length)
    if burst:
        start = np.random.randint(length // 4, length // 2)
        end = start + np.random.randint(length // 10, length // 4)
        x[start:end] = np.random.normal(0, base_std * 5, size=end - start)
    return x


def create_dataset(
    n_samples: int = 200,
    length: int = 300,
    output_folder: Path | str = 'variance_dataset.csv',
):
    data, labels = [], []
    for _ in range(n_samples // 2):
        x = generate_signal(False, length)
        data.append(x)
        labels.append(0)

        x = generate_signal(True, length)
        data.append(x)
        labels.append(1)

    df = pd.DataFrame(np.vstack(data))
    df['label'] = labels
    return df


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    train_df = create_dataset(output_folder=out_dir)
    test_df = create_dataset(n_samples=200, length=500, output_folder=out_dir)

    save_datasets(train_df, test_df, output_folder=out_dir)

    # sanity-check plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_df.iloc[0, :-1])
    plt.title("No burst (label 0)")
    plt.subplot(1, 2, 2)
    plt.plot(train_df.iloc[1, :-1])
    plt.title("Variance burst (label 1)")
    plt.tight_layout()
    plt.show()
    plt.savefig(out_dir / "variance_dataset_example.png")