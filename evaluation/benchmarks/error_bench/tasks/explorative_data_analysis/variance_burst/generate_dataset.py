#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from typing import Tuple
import math
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
    
    # Randomly add a bias term to the signal +/- 5
    bias = np.random.uniform(-5, 5)
    x += bias


    return x


def create_dataset(
    n_samples: int = 200,
    length: int = 300,
    output_folder: Path | str = 'variance_dataset.csv',
    base_std: float = 0.5,
):
    data, labels = [], []
    for _ in range(n_samples // 2):
        x = generate_signal(False, length,base_std=base_std)
        data.append(x)
        labels.append(0)

        x = generate_signal(True, length, base_std=base_std)
        data.append(x)
        labels.append(1)

    df = pd.DataFrame(np.vstack(data))
    df['label'] = labels
    return df


def main():
     # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    output_folder = Path(__file__).resolve().parent

    out_dir = Path(__file__).resolve().parent
    base_std = 0.5
    train_df = create_dataset(output_folder=out_dir, base_std=base_std)
    test_df = create_dataset(n_samples=200, length=500, output_folder=out_dir, base_std=0.1)

    save_datasets(train_df, test_df, output_folder=out_dir)

 
    # ── Plotting 10 examples ────────────────────────────────────────────────
    num_examples = 10           # change this if you want a different count
    ncols        = 5            # 2 rows × 5 columns
    nrows        = math.ceil(num_examples / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(15, 6),
                             sharex=True, sharey=False)
    axes = axes.flatten()

    for i in range(num_examples):
        axes[i].plot(train_df.iloc[i, :-1])
        label = int(train_df.iloc[i, -1])
        axes[i].set_title(f"Example {i}  (label {label})")

    # Hide any leftover empty axes
    for j in range(num_examples, len(axes)):
        axes[j].axis("off")

    fig.suptitle("First 10 training-set examples", y=1.02, fontsize=14)
    plt.tight_layout()

    # Save *before* showing so the file isn’t blank if the program exits early
    fig.savefig(out_dir / "variance_dataset_examples.png")
    plt.show()

    # Also show test set examples
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(15, 6),
                             sharex=True, sharey=False)
    axes = axes.flatten()
    for i in range(num_examples):
        axes[i].plot(test_df.iloc[i, :-1])
        label = int(test_df.iloc[i, -1])
        axes[i].set_title(f"Example {i}  (label {label})")
    # Hide any leftover empty axes
    for j in range(num_examples, len(axes)):
        axes[j].axis("off")
    fig.suptitle("First 10 test-set examples", y=1.02, fontsize=14)
    plt.tight_layout()
    # Save *before* showing so the file isn’t blank if the program exits early
    fig.savefig(out_dir / "variance_dataset_test_examples.png")




if __name__ == "__main__":
    main()