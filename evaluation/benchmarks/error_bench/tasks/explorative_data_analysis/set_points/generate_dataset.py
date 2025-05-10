import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets


# ---------------------------------------------------------------------
#  Core helpers
# ---------------------------------------------------------------------
def generate_synthetic_setpoint_signal(
    n_points: int,
    n_setpoints: int,
    noise_std: float = 0.3,
    mean_low: float = -3.0,
    mean_high: float = 3.0,
):
    """
    Create a piece-wise constant signal with `n_setpoints` mean shifts.
    A 'set-point' here is a change in the underlying level of the signal.
    """
    # Where do the changes occur?
    changepos = np.sort(
        np.random.choice(np.arange(1, n_points - 1), size=n_setpoints, replace=False)
    )
    changepos = np.append(changepos, n_points)  # ensure we reach the end

    # Build segments
    signal = np.zeros(n_points)
    start = 0
    for cp in changepos:
        mean_val = np.random.uniform(mean_low, mean_high)
        signal[start:cp] = mean_val
        start = cp

    # Add white noise
    signal += np.random.normal(scale=noise_std, size=n_points)
    return signal


def generate_dataset(
    num_samples: int = 300,
    n_points: int = 1000,
    noise_std: float = 0.3,
    mean_low: float = -3.0,
    mean_high: float = 3.0,
    min_setpoints_class1: int = 2,
    max_setpoints_class1: int = 6,
):
    """
    Returns a DataFrame where each row is a signal sample.
    * label 0 ➜ 0 or 1 set-point
    * label 1 ➜ ≥2 set-points
    """
    data, labels = [], []
    for _ in range(num_samples):
        label = int(np.random.choice([0, 1]))
        if label == 0:
            n_sp = np.random.choice([0, 1])  # 0 or 1 set-point
        else:
            n_sp = np.random.randint(min_setpoints_class1, max_setpoints_class1 + 1)
        sig = generate_synthetic_setpoint_signal(
            n_points,
            n_setpoints=n_sp,
            noise_std=noise_std,
            mean_low=mean_low,
            mean_high=mean_high,
        )
        data.append(sig)
        labels.append(label)

    df = pd.DataFrame(np.vstack(data))
    df['label'] = labels
    return df


# ---------------------------------------------------------------------
#  Script body (mirrors earlier examples)
# ---------------------------------------------------------------------
if __name__ == '__main__':
    output_folder = Path(__file__).resolve().parent

    # Two slightly different training pools for variety
    train_df_1 = generate_dataset(noise_std=0.2)
    train_df_2 = generate_dataset(noise_std=0.6)
    test_df = generate_dataset(noise_std=0.4)

    train_df = pd.concat([train_df_1, train_df_2], ignore_index=True)
    print(train_df.head())

    save_datasets(train_df=train_df, test_df=test_df, output_folder=output_folder)
    # Save CSVs identical to previous API
    # train_df.to_csv(output_folder / 'train.csv', index=False)
    # tmp = pd.read_csv(output_folder / 'train.csv')
    # y_test = test_df['label']
    # y_test.to_csv(output_folder / 'test_gt.csv', index=False)
    # X_test = test_df.drop(columns=['label'])
    # X_test.to_csv(output_folder / 'test.csv', index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=1000)

    # Plot for Class 0
    sig0 = train_df[train_df['label'] == 0].iloc[0, :-1]
    ax1.plot(sig0)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # Plot for Class 1
    sig1 = train_df[train_df['label'] == 1].iloc[0, :-1]
    ax2.plot(sig1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    for spine in ax2.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.show()
    plt.savefig("setpoint_signal.png", dpi=1000, bbox_inches='tight')

    # ---- Plot and save Class 0 ----
    sig0 = train_df[train_df['label'] == 0].iloc[0, :-1]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=1000)
    ax.plot(sig0)
    ax.axis('off')  # Remove ticks, labels, borders
    plt.tight_layout()
    plt.savefig('class_0.png', bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory

    # ---- Plot and save Class 1 ----
    sig1 = train_df[train_df['label'] == 1].iloc[0, :-1]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=1000)
    ax.plot(sig1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('class_1.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # # Save two separte figures for each class
    # plt.figure(figsize=(12, 6), dpi=1000)
    # plt.plot(sig0)
    # plt.
