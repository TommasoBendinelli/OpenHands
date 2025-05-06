from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    # Save CSVs identical to previous API
    train_df.to_csv(output_folder / 'train.csv', index=False)
    tmp = pd.read_csv(output_folder / 'train.csv')
    y_test = test_df['label']
    y_test.to_csv(output_folder / 'test_gt.csv', index=False)
    X_test = test_df.drop(columns=['label'])
    X_test.to_csv(output_folder / 'test.csv', index=False)

    # Quick visual sanity-check
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sig0 = train_df[train_df['label'] == 0].iloc[0, :-1]
    plt.plot(sig0)
    plt.title('Class 0 (≤1 set-point)')

    plt.subplot(1, 2, 2)
    sig1 = train_df[train_df['label'] == 1].iloc[0, :-1]
    plt.plot(sig1)
    plt.title('Class 1 (≥2 set-points)')

    plt.tight_layout()
    plt.show()
    plt.savefig(output_folder / 'setpoint_examples.png')
