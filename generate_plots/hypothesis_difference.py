from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    tasks = ['find_peaks', 'frequency_band', 'predict_ts_stationarity', 'set_points']
    current = {}
    for task in tasks:
        final_res = Path('end_results') / f'005_prompts_variations_{task}.csv'
        # Open the CSV file
        df = pd.read_csv(final_res)

        # Replace -1 with NaN
        df.replace(-1, pd.NA, inplace=True)
        # Take the mean across axis
        df_mean = df.mean(axis=0)
        current[task] = df_mean

    # Create a dot plot where the x-axis is the mask and the y-axis is the mean accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    for task, values in current.items():
        ax.plot(values.index, values.values, marker='o', label=task)
    ax.set_xlabel('Mask')
    ax.set_ylabel('Mean Accuracy')
    ax.set_title('Mean Accuracy vs Mask')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Save the plot
    plt.savefig('67ff59b20231d9f95909f426/figs/mean_accuracy_vs_mask.png', dpi=300)


if __name__ == '__main__':
    main()
