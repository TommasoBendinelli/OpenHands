import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

time_series_datasets = [
    "find_peaks",
    "periodic_presence",
    "predict_ts_stationarity",
    "set_points",
    "simultanus_spike",
    "spike_presence",
    "variance_burst",
    "zero_crossing",
]

time_series_datasets_v2 = [
    # "find_peaks",
    "periodic_presence",
    "predict_ts_stationarity",
    "set_points",
    # "simultanus_spike",
    # "spike_presence",
    "variance_burst",
    # "zero_crossing",
]

tabular_datasets = [
    "cofounded_group_outlier",
    "dominant_feature",
    "ground_mean_threshold",
    "outlier_ratio",
    "row_max_abs",
    "row_variance",
    "sign_rotated_generator",
    "sum_threshold",
]


def main():
    # Open the datasets from the folder explorative_data_analysis

    # fig = plt.figure(figsize=(8, 2))
    # outer = gridspec.GridSpec(2, 4, wspace=0.15, hspace=-0.22)  # 2 rows, 4 columns

    # for i, dataset_dir in enumerate(time_series_datasets):
    #     # Read the CSV files
    #     train_path = os.path.join(dataset_dir, "train.csv")
    #     labels_path = os.path.join(dataset_dir, "train_labels.csv")

    #     train_df = pd.read_csv(train_path)  # no header
    #     labels_df = pd.read_csv(labels_path)

    #     # Add labels directly as a new column
    #     train_df['label'] = labels_df['label']

    #     # Select one sample from each class
    #     sample_class0 = train_df[train_df['label'] == 0].iloc[0, :-1]
    #     sample_class1 = train_df[train_df['label'] == 1].iloc[0, :-1]

    #     # Each outer grid cell
    #     inner = gridspec.GridSpecFromSubplotSpec(
    #         2,
    #         2,
    #         subplot_spec=outer[i],
    #         wspace=0.1,
    #         height_ratios=[0.1, 0.1],
    #         hspace=-0.005,
    #     )

    #     # Class 0 plot
    #     ax0 = plt.Subplot(fig, inner[0])
    #     ax0.tick_params(axis='both', labelsize=3)
    #     ax0.plot(sample_class0.values, linewidth=0.3)
    #     # ax0.set_title(f"{dataset_dir}_Class 0", fontsize=4)
    #     fig.add_subplot(ax0)

    #     # Class 1 plot
    #     ax1 = plt.Subplot(fig, inner[1])
    #     ax1.tick_params(axis='both', labelsize=3)
    #     ax1.tick_params(
    #         axis='y', left=False, labelleft=False
    #     )  # No y-ticks, no y-labels
    #     ax1.plot(sample_class1.values, linewidth=0.3)
    #     # ax1.set_title(f"Class 1", fontsize=4)
    #     fig.add_subplot(ax1)

    #     ax_title = plt.Subplot(fig, inner[0, :])
    #     ax_title.axis('off')
    #     ax_title.set_title(dataset_dir, fontsize=4, pad=3)
    #     fig.add_subplot(ax_title)

    # # plt.subplots_adjust(top=0.95, bottom=0.03, left=0.05, right=0.95)
    # plt.subplots_adjust(
    #     top=0.9, bottom=0.0002, left=0.05, right=0.95, wspace=0.4, hspace=0.05
    # )
    # # plt.subplots_adjust(wspace=60, hspace=0.03)
    # # plt.tight_layout()
    # plt.savefig("overview_fig_v4_8.png", dpi=300, bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(4, 2))
    outer = gridspec.GridSpec(2, 2, wspace=0.12, hspace=-0.22)  # 2 rows, 4 columns

    for i, dataset_dir in enumerate(time_series_datasets_v2):
        # Read the CSV files
        train_path = os.path.join(dataset_dir, "train.csv")
        labels_path = os.path.join(dataset_dir, "train_labels.csv")

        train_df = pd.read_csv(train_path)  # no header
        labels_df = pd.read_csv(labels_path)

        # Add labels directly as a new column
        train_df['label'] = labels_df['label']

        # Select one sample from each class
        sample_class0 = train_df[train_df['label'] == 0].iloc[0, :-1]
        sample_class1 = train_df[train_df['label'] == 1].iloc[0, :-1]

        # Each outer grid cell
        inner = gridspec.GridSpecFromSubplotSpec(
            2,
            2,
            subplot_spec=outer[i],
            wspace=0.1,
            height_ratios=[0.1, 0.1],
            hspace=-0.005,
        )

        # Class 0 plot
        ax0 = plt.Subplot(fig, inner[0])
        ax0.tick_params(axis='both', labelsize=3)
        ax0.plot(sample_class0.values, linewidth=0.3, color='#1f77b4')
        # ax0.set_title(f"{dataset_dir}_Class 0", fontsize=4)
        fig.add_subplot(ax0)

        # Class 1 plot
        ax1 = plt.Subplot(fig, inner[1])
        ax1.tick_params(axis='both', labelsize=3)
        ax1.tick_params(
            axis='y', left=False, labelleft=False
        )  # No y-ticks, no y-labels
        ax1.plot(sample_class1.values, linewidth=0.3, color='#ff7f0e')
        # ax1.set_title(f"Class 1", fontsize=4)
        fig.add_subplot(ax1)

        ax_title = plt.Subplot(fig, inner[0, :])
        ax_title.axis('off')
        ax_title.set_title(dataset_dir, fontsize=4, pad=3)
        fig.add_subplot(ax_title)

    # plt.subplots_adjust(top=0.95, bottom=0.03, left=0.05, right=0.95)
    plt.subplots_adjust(
        top=0.9, bottom=0.0002, left=0.05, right=0.95, wspace=0.4, hspace=0.05
    )
    # plt.subplots_adjust(wspace=60, hspace=0.03)
    # plt.tight_layout()
    plt.savefig(
        "overview_fig_timeseries_4.png", dpi=300, bbox_inches='tight', pad_inches=0
    )

    return


if __name__ == '__main__':
    main()
