import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# tabular_datasets = [
#     # "cofounded_group_outlier",
#     "dominant_feature",
#     # "ground_mean_threshold",
#     "outlier_ratio",
#     # "row_max_abs",
#     # "row_variance",
#     "sign_rotated_generator",
#     "sum_threshold",
# ]


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

    # tabular_datasets = [
    #     # "cofounded_group_outlier",
    #     "dominant_feature",
    #     # "ground_mean_threshold",
    #     "outlier_ratio",
    #     # "row_max_abs",
    #     # "row_variance",
    #     "sign_rotated_generator",
    #     "sum_threshold",
    # ]

    # fig = plt.figure(figsize=(2, 2))

    # for i, dataset_dir in enumerate(tabular_datasets):
    #     # Read the CSV files
    #     train_path = os.path.join(dataset_dir, "train.csv")
    #     labels_path = os.path.join(dataset_dir, "train_labels.csv")

    #     train_df = pd.read_csv(train_path)  # no header
    #     labels_df = pd.read_csv(labels_path)

    #     # Add labels directly as a new column
    #     train_df['label'] = labels_df['label']

    #     if tabular_datasets == "outlier_ratio":
    #         # sanity plot: distribution of outlier ratios
    #         def outlier_ratio(df):
    #             return df.assign(is_out=lambda d: d.signal.abs() > OUTLIER_CUTOFF) \
    #                     .groupby("group_id")["is_out"].mean()

    #         r      = outlier_ratio(train)
    #         y_grp  = train.groupby("group_id")["label"].first()
    #         plt.hist(r[y_grp==0], bins=30, alpha=.6, label="label 0")
    #         plt.hist(r[y_grp==1], bins=30, alpha=.6, label="label 1")
    #         plt.axvline(THRESH_RATIO, ls="--", c="k", label=f"threshold {THRESH_RATIO}")
    #         plt.xlabel("outlier ratio  |signal|>3"); plt.ylabel("# groups")
    #         plt.title("Anomalous vs normal groups – clear split at 0.08")
    #         plt.legend(); plt.tight_layout(); plt.show()
    #         plt.savefig(out / "outlier_ratio_example.png")

    #     # if tabular_datasets == "row_max_abs":
    #     #     # ------------------- sanity plot -------------------------------------
    #     #     row_max = train.filter(regex="^feat").abs().max(axis=1)
    #     #     plt.hist(row_max[train.label == 0], bins=40, alpha=0.6, label="label 0")
    #     #     plt.hist(row_max[train.label == 1], bins=40, alpha=0.6, label="label 1")
    #     #     plt.axvline(THRESH, ls="--", color="k", label=f"threshold = {THRESH}")
    #     #     plt.title("Row-wise max(|x|) cleanly separates the classes")
    #     #     plt.xlabel("max(|feature|)"); plt.legend()

    #     if tabular_datasets = "dominant_feature":
    #         from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    #         fig = plt.figure(figsize=(6, 5))
    #         ax = fig.add_subplot(111, projection='3d')
    #         idx0 = train_df['label'] == 0
    #         ax.scatter(train_df.loc[idx0, '1'], train_df.loc[idx0, '2'],
    #                 train_df.loc[idx0, '3'], alpha=0.4, label='label 0')
    #         ax.scatter(train_df.loc[~idx0, '1'], train_df.loc[~idx0, '2'],
    #                 train_df.loc[~idx0, '3'], alpha=0.4, label='label 1')
    #         ax.set_xlabel('1'); ax.set_ylabel('2'); ax.set_zlabel('3')

    #     if tabular_datasets == "sign_rotated_generator":
    #         # visual sanity-check
    #         plt.figure(figsize=(5, 5))
    #         same = train_df["label"] == 1
    #         plt.scatter(train_df.loc[same,  "feat1"],
    #                     train_df.loc[same,  "feat2"], alpha=0.4, label="label 1")
    #         plt.scatter(train_df.loc[~same, "feat1"],
    #                     train_df.loc[~same, "feat2"], alpha=0.4, label="label 0")
    #         plt.axhline(0, c='k', lw=0.7); plt.axvline(0, c='k', lw=0.7)
    #         plt.title("Rotated quadrants with margin |u·v| ≥ {:.2f}".format(MARGIN))
    #         plt.legend(); plt.tight_layout()

    #     if tabular_datasets == "sum_threshold":
    #         plt.figure(figsize=(6, 4))
    #         sums = train_df[['feat1', 'feat2', 'feat3']].sum(axis=1)
    #         plt.hist(sums[train_df['label'] == 0], bins=30, alpha=0.6, label='label 0')
    #         plt.hist(sums[train_df['label'] == 1], bins=30, alpha=0.6, label='label 1')
    #         plt.axvline(1.5, ls='--', label='threshold')
    #         plt.title('Sum of first 3 features splits the classes'); plt.legend()
    #         plt.tight_layout(); plt.show()
    #         plt.savefig(out_dir / 'sum_dataset_example.png')

    # # plt.subplots_adjust(top=0.95, bottom=0.03, left=0.05, right=0.95)
    # plt.subplots_adjust(
    #     top=0.9, bottom=0.0002, left=0.05, right=0.95, wspace=0.4, hspace=0.05
    # )
    # # plt.subplots_adjust(wspace=60, hspace=0.03)
    # # plt.tight_layout()
    # plt.savefig("overview_fig_v4_4.png", dpi=300, bbox_inches='tight', pad_inches=0)

    # Your dataset list
    tabular_datasets = [
        # 'dominant_feature',
        "row_max_abs",
        'outlier_ratio',
        'sign_rotated_generator',
        'ground_mean_threashold',
        # "sum_threshold",
    ]

    # Constants (you need to define them properly)
    OUTLIER_CUTOFF = 3
    THRESH_RATIO = 0.08
    MARGIN = 0.5

    # Prepare the main figure
    fig, axes = plt.subplots(2, 2, figsize=(2.5, 1.5))
    axes = axes.flatten()  # Flatten to easily loop

    # Define your font size constants
    FONT_SIZE_TITLE = 2
    FONT_SIZE_LABEL = 2
    FONT_SIZE_TICKS = 2
    LINE_WIDTH = 0.5
    SCATTER_SIZE = 0.5

    for i, dataset_dir in enumerate(tabular_datasets):
        ax = axes[i]

        # After you create your plot
        for spine in ax.spines.values():
            spine.set_linewidth(0.2)

        # Make ticks consistent + set TICK FONT SIZE
        ax.tick_params(
            axis='both',
            which='both',
            width=0.2,
            length=2,
            direction='out',
            pad=1,
            labelsize=FONT_SIZE_TICKS,  # <<< set tick font size here
        )

        # Read the CSV files
        # train_path = os.path.join(dataset_dir, 'train.csv')
        # labels_path = os.path.join(dataset_dir, 'train_labels.csv')

        # train_df = pd.read_csv(train_path)
        # labels_df = pd.read_csv(labels_path)

        # # Add labels
        # train_df['label'] = labels_df['label']

        test_path = os.path.join(dataset_dir, 'test.csv')
        labels_path = os.path.join(dataset_dir, 'test_gt.csv')

        test_df = pd.read_csv(test_path)  # no header
        labels_df = pd.read_csv(labels_path)
        # Add labels directly as a new column
        test_df['label'] = labels_df['label']

        # to not change the code
        data = test_df

        # Plotting for different datasets
        if dataset_dir == 'outlier_ratio':

            def outlier_ratio(df):
                return (
                    df.assign(is_out=lambda d: d.signal.abs() > OUTLIER_CUTOFF)
                    .groupby('group_id')['is_out']
                    .mean()
                )

            r = outlier_ratio(data)
            y_grp = data.groupby('group_id')['label'].first()

            ax.hist(r[y_grp == 0], bins=10, label='label 0')
            ax.hist(r[y_grp == 1], bins=10, label='label 1')
            # ax.axvline(THRESH_RATIO, ls='--', c='k', lw=LINE_WIDTH)

            ax.set_xlabel('outlier ratio |signal|>3', fontsize=FONT_SIZE_LABEL)
            ax.set_ylabel('# groups', fontsize=FONT_SIZE_LABEL)
            ax.set_title(dataset_dir, fontsize=FONT_SIZE_TITLE, pad=2)

        elif dataset_dir == 'row_max_abs':
            row_max = data.filter(regex='^feat').abs().max(axis=1)
            ax.hist(row_max[data['label'] == 0], bins=40, label='label 0')
            ax.hist(row_max[data['label'] == 1], bins=40, label='label 1')
            # ax.axvline(OUTLIER_CUTOFF, ls='--', c='k', lw=LINE_WIDTH)

            ax.set_xlabel('max(|feature|)', fontsize=FONT_SIZE_LABEL)
            ax.set_ylabel('# rows', fontsize=FONT_SIZE_LABEL)
            ax.set_title(dataset_dir, fontsize=FONT_SIZE_TITLE, pad=2)

        elif dataset_dir == 'sign_rotated_generator':
            same = data['label'] == 1
            ax.scatter(
                data.loc[same, 'feat1'],
                data.loc[same, 'feat2'],
                label='label 1',
                s=SCATTER_SIZE,
            )
            ax.scatter(
                data.loc[~same, 'feat1'],
                data.loc[~same, 'feat2'],
                label='label 0',
                s=SCATTER_SIZE,
            )

            # ax.axhline(0, c='k',lw=LINE_WIDTH)
            # ax.axvline(0, c='k',lw=LINE_WIDTH)

            ax.set_xlabel('Feature 1', fontsize=FONT_SIZE_LABEL)
            ax.set_ylabel('Feature 2', fontsize=FONT_SIZE_LABEL)
            ax.set_title(dataset_dir, fontsize=FONT_SIZE_TITLE, pad=2)

        elif dataset_dir == 'ground_mean_threashold':
            N_GROUPS = 120
            ROWS_PER_GRP = 20
            HIGH_ROW_FRAC = 0.60  # fraction of “high-μ” rows in a positive group
            LOW_MU = 0.0
            HIGH_MU = 6.0
            THRESH = 3.0
            DISTRACTOR_COLS = 3

            grp_mean = data.groupby('group_id')['value_0'].mean()
            grp_label = data.groupby('group_id')['label'].first()

            ax.hist(
                grp_mean[grp_label == 0],
                bins=30,
                label='label 0',
            )
            ax.hist(
                grp_mean[grp_label == 1],
                bins=30,
                label='label 1',
            )
            # ax.axvline(THRESH, ls='--', c='k',lw=LINE_WIDTH)

            ax.set_xlabel(
                'group mean of value_0',
            )
            ax.set_ylabel(
                '# groups',
            )
            ax.set_title(
                dataset_dir,
                fontsize=FONT_SIZE_TITLE,
                pad=2,
            )
            ax.tick_params(axis='both')

        # ⚠️ DO NOT overwrite labels here!
        # These two lines must be REMOVED:
        ax.set_xlabel('')
        ax.set_ylabel('')
        # Set x-label, y-label, title with font size

    # Layout adjustment
    # plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01)
    #
    # # plt.subplots_adjust(
    #     top=0.9, bottom=0.0002, left=0.05, right=0.95, wspace=0.4, hspace=0.05
    # )
    # plt.tight_layout()

    # plt.subplots_adjust(
    #     top=0.9, bottom=0.0002, left=0.05, right=0.95, wspace=0.4, hspace=0.05
    # )

    # plt.subplots_adjust(wspace=0.4, hspace=0.5)
    plt.subplots_adjust(hspace=-10)
    plt.tight_layout()
    plt.savefig(
        'overview_fig_tabular_4_smaller_smaller_NEW_test.pdf',
        dpi=400,
        bbox_inches='tight',
        pad_inches=0.05,
    )
    # plt.savefig(
    #     'overview_fig_tabular_4_smaller_smaller_final.png',
    #     dpi=1000,
    #     bbox_inches='tight',
    #     pad_inches=0.05,
    # )

    return


if __name__ == '__main__':
    main()
