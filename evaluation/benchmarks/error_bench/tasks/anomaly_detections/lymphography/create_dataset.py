import pandas as pd


def main():
    df = pd.read_csv(
        'evaluation/benchmarks/error_bench/tasks/anomaly_detections/fake_job_postings/fake_job_postings.csv'
    )

    # deal with Nan values
    df['location'].fillna('Unknown', inplace=True)
    df['department'].fillna('Unknown', inplace=True)
    df['salary_range'].fillna('Not Specified', inplace=True)
    df['employment_type'].fillna('Not Specified', inplace=True)
    df['required_experience'].fillna('Not Specified', inplace=True)
    df['required_education'].fillna('Not Specified', inplace=True)
    df['industry'].fillna('Not Specified', inplace=True)
    df['function'].fillna('Not Specified', inplace=True)
    df.drop('job_id', inplace=True, axis=1)

    text_columns = [
        'title',
        'company_profile',
        'description',
        'requirements',
        'benefits',
    ]
    df[text_columns] = df[text_columns].fillna('NaN')

    df.columns = [name.replace('_', ' ') for name in df.columns]

    # Split the dataset into training and test set
    # 80% for training and 20% for test
    train_df = df.sample(frac=0.5, random_state=42)
    test_df = df.drop(train_df.index)
    # Save this as ground truth
    test_df.to_csv(
        'evaluation/benchmarks/error_bench/tasks/anomaly_detections/fake_job_postings/test_gt.csv',
        index=False,
    )

    # Remove fraudulent job postings from the test set
    test_df.drop(columns=['fraudulent'], inplace=True)
    # Save the training and test set to csv files
    train_df.to_csv(
        'evaluation/benchmarks/error_bench/tasks/anomaly_detections/fake_job_postings/train.csv',
        index=False,
    )
    test_df.to_csv(
        'evaluation/benchmarks/error_bench/tasks/anomaly_detections/fake_job_postings/test.csv',
        index=False,
    )


if __name__ == '__main__':
    main()
