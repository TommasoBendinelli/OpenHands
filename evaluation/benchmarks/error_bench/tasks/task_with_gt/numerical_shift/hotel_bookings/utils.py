import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import pandas as pd
import pydantic
from sklearn.ensemble import RandomForestClassifier


@pydantic.dataclasses.dataclass
class CorruptedDatasetJson:
    clean_df_shape: tuple
    clean_performance_test: float
    # baseline_performance_train: float
    df_shape: tuple
    performance_test: float
    # performance_train: float
    pipeline_code: str
    hints: List[str]
    corrupted_indices: Dict[int, Any]  # Lazy to define the schema
    dataset_description: str
    target_column: str


class BasePipeline:
    @classmethod
    def preprocess(cls, df: pd.DataFrame, drop_competition_index=True) -> pd.DataFrame:
        df = copy.deepcopy(df)
        # Drop _competition_idx
        if drop_competition_index:
            df.drop(
                columns=['_competition_index'],
                inplace=True,
            )
        return df

    @classmethod
    def compute_metrics(cls, y, y_pred) -> dict:
        from sklearn.metrics import f1_score

        metrics = {}
        metrics['F1_score'] = f1_score(y, y_pred, average='macro')

        return metrics

    @classmethod
    def run_evaluation(cls, df: pd.DataFrame, df_test) -> Tuple[dict, pd.Series]:
        metrics = {}
        for seed in [1, 2, 3]:
            clf, _, feature_importance = cls.train_model(df, seed)
            metric = cls.inference_model(df_test, clf)
            metrics[seed] = metric
        metrics = pd.DataFrame(metrics).mean(axis=1).to_dict()
        # Sort the feature importance by descending order
        return metrics, feature_importance

    @classmethod
    def train_model(
        cls, df: pd.DataFrame, seed
    ) -> Tuple[RandomForestClassifier, dict, pd.Series]:
        from sklearn.ensemble import RandomForestClassifier

        X, y = cls.preprocess(df)
        clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        clf.fit(X, y)
        # Measure TP, TN, FP, FN
        y_pred = clf.predict(X)
        metrics = cls.compute_metrics(y, y_pred)
        return clf, metrics, pd.Series(clf.feature_importances_, index=X.columns)

    @classmethod
    def inference_model(cls, df: pd.DataFrame, rf: 'Classifier') -> dict:  # noqa: F821
        X, y = cls.preprocess(df, drop_competition_index=False)
        y_pred = rf.predict(X)
        metrics = cls.compute_metrics(y, y_pred)
        return metrics


def run_sanity_check_print(metadata):
    print('Clean performance on test set: ', metadata['clean_performance_test'])
    print('Corrupted performance on test set: ', metadata['performance_test'])
    print('Clean shape: ', metadata['clean_df_shape'])
    print('Corrupted shape: ', metadata['df_shape'])


def compute_corrupted_indices(
    bool_series, column, df, values_before=None, values_after=None
) -> str:
    indices = df.loc[bool_series, '_competition_index']
    corrupted_indices = [(idx, column) for idx in indices]
    df = pd.DataFrame(corrupted_indices, columns=['_competition_index', 'column'])
    df['before'] = values_before
    df['after'] = values_after
    return df.to_json()


def read_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Reads the CSV file into a DataFrame.

    :param csv_path: A Path object pointing to the CSV file
    :return: A pandas DataFrame containing the CSV data
    """
    return pd.read_csv(csv_path)


def submit_clean_data(path_to_clean_data: str, context_variables: dict | None = None):
    """
    Submit the cleaned data to train the model and get the performance.
    Make sure that the train_cleaned_v*.csv exists and it's updated before calling the function! Each time this function is called, remember to update i.e. first time should be train_cleaned_v1.csv, second time should be train_cleaned_v2.csv, etc.
    If you don't do this, it will result in an error.

    """
    if context_variables is None:
        context_variables = {}
    # Load the dataset from the sandbox
    assert (
        'Pipeline' in context_variables
    ), "The context_variables dictionary must contain the 'run_model' key"
    try:
        context_variables['running_statistics_submission'].append(
            context_variables['running_statistics_current']
        )
        if not Path(path_to_clean_data).exists():
            context_variables['cleaned_datasets_path'].append(
                f'Invalid submission {submit_clean_data.invalid_submission}'
            )
            submit_clean_data.invalid_submission += 1
            raise ValueError(
                f'Your current submission {path_to_clean_data} does not exist. Make sure that the dataset exists.'
            )
        df = pd.read_csv(path_to_clean_data)

        if path_to_clean_data in context_variables['cleaned_datasets_path']:
            context_variables['cleaned_datasets_path'].append(path_to_clean_data)
            raise ValueError(
                'This dataset path has already been submitted. Please submit a new dataset path with the index incremented by 1.'
            )
        else:
            context_variables['cleaned_datasets_path'].append(path_to_clean_data)

        # Create a folder to save the cleaned datasets
        cleaned_datasets_folder = Path('submission_datasets')
        cleaned_datasets_folder.mkdir(parents=True, exist_ok=True)
        # Save the dataset
        df.to_csv(
            cleaned_datasets_folder / f'{Path(path_to_clean_data).name}', index=False
        )

        detective_path = Path(
            hydra.utils.to_absolute_path(
                f"dataset/{context_variables['cfg'].dataset.dataset_name}/detective"
            )
        )
        df_test = pd.read_csv(detective_path / 'test.csv')
        additional_columns = set(df.columns) - set(df_test.columns)

        if '_competition_index' not in df.columns:
            raise ValueError(
                "The column '_competition_index' is missing from the training data. Please add it to the training data."
            )
        additional_columns = additional_columns - {'_competition_index'}
        if additional_columns != set():
            raise ValueError(
                f"The columnes {additional_columns} are present in the training data but not in the test data. Please remove them from the training data. You can't add new columns to the test data."
            )
        less_columns = set(df_test.columns) - set(df.columns)
        if less_columns:
            # Drop the columns from the test set that are not in the training set
            df_test = df_test.drop(columns=list(less_columns), inplace=False)
        pipeline_class = context_variables['Pipeline']
        result, _ = pipeline_class.run_evaluation(df, df_test)
        # clf, _ = pipeline_class.train_model(df)

        # result = pipeline_class.inference_model(df_test, clf)
        to_append = result[context_variables['target_metric']]
    except Exception as e:
        to_append = 'An error occurred while running the model {}'.format(str(e))

    context_variables['performance_obtained'].append(to_append)
    return to_append


submit_clean_data.invalid_submission = 0
