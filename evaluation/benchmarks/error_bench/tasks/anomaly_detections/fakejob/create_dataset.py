import os
import pickle
from pathlib import Path
from typing import Optional

import gensim.downloader as api
import numpy as np
import pandas as pd
from feature_engine.encoding import RareLabelEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def split_data(
    n_splits: int,
    train_ratio: Optional[float] = 0.5,
    split_dir: Optional[Path] = None,
    y: Optional[np.ndarray] = None,
) -> tuple:  # list of train indices and test indices
    train_indices, test_indices = [], []
    for i in range(n_splits):
        pkl_file = split_dir / 'index{}.pkl'.format(i)
        with open(pkl_file, 'rb') as f:
            train_index, test_index = pickle.load(f)

        train_indices.append(train_index)
        test_indices.append(test_index)
    return train_indices, test_indices


def df_to_numpy(
    X: pd.DataFrame,
    method: Optional[str] = 'ordinal',
    normalize_numbers: Optional[bool] = False,
    verbose: Optional[bool] = False,
    textual_encoding: Optional[
        str
    ] = 'word2vec',  # bag_of_words, tfidf, word2vec, or none
    textual_columns: Optional[list] = None,
) -> np.ndarray:
    # if dataset_name == 'ecoli':
    #     X_np = X.drop(X.columns[0], axis=1).to_numpy()
    #     return X_np

    numeric_data = X.select_dtypes(
        include=['float64', 'int64', 'uint8', 'int16', 'float32']
    )
    numeric_columns = numeric_data.columns.tolist()
    categorical_data = X.select_dtypes(include=['object', 'category'])
    categorical_columns = categorical_data.columns.tolist()

    if verbose:
        print('Number of categorical data', len(categorical_columns))
        print('Categorical columns:', categorical_columns)

    # fill na
    if len(numeric_columns) > 0:
        for numeric_col in numeric_columns:
            X[numeric_col] = X[numeric_col].fillna(X[numeric_col].mean())

        if normalize_numbers:
            # normalize it to have zero mean and unit variance
            scaler = StandardScaler()
            X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

    # Handle textual data
    if textual_encoding == 'none' and len(textual_columns) > 0:
        for col in textual_columns:
            categorical_columns.remove(col)
        X = X.drop(columns=textual_columns)
        textual_columns = []

    if len(textual_columns) > 0:
        if textual_encoding == 'word2vec':
            model = api.load('word2vec-google-news-300')
            tmp = X[textual_columns].agg(' '.join, axis=1)
            X_vecs = []
            for i in range(len(X)):
                words = []
                for word in tmp[i].split():
                    if word in model.key_to_index:
                        words.append(word)
                # Compute the average word embedding
                if words:  # Ensure there are valid words left
                    word_vectors = [model[word] for word in words]
                    X_vec = np.mean(word_vectors, axis=0)
                else:
                    X_vec = np.zeros(
                        model.vector_size
                    )  # Handle the case where no words are in the vocabulary
                X_vecs.append(X_vec)
            X_vecs = np.array(X_vecs)
        for col in textual_columns:
            categorical_columns.remove(col)

    if len(categorical_columns) > 0:
        # categorical features:
        # group categories with low frequency into a single category
        encoder = RareLabelEncoder(
            tol=0.01,  # Minimum frequency to be considered as a separate class
            max_n_categories=None,  # Maximum number of categories to keep
            replace_with='Rare',  # Value to replace rare categories with
            variables=categorical_columns,  # Columns to encode
            missing_values='ignore',
        )
        X = encoder.fit_transform(X)

        # Remove columns that contain identical values
        X = X.loc[:, (X != X.iloc[0]).any()]

        # remove categories that have only one value
        for column in categorical_columns:
            if X[column].nunique() == 1:
                X.drop(column, inplace=True, axis=1)

        if method == 'ordinal':
            le = LabelEncoder()
            for i in categorical_data.columns:
                categorical_data[i] = le.fit_transform(categorical_data[i])
        elif method == 'one_hot':
            enc = OneHotEncoder(
                handle_unknown='ignore', sparse_output=False, drop='first'
            )
            one_hot_encoded = enc.fit_transform(X[categorical_columns])
            categorical_data = pd.DataFrame(
                one_hot_encoded, columns=enc.get_feature_names_out(categorical_columns)
            )
        else:
            raise ValueError('Invalid method. Choose either ordinal or one_hot')

        X_prime = X.drop(categorical_columns, axis=1)
        X = pd.concat([X_prime, categorical_data], axis=1)

    # remove columns that contain identical values
    print(X.shape)
    X = X.loc[:, (X != X.iloc[0]).any()]
    X_np = X.to_numpy()
    return X_np


def main():
    # Get __file__ path
    file_folder = Path(__file__).resolve().parent
    df = pd.read_csv(file_folder / 'fake_job_postings.csv')

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
    y = df['fraudulent'].map({0: 0, 1: 1})
    # Split the dataset into training and test set
    # 80% for training and 20% for test
    X = df.drop(columns=['fraudulent'])
    X_np = df_to_numpy(
        X,
        method='one_hot',
        verbose=True,
        textual_encoding='word2vec',
        textual_columns=[],
    )
    train_ratio = 0.5
    train_indices, test_indices = split_data(
        5, train_ratio, file_folder / 'unsupervised/split5', y=y
    )
    # Iterate over the splits
    cnt = 0
    for train_indice, test_indice in zip(train_indices, test_indices):
        # Get the train and test data
        if type(X_np) == pd.DataFrame:
            train_df = X_np.iloc[train_indice]
            test_df = X_np.iloc[test_indice]
        elif type(X_np) == np.ndarray:
            train_df = X_np[train_indice]
            test_df = X_np[test_indice]
        else:
            raise ValueError('df should be either a pandas DataFrame or a numpy array.')
        # Drop the y class from the train and test data

        y_gt = df.iloc[test_indice]['fraudulent'].values
        # train_df.drop(columns=["class"], inplace=True)
        # test_df.drop(columns=["class"], inplace=True)
        # # Save the training and test set to csv files
        trg_folder = file_folder / Path(f'fold_{cnt}')
        os.makedirs(trg_folder, exist_ok=True)
        # Transform the data to a pandas DataFrame if it is a numpy array
        if isinstance(train_df, np.ndarray):
            train_df = pd.DataFrame(train_df)
        if isinstance(test_df, np.ndarray):
            test_df = pd.DataFrame(test_df)
        train_df.to_csv(trg_folder / 'train.csv', index=False)
        test_df.to_csv(trg_folder / 'test.csv', index=False)
        y_gt_df = pd.DataFrame(y_gt, columns=['class'])
        y_gt_df.to_csv(trg_folder / 'test_gt.csv', index=False)
        cnt += 1

        # # Get the train and test data
        # train_df = df.iloc[train_indice]
        # test_df = df.iloc[test_indices]
        # # Drop the y class from the train and test data
        # y_gt = test_df['fraudulent'].values
        # train_df.drop(columns=["fraudulent"], inplace=True)
        # test_df.drop(columns=["fraudulent"], inplace=True)

        # # Save the training and test set to csv files
        # trg_folder = Path(file_folder/ f"fold_{cnt}")
        # os.makedirs(trg_folder, exist_ok=True)
        # train_df.to_csv(trg_folder / "train.csv", index=False)
        # test_df.to_csv(trg_folder / "test.csv", index=False)
        # y_gt_df = pd.DataFrame(y_gt, columns=["fraudulent"])
        # y_gt_df.to_csv(trg_folder / "test_gt.csv", index=False)
        # cnt += 1


if __name__ == '__main__':
    main()
