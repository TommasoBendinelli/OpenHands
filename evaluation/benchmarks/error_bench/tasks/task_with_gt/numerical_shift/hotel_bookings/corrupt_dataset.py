import inspect
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import kagglehub
import numpy as np
import pandas as pd
from utils import (
    BasePipeline,
    CorruptedDatasetJson,
    compute_corrupted_indices,
    read_dataset,
    run_sanity_check_print,
)


class Pipeline(BasePipeline):
    # fmt: off
    customer_type = {'Transient': 0, 'Contract': 1, 'Transient-Party': 2, 'Group': 3}
    meal_mapping = {'BB': 0, 'HB': 1, 'SC': 2, 'Undefined': 3, 'FB': 4}
    market_segment = {'Online TA': 0, 'Offline TA/TO': 1, 'Direct': 2, 'Corporate': 3, 'Complementary': 4, 'Aviation': 5}
    distribution_channel = {'TA/TO': 0, 'Direct': 1, 'Corporate': 2, 'GDS': 3, 'Undefined': 4}
    deposit_type = {'No Deposit': 0, 'Refundable': 1, 'Non Refund': 2}
    countries = ['PRT', 'GBR', 'IRL', 'BRA', 'CN', 'ESP', 'ITA', 'DEU', 'FRA', 'AUS', 'LUX', 'AGO', 'RUS', 'POL', 'NLD', 'IDN', 'USA', 'CHE', 'AUT', 'NOR', 'LKA', 'BEL', 'ARG', 'FIN', 'MAR', 'DNK', 'LTU', 'GRC', 'ROU', 'KOR', 'SWE', 'JEY', 'NGA', 'HRV', 'JPN', 'LVA', 'SVN', 'ZAF', 'CYM', 'CHN', 'PHL', 'SRB', 'HUN', 'EST', 'DZA', 'CPV', 'GEO', 'GIB', 'ISR', 'CYP', 'CRI', 'FJI', 'SYC', 'MEX', 'KAZ', 'BHR', 'ARE', 'CZE', 'HKG', 'ZWE', 'SVK', 'BLR', 'UKR', 'NZL', 'TUR', 'MDV', 'ARM', 'COL', 'KWT', 'TUN', 'THA', 'IND', 'URY', 'BWA', 'MYS', 'CUB', 'GGY', 'CAF', 'LBN', 'AZE', 'IRN', 'PRI', 'PAK', 'OMN', 'MOZ', 'DOM', 'CHL', 'VEN', 'CMR', 'JAM', 'Other'] #
    # fmt: on

    @classmethod
    def preprocess(
        cls, df: pd.DataFrame, drop_competition_index=True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = super().preprocess(df, drop_competition_index)

        df['hotel'] = df['hotel'].map({'Resort Hotel': 0, 'City Hotel': 1})
        df['meal'] = df['meal'].map(cls.meal_mapping)
        df['customer_type'] = df['customer_type'].map(cls.customer_type)

        if 'country' in df.columns:
            df['country'] = df['country'].apply(
                lambda x: x if x in cls.countries else 'Other'
            )
            df['country'] = df['country'].map(
                {country: i for i, country in enumerate(cls.countries)}
            )

        df['market_segment'] = df['market_segment'].map(cls.market_segment)
        df['distribution_channel'] = df['distribution_channel'].map(
            cls.distribution_channel
        )
        df['deposit_type'] = df['deposit_type'].map(cls.deposit_type)
        X = df.drop(columns=['is_canceled'])
        y = df['is_canceled']
        return X, y


DESCRIPTION = """
            Each observation represents a hotel booking between the 1st of July 2015 and 31st of August 2017, including booking that effectively arrived and booking that were canceled.
            The dataset has already been cleaned with the following operations:
            df = df.loc[~((df["adults"] + df["children"] + df["babies"]) == 0)]
            df = df.loc[~((df["lead_time"] == 0) & (df["is_canceled"] == 1))]
            df = df.loc[~((df["lead_time"] == 0) & (df["booking_changes"] > 0))]

            hotel
            The datasets contains the booking information of two hotel. One of the hotels is a resort hotel and the other is a city hotel.

            is_canceled
            Value indicating if the booking was canceled (1) or not (0).

            lead_time
            Number of days that elapsed between the entering date of the booking into the PMS and the arrival date.

            arrival_date_year
            Year of arrival date

            arrival_date_month
            Month of arrival date with 12 categories: “January” to “December”

            arrival_date_week_number
            Week number of the arrival date

            arrival_date_day_of_month
            Day of the month of the arrival date

            stays_in_weekend_nights
            Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel

            stays_in_week_nights
            Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel BO and BL/Calculated by counting the number of week nights

            adults
            Number of adults

            children
            Number of children

            babies
            Number of babies

            meal
            BB – Bed & Breakfast

            country
            Country of origin.

            market_segment
            Market segment designation. In categories, the term “TA” means “Travel Agents” and “TO” means “Tour Operators”

            distribution_channel
            Booking distribution channel. The term “TA” means “Travel Agents” and “TO” means “Tour Operators”

            is_repeated_guest
            Value indicating if the booking name was from a repeated guest (1) or not (0)

            previous_cancellations
            Number of previous bookings that were cancelled by the customer prior to the current booking

            previous_bookings_not_canceled
            Number of previous bookings not cancelled by the customer prior to the current booking

            booking_changes
            Number of changes/amendments made to the booking from the moment the booking was entered on the PMS until the moment of check-in or cancellation

            deposit_type
            No Deposit – no deposit was made; Non Refund – a deposit was made in the value of the total stay cost; Refundable – a deposit was made with a value under the total cost of stay.

            agent
            ID of the travel agency that made the booking

            days_in_waiting_list
            Number of days the booking was in the waiting list before it was confirmed to the customer

            customer_type
            Group – when the booking is associated to a group; Transient – when the booking is not part of a group or contract, and is not associated to other transient booking; Transient-party – when the booking is transient, but is associated to at least other transient booking

            adr
            Average Daily Rate (Calculated by dividing the sum of all lodging transactions by the total number of staying nights)

            required_car_parking_spaces
            Number of car parking spaces required by the customer

            total_of_special_requests
            Number of special requests made by the customer (e.g. twin bed or high floor)

            name
            Name of the Guest (Not Real)

            email
            Email (Not Real)

            phone-number
            Phone number (not real)

            credit_card
            Credit Card Number (not Real)
            """


def get_kaggle_dataset():
    """
    Download the dataset from Kaggle and extract it and save it in the current directory.
    """

    dataset = 'mojtaba142/hotel-booking'

    custom_path = './'
    os.environ['KAGGLEHUB_CACHE'] = custom_path

    path = kagglehub.dataset_download(dataset, force_download=False)
    print(f'Dataset downloaded to: {path}')

    return path


def main():
    # Set seed for reproducibility
    np.random.seed(42)
    metadata = {}

    # Set Path() to the current directory which contains this script and the dataset
    DATA_PATH = Path(__file__).parent
    DETECTIVE_PATH = DATA_PATH / 'detective'
    DETECTIVE_PATH.mkdir(parents=True, exist_ok=True)
    path = get_kaggle_dataset()

    # Load the dataset and take a subset of it
    df = read_dataset(f'{path}/hotel_booking.csv')

    df['arrival_date_month'] = df['arrival_date_month'].map(
        {
            'January': 1,
            'February': 2,
            'March': 3,
            'April': 4,
            'May': 5,
            'June': 6,
            'July': 7,
            'August': 8,
            'September': 9,
            'October': 10,
            'November': 11,
            'December': 12,
        }
    )
    df = df.drop(
        columns=[
            'reserved_room_type',
            'assigned_room_type',
            'reservation_status_date',
            'company',
            'reservation_status',  # Delete this column as it is a leakage
        ],
        inplace=False,
    )

    df = df.loc[~((df['adults'] + df['children'] + df['babies']) == 0)]
    df = df.loc[~((df['lead_time'] == 0) & (df['is_canceled'] == 1))]
    df = df.loc[~((df['lead_time'] == 0) & (df['booking_changes'] > 0))]

    # Sort by month and year
    df_raw = df.sort_values(by=['arrival_date_year', 'arrival_date_month'])
    CUTOFF = 100000
    df_test = df_raw.iloc[-CUTOFF:]

    df = df_raw.iloc[:CUTOFF]
    df = df.reset_index(drop=True)

    df_test = df_test.reset_index(drop=True)
    df = df.reset_index(drop=True)

    # Add a new column to keep_track of the errors
    df['_competition_index'] = [x for x in range(df.shape[0])]
    df_test.to_csv(DETECTIVE_PATH / 'test.csv', index=False)
    corrupted_indices = dict()

    # Replace all nans with PRT
    metric_test_cleaned, _ = Pipeline.run_evaluation(df, df_test)
    metadata['clean_df_shape'] = df.shape
    metadata['clean_performance_test'] = metric_test_cleaned['F1_score']

    # Corruption 1: Add a systematic bias to lead_time in 2016
    series = df['arrival_date_year'] == 2016
    indices = df.loc[series].index
    values_before = df.loc[indices, 'lead_time'].values
    df.loc[indices, 'lead_time'] = df.loc[indices, 'lead_time'] + 10
    values_after = df.loc[indices, 'lead_time'].values
    corrupted_indices[0] = compute_corrupted_indices(
        series, 'lead_time', df, values_before, values_after
    )
    to_be_saved = inspect.getsource(Pipeline)

    metadata['train'] = df.shape
    metadata['test'] = df_test.shape

    df_corrupted = df.copy()
    metric_test_cleaned, _ = Pipeline.run_evaluation(df, df_test)
    metadata['df_shape'] = df_corrupted.shape
    metadata['performance_test'] = metric_test_cleaned['F1_score']
    metadata['pipeline_code'] = to_be_saved
    metadata['hints'] = [
        'Errors are in the lead_time, deposit and country columns, there are no errors in any entries from 2015.',
        'Errors are here: There is a systematic bias in the lead_time of 2016, the deposit with distribution_channel TA/TO looks wrong in 2017 and often when people arrive from PRT, the country is not recorded.',
        'Errors are here: There is a systematic bias in the lead_time of 2016 (+10 days), the deposit with distribution_channel TA/TO has been changed to Non Refund in 2017 and in case of NaN except for 2015, the country is set to PRT.',
    ]
    metadata['corrupted_indices'] = corrupted_indices
    metadata['dataset_description'] = DESCRIPTION
    metadata['target_column'] = 'is_canceled'
    df.to_csv(DETECTIVE_PATH / 'train_corrupted.csv', index=False)
    to_be_saved = asdict(CorruptedDatasetJson(**metadata))
    run_sanity_check_print(metadata)
    with open(DETECTIVE_PATH / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)


if __name__ == '__main__':
    main()
