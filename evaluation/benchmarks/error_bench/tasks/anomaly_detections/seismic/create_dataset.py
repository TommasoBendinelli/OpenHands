import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy


def main():
    data_dir = 'evaluation/benchmarks/error_bench/tasks/anomaly_detections/seismic'
    # downloaded from https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff
    data_path = Path(data_dir) / 'seismic-bumps.arff'
    if not os.path.exists(data_path):
        print(
            'Please dwnload the dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff and put it to data/seismic'
        )
        raise ValueError('mulcross.arff is not found in {}'.format(data_path))
    data, meta = scipy.io.arff.loadarff(data_path)
    df = pd.DataFrame(data)

    column_replacement = {
        'seismic': 'result of shift seismic hazard assessment in the mine working obtained by the seismic method',
        'seismoacoustic': 'result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method',
        'shift': 'information about type of a shift',
        'genergy': 'seismic energy recorded within previous shift by the most active geophone (GMax) out of geophones monitoring the longwall',
        'gpuls': 'a number of pulses recorded within previous shift by GMax',
        'gdenergy': 'a deviation of energy recorded within previous shift by GMax from average energy recorded during eight previous shifts',
        'gdpuls': 'a deviation of a number of pulses recorded within previous shift by GMax from average number of pulses recorded during eight previous shifts',
        'ghazard': 'result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method based on registration coming from GMax only',
        'nbumps': 'the number of seismic bumps recorded within previous shift',
        'nbumps2': 'the number of seismic bumps (in energy range [10^2,10^3)) registered within previous shift',
        'nbumps3': 'the number of seismic bumps (in energy range [10^3,10^4)) registered within previous shift',
        'nbumps4': 'the number of seismic bumps (in energy range [10^4,10^5)) registered within previous shift',
        'nbumps5': 'the number of seismic bumps (in energy range [10^5,10^6)) registered within the last shift',
        'nbumps6': 'the number of seismic bumps (in energy range [10^6,10^7)) registered within previous shift',
        'nbumps7': 'the number of seismic bumps (in energy range [10^7,10^8)) registered within previous shift',
        'nbumps89': 'the number of seismic bumps (in energy range [10^8,10^10)) registered within previous shift',
        'energy': 'total energy of seismic bumps registered within previous shift',
        'maxenergy': 'the maximum energy of the seismic bumps registered within previous shift',
    }
    # take log on magnitude columns
    df['maxenergy'] = np.log(df['maxenergy'].replace(0, 1e-6))
    df['energy'] = np.log(df['energy'].replace(0, 1e-6))
    # Rename the columns
    df.rename(columns=column_replacement, inplace=True)

    # Replace categorical values in the columns
    df[
        'result of shift seismic hazard assessment in the mine working obtained by the seismic method'
    ] = df[
        'result of shift seismic hazard assessment in the mine working obtained by the seismic method'
    ].replace(
        {
            b'a': 'lack of hazard',
            b'b': 'low hazard',
            b'c': 'high hazard',
            b'd': 'danger state',
        }
    )
    df[
        'result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method'
    ] = df[
        'result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method'
    ].replace(
        {
            b'a': 'lack of hazard',
            b'b': 'low hazard',
            b'c': 'high hazard',
            b'd': 'danger state',
        }
    )
    df[
        'result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method based on registration coming from GMax only'
    ] = df[
        'result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method based on registration coming from GMax only'
    ].replace(
        {
            b'a': 'lack of hazard',
            b'b': 'low hazard',
            b'c': 'high hazard',
            b'd': 'danger state',
        }
    )
    df['information about type of a shift'] = df[
        'information about type of a shift'
    ].replace({'W': 'coal-getting', 'N': 'preparation shift'})

    df['class'] = df['class'].map({b'0': 0, b'1': 1})

    # Split the dataset into training and test set
    # 50% for training and 50% for test
    train_df = df.sample(frac=0.5, random_state=42)
    test_df = df.drop(train_df.index)
    test_df.to_csv(data_dir + '/test_gt.csv', index=False)
    # Remove the class column from the test set
    test_df.drop(columns=['class'], inplace=True)
    # Save the training and test set to csv files
    train_df.to_csv(data_dir + '/train.csv', index=False)
    test_df.to_csv(data_dir + '/test.csv', index=False)
    # Save the test set with the class column
    test_df.to_csv(data_dir + '/test_gt.csv', index=False)


if __name__ == '__main__':
    main()
