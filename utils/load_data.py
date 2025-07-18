import pandas as pd
import numpy as np
from typing import Tuple


def load_data(type: str = 'centers', test_type='uniform') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load RSSI and position data for training and testing.

    Depending on the specified type, this function loads either the 'centers' or 'samples'
    dataset. Each dataset includes RSSI measurements and corresponding position labels
    for both training and test sets.

    Args:
        type (str, optional): Dataset type to load.
            - 'centers': Loads the centers dataset.
            - 'samples': Loads the samples dataset.
            - 'trajectory': Loads the trajectory dataset.
            Defaults to 'centers'.
        test_type (str, optional): Test dataset type to load.
            - 'uniform': Loads the uniform dataset.
            - 'trajectory': Loads the trajectory dataset.
            Defaults to 'uniform'.

    Returns:
        Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
            A tuple containing:
                - df_X_train: RSSI measurements for training.
                - df_y_train: Position and cell IDs for training.
                - df_X_test: RSSI measurements for testing.
                - df_y_test: Position and cell IDs for testing.

    Raises:
        ValueError: If an unsupported type is provided.
    """
    if type == 'centers':
        df_X_train = pd.read_csv('./data/centers_rssi.csv', index_col=0)
        df_y_train = pd.read_csv('./data/centers_pos_cell.csv', index_col=0)
    elif type == 'samples':
        df_X_train = pd.read_csv('./data/samples_rssi.csv', index_col=0)
        df_y_train = pd.read_csv('./data/samples_pos_cell.csv', index_col=0)
    elif type == 'trajectory':
        df_X_train = pd.read_csv('./data/train_traj_rssi.csv', index_col=0)
        df_y_train = pd.read_csv('./data/train_traj_pos_cell.csv', index_col=0)
    else:
        raise ValueError

    if test_type == 'uniform':
        df_X_test = pd.read_csv('./data/tests_rssi.csv', index_col=0)
        df_y_test = pd.read_csv('./data/tests_pos_cell.csv', index_col=0)
    elif test_type == 'trajectory':
        df_X_test = pd.read_csv('./data/tests_traj_rssi.csv', index_col=0)
        df_y_test = pd.read_csv('./data/tests_traj_pos_cell.csv', index_col=0)
    else:
        raise ValueError

    return df_X_train, df_y_train, df_X_test, df_y_test


if __name__ == '__main__':
    print(load_data())
