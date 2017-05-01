"""Facilities for preprocessing data."""

import numpy as np


def _split_data_into_vectors(df_x, df_y, n_behind, n_ahead):
    assert len(df_x) == len(df_y)
    n_samples = len(df_x) - (n_behind + n_ahead)
    data_x = df_x.values
    data_y = df_y.values

    x_vecs = [np.expand_dims(np.atleast_2d(data_x[i:n_behind + i, :]), axis=0)
              for i in xrange(n_samples)]
    y_vecs = [np.atleast_2d(data_y[i + n_behind: i + n_behind + n_ahead, 0])
              for i in xrange(n_samples)]
    x_out = np.concatenate(x_vecs, axis=0) if x_vecs else np.array([])
    y_out = np.concatenate(y_vecs, axis=0) if y_vecs else np.array([])
    return x_out, y_out


def train_test_split(  # pylint: disable=too-many-arguments
        omni_data,
        x_fields,
        y_fields,
        percent=.70,
        n_points_behind=15,
        n_points_ahead=3):
    """Split the data into vectors for training and testing.

    Args:
        omni_data (pd.DataFrame): A dataframe that holds the data as a
            timeseries.  The dataframe should be in the same format as
            returned by read_omni_data.py
        x_fields (list): A list of column names in ``omni_data`` that will be
            used to make the prediction.
        y_fields (list): A list of column names in ``omni_data`` that will be
            predicted.
        percent (float): A number between 0 and 1 that indicates how much of
            the data to use as training data (the rest will be reserved for
            testing).
        n_points_behind (int): A number of points to use as the input vector
            in x_fields.  For example, if you are using the 5 minute data and
            you want to use a running window of 2 hours of data as the input to
            the model, then this should be 12 * 2 = 24.
        n_points_ahead (int): A number of points to use as the desired
            prediction window.  e.g. if you are using the 5 minute data and
            you want to predict 1 hour into the future, use a value of
            12 * 1 = 12.
    Returns:
        tuple: Tuple with 4 values -- x_train, y_train, x_test, y_test
    """
    x_data = omni_data[x_fields]
    y_data = omni_data[y_fields]
    x_data_train = []
    y_data_train = []

    train_mask = np.zeros(len(omni_data), dtype=bool)
    train_mask[:int(percent * len(omni_data))] = True

    x_data_train, y_data_train = _split_data_into_vectors(
        x_data[train_mask],
        y_data[train_mask],
        n_points_behind,
        n_points_ahead)

    test_mask = ~train_mask
    x_data_test, y_data_test = _split_data_into_vectors(
        x_data[test_mask],
        y_data[test_mask],
        n_points_behind,
        n_points_ahead)

    return x_data_train, y_data_train, x_data_test, y_data_test
