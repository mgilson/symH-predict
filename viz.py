"""Utilities for visualizing stuff."""
import pandas as pd

import preprocessing

def plot_event_predictions(  # pylint: disable=too-many-arguments, too-many-locals
        model,
        data,
        start_time,
        stop_time,
        x_fields,
        y_fields,
        n_points_behind=15,
        n_points_ahead=3):
    """Show predictions along-side real data.

    Args:
        model (keras.Sequential|models.ModelWrapper): The trained model.
        data (pd.DataFrame): The dataframe with the actual data.
        start_time (pd.Timestamp): The time that the event starts.
        stop_time (pd.Timestamp): The time that the event ends.
        x_fields (list): List of column names used to train the model.
        y_fields (list): List of column names used to train the model.
        n_points_behind (int): input vector size.
        n_points_ahead (int): output vector size.
    """

    event = data[(data.index > start_time) & (data.index < stop_time)]

    x, _, _, _ = preprocessing.train_test_split(  # pylint: disable=invalid-name
        event,
        x_fields,
        y_fields,
        percent=1,
        n_points_behind=n_points_behind,
        n_points_ahead=n_points_ahead)

    times = event.index.get_values()
    n_samples = len(event) - (n_points_behind + n_points_ahead)
    times = [times[i + n_points_behind: i + n_points_behind + n_points_ahead]
             for i in xrange(n_samples)]
    results = model.predict(x)

    event[y_fields].plot()
    for time, result in zip(times, results):
        index = pd.DatetimeIndex(time)
        series = pd.Series(result, index=index)
        series.plot(c='green')
