"""Read data."""
import sqlite3

import pandas as pd
import numpy as np


# The dataset has unphysical values to mark data as missing.  For any
# given row in the dataset that we have, "mask" the value based on the
# unphysical values that OMNI uses to mark data as missing.
_MASK_FACTORY = {}

for vec in ('Bx', 'By', 'Bz', 'magB'):
    _MASK_FACTORY[vec] = lambda df, v=vec: df[v] < 500

for vec in ('Vx', 'Vy', 'Vz', 'magV'):
    _MASK_FACTORY[vec] = lambda df, v=vec: df[v] < 5000

_MASK_FACTORY['Density'] = lambda df: df['Density'] < 999.98
_MASK_FACTORY['Temperature'] = lambda df: df['Temperature'] < 9.98e6
_MASK_FACTORY['SymH'] = lambda df: (df['SymH'] < 600) & (df['SymH'] > -600)
_MASK_FACTORY['DynamicPressure'] = lambda df: (df['DynamicPressure'] < 98)


def _get_valid_data_masks(dataframe):
    """Get the masks that indicate that a given value is valid.

    Returns:
        pandas.DataFrame
    """
    masks = {
        col: _MASK_FACTORY[col](dataframe)
        for col in dataframe.keys()
        if col in _MASK_FACTORY}
    return pd.DataFrame(masks, index=dataframe.index)


def _replace_missing_with_nan(dataframe):
    """Replace missing values with NaN."""
    masks = _get_valid_data_masks(dataframe)
    for col in dataframe.keys():
        try:
            mask = masks[col].values
            dataframe[col][~mask] = np.nan
        except KeyError:
            pass


def read_omni_db(features, db_name='omni.db'):
    """Read the given features from the Omni Database.

    Available features are:

    Bx,
    By,
    Bz,
    magB,
    Vx,
    Vy,
    Vz,
    magV,
    Density,
    Temperature,
    DynamicPressure,
    SymH

    Args:
        features (Iterable): An iterable of features to read.
        db_name (string): Name of a file that holds the omni data.
    Returns:
        pandas.DataFrame
    """

    # Yes, this is liable for SQL injection.  Do not use in production :-)
    features = set(features)
    features.add('datetime')

    query = 'SELECT %s FROM OmniData ORDER BY datetime;' % (', '.join(features))

    database = sqlite3.connect(db_name)

    dataframe = pd.read_sql_query(
        query,
        database,
        parse_dates={'datetime': '%Y-%m-%d %H:%M:%S'}, index_col='datetime')
    _replace_missing_with_nan(dataframe)
    return dataframe


if __name__ == '__main__':
    # pylint: disable=invalid-name
    data = read_omni_db(['Bx'])
