"""
data_loader.py
~~~~~~~~~~~~~~~~

This module provides helper functions to load the raw TravelTide dataset
from disk. Keeping the data loading logic in a dedicated module makes
it easier to maintain and extend as the project grows. The default
`load_raw_data` function assumes a CSV file and returns a pandas DataFrame.

Functions
---------
load_raw_data(file_path: str) -> pd.DataFrame
    Load the raw data from a CSV file into a pandas DataFrame.
"""

from __future__ import annotations

import pandas as pd

def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load the raw dataset from a CSV file.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the raw session‑level data.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the unprocessed session‑level records.

    Notes
    -----
    This function uses ``pandas.read_csv`` to read the input file and
    infers the column types automatically. Errors during reading will
    propagate to the caller.
    """
    df = pd.read_csv(file_path)
    return df
