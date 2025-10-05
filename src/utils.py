"""
utils.py
~~~~~~~~

Helper functions for the TravelTide project. These utilities handle
common tasks such as calculating ages from birth dates, parsing date
strings into ``datetime`` objects and normalising categorical variables.

Functions
---------
parse_datetime(date_str: str) -> Optional[datetime]
    Parse an ISO‑formatted string into a ``datetime`` object.

calculate_age(birthdate: Union[str, datetime], reference_year: int) -> Optional[int]
    Compute the age in years given a birth date and a reference year.

bin_categories(series: pd.Series, top_n: int) -> pd.Series
    Collapse a high cardinality categorical column into the top ``n``
    most frequent categories and an ``"Other"`` bucket.
"""

from __future__ import annotations

from typing import Optional, Union
import pandas as pd
from datetime import datetime

def parse_datetime(date_str: Union[str, None]) -> Optional[datetime]:
    """Parse an ISO formatted date string into a datetime.

    Parameters
    ----------
    date_str : str | None
        The date string to parse. Expected formats include
        ``YYYY‑MM‑DD`` or ``YYYY‑MM‑DD HH:MM:SS``. If ``None`` or an
        empty string is provided the function returns ``None``.

    Returns
    -------
    datetime | None
        A ``datetime`` object if parsing was successful, otherwise
        ``None``.
    """
    if not date_str or pd.isna(date_str):
        return None
    try:
        return datetime.fromisoformat(date_str)
    except Exception:
        # Fallback to more forgiving parsing
        try:
            return pd.to_datetime(date_str, errors='coerce')
        except Exception:
            return None

def calculate_age(birthdate: Union[str, datetime, None], reference_year: int) -> Optional[int]:
    """Calculate the age in years given a birth date and reference year.

    Parameters
    ----------
    birthdate : str | datetime | None
        The birth date of the user. Accepts either a string in
        ISO format or a ``datetime`` object. If ``None`` the function
        returns ``None``.

    reference_year : int
        The year in which to calculate the age (e.g. the current year).

    Returns
    -------
    int | None
        Age in years if calculable, otherwise ``None``.
    """
    if birthdate is None or pd.isna(birthdate):
        return None
    if isinstance(birthdate, str):
        dt = parse_datetime(birthdate)
    else:
        dt = birthdate
    if dt is None:
        return None
    age = reference_year - dt.year
    # Prevent negative ages or unrealistic values
    if age < 0 or age > 120:
        return None
    return age

def bin_categories(series: pd.Series, top_n: int = 5) -> pd.Series:
    """Collapse a categorical column into the top N categories and 'Other'.

    Parameters
    ----------
    series : pd.Series
        The categorical column to reduce.
    top_n : int, optional
        Number of most frequent categories to preserve. All other
        categories will be labeled as ``'Other'``. Defaults to 5.

    Returns
    -------
    pd.Series
        A new Series where categories outside the top ``top_n`` are
        replaced with ``'Other'``. Missing values are propagated as
        ``NaN``.
    """
    value_counts = series.value_counts(dropna=True)
    top_categories = value_counts.nlargest(top_n).index
    def _label_cat(cat):
        if pd.isna(cat):
            return None
        return cat if cat in top_categories else 'Other'
    return series.apply(_label_cat)
