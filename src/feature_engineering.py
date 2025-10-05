"""
feature_engineering.py
~~~~~~~~~~~~~~~~~~~~~~

This module implements the feature engineering pipeline used to
aggregate the raw session‑level TravelTide data into a user‑level
feature table. The goal of this pipeline is to extract meaningful
behavioural and demographic attributes that can be passed to
unsupervised learning algorithms for segmentation.

The ``engineer_features`` function returns a DataFrame indexed by
``user_id`` where each row represents a unique traveller. Numeric
attributes are aggregated using sums and means, and categorical
attributes are encoded into numerical or low‑cardinality categorical
variables.

Example
-------
>>> from traveltide_ai.src.data_loader import load_raw_data
>>> from traveltide_ai.src.feature_engineering import engineer_features
>>> df_raw = load_raw_data('data/base-data.csv')
>>> df_features = engineer_features(df_raw)
>>> df_features.head()

"""

from __future__ import annotations

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from .utils import calculate_age, bin_categories, parse_datetime


def _encode_gender(series: pd.Series) -> pd.Series:
    """Encode the gender column into numeric codes.

    Parameters
    ----------
    series : pd.Series
        The gender column containing strings like ``'male'`` and
        ``'female'``. Missing or unknown values are mapped to 2.

    Returns
    -------
    pd.Series
        A series of integer codes: 0 for male, 1 for female, 2 for
        unknown/other.
    """
    mapping = {
        'male': 0,
        'Female': 1,
        'female': 1,
        'Male': 0,
        'other': 2,
        'Other': 2,
        'non-binary': 2,
        'Non-binary': 2,
    }
    def map_gender(x: Any) -> int:
        if pd.isna(x):
            return 2
        return mapping.get(str(x).strip(), 2)
    return series.apply(map_gender)


def engineer_features(raw_df: pd.DataFrame, reference_year: int = 2025) -> pd.DataFrame:
    """Aggregate raw session‑level data into user‑level features.

    This function performs the following steps:

    1. Normalises boolean fields to numeric (0/1).
    2. Fills missing numeric fields with sensible defaults (0 for counts).
    3. Calculates demographic attributes such as age, gender encoding,
       marital status and parenthood.
    4. Aggregates per‑user statistics like total sessions, number of
       bookings, cancellations, discount usage and spend.
    5. Calculates derived metrics (e.g. cancellation rate).

    Parameters
    ----------
    raw_df : pd.DataFrame
        The raw TravelTide session‑level dataset.
    reference_year : int, optional
        The year used to calculate ages from birth dates. Defaults to
        2025 (assumed current year for this project).

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by ``user_id`` containing engineered
        features ready for clustering.
    """
    df = raw_df.copy()

    # Ensure columns exist to avoid KeyErrors when modifying
    required_cols = [
        'flight_booked', 'hotel_booked', 'cancellation', 'return_flight_booked',
        'page_clicks', 'flight_discount_amount', 'hotel_discount_amount',
        'base_fare_usd', 'hotel_price_per_room_night_usd', 'nights', 'seats',
        'checked_bags'
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Convert boolean-like columns to numeric (0/1)
    bool_cols = ['flight_booked', 'hotel_booked', 'cancellation', 'return_flight_booked']
    for col in bool_cols:
        df[col] = df[col].fillna(0).astype(int)

    # Numeric fields: fill missing with 0 for aggregations
    numeric_fields_zero = ['page_clicks', 'flight_discount_amount', 'hotel_discount_amount',
                           'base_fare_usd', 'hotel_price_per_room_night_usd', 'nights', 'seats',
                           'checked_bags']
    for col in numeric_fields_zero:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Demographic features
    # Calculate age from birthdate
    df['age'] = df['birthdate'].apply(lambda x: calculate_age(x, reference_year))
    # Encode gender to numeric codes
    df['gender_code'] = _encode_gender(df.get('gender', pd.Series(dtype=object)))
    # Married and has_children as numeric
    df['married_flag'] = df.get('married', 0).fillna(0).astype(int)
    df['has_children_flag'] = df.get('has_children', 0).fillna(0).astype(int)
    # Sign up year
    df['sign_up_year'] = df['sign_up_date'].apply(
        lambda x: parse_datetime(x).year if not pd.isna(x) and parse_datetime(x) is not None else np.nan
    )

    # Bin home_country to reduce cardinality
    if 'home_country' in df.columns:
        df['home_country_bin'] = bin_categories(df['home_country'], top_n=5)
    else:
        df['home_country_bin'] = None

    # Aggregations per user
    agg_dict: Dict[str, Any] = {
        'session_id': 'nunique',
        'flight_booked': 'sum',
        'hotel_booked': 'sum',
        'cancellation': 'sum',
        'return_flight_booked': 'sum',
        'page_clicks': 'sum',
        'flight_discount_amount': ['sum', 'mean'],
        'hotel_discount_amount': ['sum', 'mean'],
        'base_fare_usd': ['sum', 'mean'],
        'hotel_price_per_room_night_usd': ['sum', 'mean'],
        'nights': 'sum',
        'seats': 'mean',
        'checked_bags': 'mean',
        'age': 'mean',
        'gender_code': 'mean',
        'married_flag': 'max',
        'has_children_flag': 'max',
        'sign_up_year': 'min',
        'home_country_bin': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None,
    }
    grouped = df.groupby('user_id').agg(agg_dict)

    # Flatten MultiIndex columns resulting from multi‑aggregations
    grouped.columns = ['_'.join([c for c in col if c]) if isinstance(col, tuple) else col for col in grouped.columns.values]

    grouped = grouped.rename(columns={
        'session_id_nunique': 'total_sessions',
        'flight_booked_sum': 'flights_booked_count',
        'hotel_booked_sum': 'hotels_booked_count',
        'cancellation_sum': 'cancellations_count',
        'return_flight_booked_sum': 'returns_booked_count',
        'page_clicks_sum': 'total_page_clicks',
        'flight_discount_amount_sum': 'flight_discount_sum',
        'flight_discount_amount_mean': 'flight_discount_avg',
        'hotel_discount_amount_sum': 'hotel_discount_sum',
        'hotel_discount_amount_mean': 'hotel_discount_avg',
        'base_fare_usd_sum': 'base_fare_sum',
        'base_fare_usd_mean': 'base_fare_avg',
        'hotel_price_per_room_night_usd_sum': 'hotel_price_sum',
        'hotel_price_per_room_night_usd_mean': 'hotel_price_avg',
        'nights_sum': 'total_nights',
        'seats_mean': 'avg_seats',
        'checked_bags_mean': 'avg_checked_bags',
        'age_mean': 'age_mean',
        'gender_code_mean': 'gender_code_avg',
        'married_flag_max': 'is_married',
        'has_children_flag_max': 'has_children',
        'sign_up_year_min': 'sign_up_year',
        'home_country_bin_<lambda>': 'home_country_mode'
    })

    # Derived metrics
    grouped['cancellation_rate'] = grouped['cancellations_count'] / grouped[['flights_booked_count', 'hotels_booked_count']].sum(axis=1).replace({0: np.nan})
    grouped['booking_count'] = grouped['flights_booked_count'] + grouped['hotels_booked_count']
    # Ratio of discount usage to bookings
    grouped['discount_usage_rate'] = (grouped['flight_discount_sum'] + grouped['hotel_discount_sum']) / grouped['booking_count'].replace({0: np.nan})
    # Average spend per booking
    grouped['avg_spend_per_booking'] = (grouped['base_fare_sum'] + grouped['hotel_price_sum']) / grouped['booking_count'].replace({0: np.nan})
    # Fill NaN resulting from division by zero with 0
    grouped = grouped.fillna(0)

    return grouped
