"""
Feature engineering module for the TravelTide project.

This module defines a :class:`FeatureEngineer` class that encapsulates common
feature engineering steps for user-level aggregations. It exposes a single
public entrypoint via the :meth:`FeatureEngineer.transform` method which returns
a :class:`pandas.DataFrame` of aggregated features keyed by ``user_key``.

Additional convenience wrappers (`engineer_features`, `engineer_features2`,
`engineer_features_combined` and `engineer_features_optimized`) are provided
for backward compatibility and delegate to the same underlying implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

class FeatureEngineer:
    """
    Encapsulates feature engineering logic for a user-level dataset.

    Parameters
    ----------
    user_key : str, optional
        Column name that uniquely identifies a user in ``df``. Defaults to
        ``"user_id"``.
    """

    def __init__(self, user_key: str = "user_id") -> None:
        self.user_key = user_key

    def _compute_age(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series of ages computed from the ``birthdate`` column.

        If ``birthdate`` is not present, an empty series is returned.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        pandas.Series
        """
        if 'birthdate' in df.columns:
            ages = (datetime.now() - pd.to_datetime(df['birthdate'])).dt.days // 365
            return ages
        # return an empty series with the same index but no data
        return pd.Series(index=df.index, dtype=float)

    def _compute_session_duration(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series of session durations in minutes.

        If either ``session_start`` or ``session_end`` is missing, a series
        filled with zeros is returned.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        pandas.Series
        """
        if "session_start" in df.columns and "session_end" in df.columns:
            start = pd.to_datetime(df['session_start'])
            end = pd.to_datetime(df['session_end'])
            return (end - start).dt.total_seconds() / 60
        return pd.Series(0, index=df.index, dtype=float)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute engineered features for the given DataFrame.

        The function returns a new DataFrame indexed by ``user_key``,
        containing aggregated features such as age, session duration,
        click statistics, booking metrics, discount ratios, and more.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw event-level data.

        Returns
        -------
        pandas.DataFrame
            Aggregated user-level features.
        """
        df = df.copy()

        # Add derived columns
        df['age'] = self._compute_age(df)
        df['session_duration'] = self._compute_session_duration(df)

        grouped = df.groupby(self.user_key)
        user_features = pd.DataFrame(index=grouped.groups.keys())
        user_features.index.name = self.user_key

        # Demographics
        if 'age' in df.columns:
            user_features['age'] = grouped['age'].last().fillna(0)
        if 'gender' in df.columns:
            user_features['gender'] = (
                grouped['gender'].first().map({'F': 0, 'M': 1}).fillna(0)
            )
        if 'married' in df.columns:
            user_features['married'] = grouped['married'].first().astype(int).fillna(0)
        if 'has_children' in df.columns:
            user_features['has_children'] = grouped['has_children'].first().astype(int).fillna(0)

        # Clicks
        if "page_clicks" in df.columns:
            user_features["avg_clicks"] = grouped["page_clicks"].mean().fillna(0)
            user_features["total_clicks"] = grouped["page_clicks"].sum().fillna(0)

        # Cancellations
        if "cancellation" in df.columns:
            user_features["total_cancellations"] = grouped["cancellation"].sum().fillna(0)

        # Session aggregates
        if "session_duration" in df.columns:
            user_features["avg_session_duration"] = grouped["session_duration"].mean().fillna(0)
            user_features["total_session_duration"] = grouped["session_duration"].sum().fillna(0)
            user_features["total_sessions"] = grouped.size()

        # Destinations and trip types
        if "destination" in df.columns:
            user_features["n_unique_destinations"] = grouped["destination"].nunique().fillna(0)
        if "trip_type" in df.columns:
            user_features["n_unique_trip_types"] = grouped["trip_type"].nunique().fillna(0)

        # Spending
        if "base_fare_usd" in df.columns:
            user_features["avg_flight_fare_usd"] = grouped["base_fare_usd"].mean().fillna(0)
            user_features["total_flight_fare_usd"] = grouped["base_fare_usd"].sum().fillna(0)
        if "hotel_total_spend_usd" in df.columns:
            user_features["avg_hotel_spend_usd"] = grouped["hotel_total_spend_usd"].mean().fillna(0)
            user_features["total_hotel_spend_usd"] = grouped["hotel_total_spend_usd"].sum().fillna(0)

        # Bookings
        if "flight_booked" in df.columns:
            user_features["total_flights_booked"] = grouped["flight_booked"].sum().fillna(0)
        if "hotel_booked" in df.columns:
            user_features["total_hotels_booked"] = grouped["hotel_booked"].sum().fillna(0)

        # Discounts
        if "flight_discount" in df.columns:
            user_features["total_flight_discount"] = grouped["flight_discount"].sum().fillna(0)
            user_features["avg_flight_discount"] = grouped["flight_discount"].mean().fillna(0)
        if "hotel_discount" in df.columns:
            user_features["total_hotel_discount"] = grouped["hotel_discount"].sum().fillna(0)
            user_features["avg_hotel_discount"] = grouped["hotel_discount"].mean().fillna(0)

        # Totals for ratio calculations
        total_flights = user_features.get("total_flights_booked", pd.Series(0, index=user_features.index))
        total_hotels = user_features.get("total_hotels_booked", pd.Series(0, index=user_features.index))
        total_bookings = total_flights.fillna(0) + total_hotels.fillna(0)

        # Cancellation rate
        if "total_cancellations" in user_features.columns:
            user_features["cancellation_rate"] = user_features["total_cancellations"] / (total_bookings + 1e-6)

        # Flight + Hotel combo ratio
        if {"flight_booked", "hotel_booked"}.issubset(df.columns):
            user_features["flight_hotel_combo_ratio"] = grouped.apply(
                lambda g: ((g["flight_booked"] > 0) & (g["hotel_booked"] > 0)).sum() / (len(g) + 1e-6)
            )

        # Hotel booking ratio
        if "hotel_booked" in df.columns:
            user_features["hotel_booking_ratio"] = user_features.get("total_hotels_booked", pd.Series(0, index=user_features.index)) / (total_bookings + 1e-6)

        # People counts (use seats, fallback to rooms)
        if "seats" in df.columns or "rooms" in df.columns:
            df["people_count"] = np.where(
                df.get("seats", 0) > 0,
                df.get("seats", 0),
                np.where(df.get("rooms", 0) > 0, df.get("rooms", 0), 0),
            )
            user_features["total_people"] = grouped["people_count"].sum().fillna(0)
            user_features["avg_people"] = grouped["people_count"].mean().fillna(0)

        # Countries
        if "country" in df.columns:
            user_features["n_unique_countries"] = grouped["country"].nunique().fillna(0)

        # Date-based features
        if "booking_date" in df.columns:
            first = grouped["booking_date"].min()
            last = grouped["booking_date"].max()
            user_features["first_booking"] = first
            user_features["last_booking"] = last
            user_features["days_active"] = (last - first).dt.days
            if "total_flights_booked" in user_features.columns:
                user_features["bookings_per_month"] = user_features["total_flights_booked"] / (user_features["days_active"] / 30.44 + 1)

        # Discount percentages relative to spending totals
        if {"total_flight_discount", "total_flight_fare_usd"}.issubset(user_features.columns):
            user_features["avg_flight_discount_percentage"] = user_features["total_flight_discount"] / (user_features["total_flight_fare_usd"] + 1e-6)
        if {"total_hotel_discount", "total_hotel_spend_usd"}.issubset(user_features.columns):
            user_features["avg_hotel_discount_percentage"] = user_features["total_hotel_discount"] / (user_features["total_hotel_spend_usd"] + 1e-6)

        # Average checked bags per flight booked
        if "checked_bags" in df.columns and "flight_booked" in df.columns:
            user_features["avg_checked_bags"] = grouped["checked_bags"].sum() / (total_flights + 1e-6)

        # Final cleanup
        return user_features.fillna(0)


def engineer_features(df: pd.DataFrame, user_key: str = "user_id") -> pd.DataFrame:
    """
    Backwards compatible wrapper around :class:`FeatureEngineer`.

    Parameters
    ----------
    df : pandas.DataFrame
        The raw input data.
    user_key : str, optional
        The user identifier column.

    Returns
    -------
    pandas.DataFrame
        Aggregated user-level features.
    """
    return FeatureEngineer(user_key=user_key).transform(df)


# Delegate legacy function names for compatibility
engineer_features2 = engineer_features
engineer_features_combined = engineer_features
engineer_features_optimized = engineer_features