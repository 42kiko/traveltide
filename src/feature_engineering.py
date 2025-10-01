import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import silhouette_score
import seaborn as sns


def engineer_features(df, user_key="user_id"):

    df = df.copy()

    # Age
    if 'birthdate' in df.columns:
        df['age'] = (datetime.now() - pd.to_datetime(df['birthdate'])).dt.days // 365

    # Session-Duration
    if "session_start" in df.columns and "session_end" in df.columns:
        df['session_duration'] = (pd.to_datetime(df['session_end']) - pd.to_datetime(df['session_start'])).dt.total_seconds() / 60
    else:
        df['session_duration'] = 0

    # Groupby User
    grouped = df.groupby(user_key)

    # Datafrome by User
    user_features = pd.DataFrame(index=grouped.groups.keys())
    user_features.index.name = user_key

    # Demographics
    if 'age' in df.columns:
        # Age: take last entry
        user_features['age'] = grouped['age'].last()

    if 'gender' in df.columns:
        # Gender : take first entry
        user_features['gender'] = grouped['gender'].first()
        # Umwandlung in numerisch: F=0, M=1
        user_features['gender'] = user_features['gender'].map({'F':0, 'M':1})

    if 'married' in df.columns:
        user_features['married'] = grouped['married'].first().astype(int)

    if 'has_children' in df.columns:
        user_features['has_children'] = grouped['has_children'].first().astype(int)

    # Clicks
    if "page_clicks" in df.columns:
        user_features["avg_clicks"] = grouped["page_clicks"].mean()
        user_features["total_clicks"] = grouped["page_clicks"].sum()

    # Cancellations
    if "cancellation" in df.columns:
        user_features["total_cancellation"] = grouped["cancellation"].sum()

    # Session-Duration
    if "session_duration" in df.columns:
        user_features["avg_session_duration"] = grouped["session_duration"].mean()
        user_features["total_session_duration"] = grouped["session_duration"].sum()
        user_features["total_sessions"] = grouped.size()

    # Discounts
    if "flight_discount" in df.columns:
        user_features["avg_flight_discount"] = grouped["flight_discount"].mean()
        user_features["total_flight_discount"] = grouped["flight_discount"].sum()
    if "hotel_discount" in df.columns:
        user_features["avg_hotel_discount"] = grouped["hotel_discount"].mean()
        user_features["total_hotel_discount"] = grouped["hotel_discount"].sum()

    # Bookings
    if "flight_booked" in df.columns:
        user_features["total_flights_booked"] = grouped["flight_booked"].sum()
    if "hotel_booked" in df.columns:
        user_features["total_hotels_booked"] = grouped["hotel_booked"].sum()

    # Spending
    if "base_fare_usd" in df.columns:
        user_features["avg_flight_fare_usd"] = grouped["base_fare_usd"].mean()
        user_features["total_flight_fare_usd"] = grouped["base_fare_usd"].sum()
    if "hotel_total_spend_usd" in df.columns:
        user_features["avg_hotel_spend_usd"] = grouped["hotel_total_spend_usd"].mean()
        user_features["total_hotel_spend_usd"] = grouped["hotel_total_spend_usd"].sum()

    # Countries
    if "country" in df.columns:
        user_features["n_unique_countries"] = grouped["country"].nunique()

    # Destinations
    if "destination" in df.columns:
        user_features["n_unique_destinations"] = grouped["destination"].nunique()
    if "trip_type" in df.columns:
        user_features["n_unique_trip_types"] = grouped["trip_type"].nunique()

    # Dates
    if "booking_date" in df.columns:
        user_features["first_booking"] = grouped["booking_date"].min()
        user_features["last_booking"] = grouped["booking_date"].max()
        user_features["days_active"] = (user_features["last_booking"] - user_features["first_booking"]).dt.days

        user_features["bookings_per_month"] = user_features["total_flights_booked"] / (user_features["days_active"] / 30.44 + 1)  # 30.44 days in a month


    # Total people per booking
    df["people_count"] = np.where(
        df["seats"] > 0,
        df["seats"],
        np.where(df["rooms"] > 0, df["rooms"], 0)
    )

    # Aggregiert pro User
    user_features["total_people"] = grouped["people_count"].sum()
    user_features["avg_people"] = grouped["people_count"].mean()



    # Fill NAs
    user_features = user_features.fillna(0)

    return user_features



def engineer_features2(df, user_key="user_id"):

    df = df.copy()

    # Age
    if 'birthdate' in df.columns:
        df['age'] = (datetime.now() - pd.to_datetime(df['birthdate'])).dt.days // 365



    # Session-Duration
    if "session_start" in df.columns and "session_end" in df.columns:
        df['session_duration'] = (pd.to_datetime(df['session_end']) - pd.to_datetime(df['session_start'])).dt.total_seconds() / 60
    else:
        df['session_duration'] = 0

    # Groupby User
    grouped = df.groupby(user_key)

    # DataFrame by User
    user_features = pd.DataFrame(index=grouped.groups.keys())
    user_features.index.name = user_key

    # Basisdemografie
    if 'age' in df.columns:
        user_features['age'] = grouped['age'].last()
    if 'gender' in df.columns:
        user_features['gender'] = grouped['gender'].first().map({'F':0, 'M':1})
    if 'married' in df.columns:
        user_features['married'] = grouped['married'].first().astype(int)
    if 'has_children' in df.columns:
        user_features['has_children'] = grouped['has_children'].first().astype(int)

    # Destinations
    if "destination" in df.columns:
        user_features["n_unique_destinations"] = grouped["destination"].nunique()
    if "trip_type" in df.columns:
        user_features["n_unique_trip_types"] = grouped["trip_type"].nunique()

    # Spending
    if "base_fare_usd" in df.columns:
        user_features["avg_flight_fare_usd"] = grouped["base_fare_usd"].mean()
        user_features["total_flight_fare_usd"] = grouped["base_fare_usd"].sum()
    if "hotel_total_spend_usd" in df.columns:
        user_features["avg_hotel_spend_usd"] = grouped["hotel_total_spend_usd"].mean()
        user_features["total_hotel_spend_usd"] = grouped["hotel_total_spend_usd"].sum()

    # Buchungen
    if "flight_booked" in df.columns:
        user_features["total_flights_booked"] = grouped["flight_booked"].sum()
    if "hotel_booked" in df.columns:
        user_features["total_hotels_booked"] = grouped["hotel_booked"].sum()

    # Rabatte
    if "flight_discount" in df.columns and "base_fare_usd" in df.columns:
        user_features["total_flight_discount"] = grouped["flight_discount"].sum()
        user_features["avg_flight_discount"] = grouped["flight_discount"].mean()
        user_features["flight_discount_ratio"] = grouped.apply(
            lambda g: (g["flight_discount"] > 0).sum() / (g["flight_booked"].sum() + 1e-6)
        )
        user_features["avg_flight_discount_percentage"] = (
            user_features["total_flight_discount"] / (user_features["total_flight_fare_usd"] + 1e-6)
        )
    if "hotel_discount" in df.columns and "hotel_total_spend_usd" in df.columns:
        user_features["total_hotel_discount"] = grouped["hotel_discount"].sum()
        user_features["avg_hotel_discount"] = grouped["hotel_discount"].mean()
        user_features["hotel_discount_ratio"] = grouped.apply(
            lambda g: (g["hotel_discount"] > 0).sum() / (g["hotel_booked"].sum() + 1e-6)
        )

    # Checked Bags
    if "checked_bags" in df.columns and "flight_booked" in df.columns:
        user_features["avg_checked_bags"] = grouped["checked_bags"].sum() / (user_features["total_flights_booked"] + 1e-6)

    # Cancellation Rate
    if "cancellation" in df.columns:
        user_features["total_cancellations"] = grouped["cancellation"].sum()
        total_bookings = (
            user_features["total_flights_booked"].fillna(0) +
            user_features["total_hotels_booked"].fillna(0)
        )
        user_features["cancellation_rate"] = user_features["total_cancellations"] / (total_bookings + 1e-6)

    # Flight + Hotel Kombination
    if "flight_booked" in df.columns and "hotel_booked" in df.columns:
        user_features["flight_hotel_combo_ratio"] = grouped.apply(
            lambda g: ((g["flight_booked"] > 0) & (g["hotel_booked"] > 0)).sum() / (len(g) + 1e-6)
        )

    # Hotel focus
    if "hotel_booked" in df.columns:
        total_bookings = (
            user_features["total_flights_booked"].fillna(0) +
            user_features["total_hotels_booked"].fillna(0)
        )
        user_features["hotel_booking_ratio"] = user_features["total_hotels_booked"] / (total_bookings + 1e-6)

    # Fill NAs
    user_features = user_features.fillna(0)

    return user_features


def engineer_features_combined(df, user_key="user_id"):
    df = df.copy()

    # --- Age ---
    if 'birthdate' in df.columns:
        df['age'] = (datetime.now() - pd.to_datetime(df['birthdate'])).dt.days // 365

    # --- Session Duration ---
    if "session_start" in df.columns and "session_end" in df.columns:
        df['session_duration'] = (
            pd.to_datetime(df['session_end']) - pd.to_datetime(df['session_start'])
        ).dt.total_seconds() / 60
    else:
        df['session_duration'] = 0

    # --- Group by user ---
    grouped = df.groupby(user_key)
    user_features = pd.DataFrame(index=grouped.groups.keys())
    user_features.index.name = user_key

    # --- Demographics ---
    if 'age' in df.columns:
        user_features['age'] = grouped['age'].last()
    if 'gender' in df.columns:
        user_features['gender'] = grouped['gender'].first().map({'F': 0, 'M': 1})
    if 'married' in df.columns:
        user_features['married'] = grouped['married'].first().astype(int)
    if 'has_children' in df.columns:
        user_features['has_children'] = grouped['has_children'].first().astype(int)

    # --- Clicks ---
    if "page_clicks" in df.columns:
        user_features["avg_clicks"] = grouped["page_clicks"].mean()
        user_features["total_clicks"] = grouped["page_clicks"].sum()

    # --- Cancellations ---
    if "cancellation" in df.columns:
        user_features["total_cancellations"] = grouped["cancellation"].sum()

    # --- Session Aggregates ---
    if "session_duration" in df.columns:
        user_features["avg_session_duration"] = grouped["session_duration"].mean()
        user_features["total_session_duration"] = grouped["session_duration"].sum()
        user_features["total_sessions"] = grouped.size()

    # --- Destinations & Trips ---
    if "destination" in df.columns:
        user_features["n_unique_destinations"] = grouped["destination"].nunique()
    if "trip_type" in df.columns:
        user_features["n_unique_trip_types"] = grouped["trip_type"].nunique()

    # --- Spending ---
    if "base_fare_usd" in df.columns:
        user_features["avg_flight_fare_usd"] = grouped["base_fare_usd"].mean()
        user_features["total_flight_fare_usd"] = grouped["base_fare_usd"].sum()
    if "hotel_total_spend_usd" in df.columns:
        user_features["avg_hotel_spend_usd"] = grouped["hotel_total_spend_usd"].mean()
        user_features["total_hotel_spend_usd"] = grouped["hotel_total_spend_usd"].sum()

    # --- Bookings ---
    if "flight_booked" in df.columns:
        user_features["total_flights_booked"] = grouped["flight_booked"].sum()
    if "hotel_booked" in df.columns:
        user_features["total_hotels_booked"] = grouped["hotel_booked"].sum()

    # --- Discounts ---
    if "flight_discount" in df.columns:
        user_features["total_flight_discount"] = grouped["flight_discount"].sum()
        user_features["avg_flight_discount"] = grouped["flight_discount"].mean()
        if "flight_booked" in df.columns:
            user_features["flight_discount_ratio"] = grouped.apply(
                lambda g: (g["flight_discount"] > 0).sum() / (g["flight_booked"].sum() + 1e-6)
            )
        if "base_fare_usd" in df.columns:
            user_features["avg_flight_discount_percentage"] = (
                user_features["total_flight_discount"] / (user_features["total_flight_fare_usd"] + 1e-6)
            )
    if "hotel_discount" in df.columns:
        user_features["total_hotel_discount"] = grouped["hotel_discount"].sum()
        user_features["avg_hotel_discount"] = grouped["hotel_discount"].mean()
        if "hotel_booked" in df.columns:
            user_features["hotel_discount_ratio"] = grouped.apply(
                lambda g: (g["hotel_discount"] > 0).sum() / (g["hotel_booked"].sum() + 1e-6)
            )

    # --- Checked Bags ---
    if "checked_bags" in df.columns and "flight_booked" in df.columns:
        user_features["avg_checked_bags"] = grouped["checked_bags"].sum() / (user_features["total_flights_booked"] + 1e-6)

    # --- Cancellation Rate ---
    if "cancellation" in df.columns:
        total_bookings = (
            user_features["total_flights_booked"].fillna(0) +
            user_features["total_hotels_booked"].fillna(0)
        )
        user_features["cancellation_rate"] = user_features["total_cancellations"] / (total_bookings + 1e-6)

    # --- Flight + Hotel Combo ---
    if "flight_booked" in df.columns and "hotel_booked" in df.columns:
        user_features["flight_hotel_combo_ratio"] = grouped.apply(
            lambda g: ((g["flight_booked"] > 0) & (g["hotel_booked"] > 0)).sum() / (len(g) + 1e-6)
        )

    # --- Hotel Focus ---
    if "hotel_booked" in df.columns:
        total_bookings = (
            user_features["total_flights_booked"].fillna(0) +
            user_features["total_hotels_booked"].fillna(0)
        )
        user_features["hotel_booking_ratio"] = user_features["total_hotels_booked"] / (total_bookings + 1e-6)

    # --- Countries ---
    if "country" in df.columns:
        user_features["n_unique_countries"] = grouped["country"].nunique()

    # --- Dates ---
    if "booking_date" in df.columns:
        user_features["first_booking"] = grouped["booking_date"].min()
        user_features["last_booking"] = grouped["booking_date"].max()
        user_features["days_active"] = (user_features["last_booking"] - user_features["first_booking"]).dt.days
        if "flight_booked" in df.columns:
            user_features["bookings_per_month"] = user_features["total_flights_booked"] / (
                user_features["days_active"] / 30.44 + 1
            )

    # --- People per booking (Seats preferred, fallback Rooms) ---
    if "seats" in df.columns or "rooms" in df.columns:
        df["people_count"] = np.where(
            df.get("seats", 0) > 0,
            df.get("seats", 0),
            np.where(df.get("rooms", 0) > 0, df.get("rooms", 0), 0)
        )
        user_features["total_people"] = grouped["people_count"].sum()
        user_features["avg_people"] = grouped["people_count"].mean()

    # --- Fill NAs ---
    user_features = user_features.fillna(0)

    return user_features


def engineer_features_optimized(df, user_key="user_id"):
    """
    Optimized feature engineering for user data.

    This function takes a DataFrame of user data and returns a
    new DataFrame with the engineered features. The features
    are optimized for the engagement and value scores.

    Returns:
        A DataFrame with the engineered features.

    Examples:
        >>> engineer_features_optimized(user_data)
        DataFrame of engineered features
    """
    df = df.copy()

    # Age
    if 'birthdate' in df.columns:
        df['age'] = (datetime.now() - pd.to_datetime(df['birthdate'])).dt.days // 365

    # Session-Duration mit erweiterten Zeit-Features
    if "session_start" in df.columns and "session_end" in df.columns:
        df['session_start'] = pd.to_datetime(df['session_start'])
        df['session_end'] = pd.to_datetime(df['session_end'])

        df['session_duration'] = (df['session_end'] - df['session_start']).dt.total_seconds() / 60
        df['session_hour'] = df['session_start'].dt.hour
        df['session_day_of_week'] = df['session_start'].dt.dayofweek
        df['session_month'] = df['session_start'].dt.month
    else:
        df['session_duration'] = 0
        df['session_hour'] = 0
        df['session_day_of_week'] = 0
        df['session_month'] = 0

    # Groupby User
    grouped = df.groupby(user_key)

    # DataFrame by User
    user_features = pd.DataFrame(index=grouped.groups.keys())
    user_features.index.name = user_key

    # Demographics
    if 'age' in df.columns:
        user_features['age'] = grouped['age'].last()
    if 'gender' in df.columns:
        user_features['gender'] = grouped['gender'].first()
        user_features['gender'] = user_features['gender'].map({'F':0, 'M':1})
    if 'married' in df.columns:
        user_features['married'] = grouped['married'].first().astype(int)
    if 'has_children' in df.columns:
        user_features['has_children'] = grouped['has_children'].first().astype(int)

    # Clicks
    if "page_clicks" in df.columns:
        user_features["avg_clicks"] = grouped["page_clicks"].mean()
        user_features["total_clicks"] = grouped["page_clicks"].sum()
        user_features["clicks_per_session"] = user_features["total_clicks"] / grouped.size()

    # Cancellations
    if "cancellation" in df.columns:
        user_features["total_cancellations"] = grouped["cancellation"].sum()

    # Session-Duration mit erweiterten Metriken
    if "session_duration" in df.columns:
        user_features["avg_session_duration"] = grouped["session_duration"].mean()
        user_features["total_session_duration"] = grouped["session_duration"].sum()
        user_features["total_sessions"] = grouped.size()
        user_features["session_frequency"] = grouped.size() / 30  # Sessions pro Tag (angenommener Zeitraum)

    # Zeit-basierte Features
    if "session_hour" in df.columns:
        user_features["avg_session_hour"] = grouped["session_hour"].mean()
        user_features["session_hour_std"] = grouped["session_hour"].std()
    if "session_day_of_week" in df.columns:
        user_features["avg_session_day"] = grouped["session_day_of_week"].mean()
    if "session_month" in df.columns:
        user_features["active_months"] = grouped["session_month"].nunique()

    # Discounts mit erweiterten Metriken
    if "flight_discount" in df.columns:
        user_features["flight_discount_sessions"] = grouped["flight_discount"].sum()
        user_features["flight_discount_ratio"] = grouped["flight_discount"].mean()
    if "hotel_discount" in df.columns:
        user_features["hotel_discount_sessions"] = grouped["hotel_discount"].sum()
        user_features["hotel_discount_ratio"] = grouped["hotel_discount"].mean()

    # Discount Amounts
    if "flight_discount_amount" in df.columns:
        user_features["avg_flight_discount_amount"] = grouped["flight_discount_amount"].mean()
        user_features["total_flight_discount_amount"] = grouped["flight_discount_amount"].sum()
    if "hotel_discount_amount" in df.columns:
        user_features["avg_hotel_discount_amount"] = grouped["hotel_discount_amount"].mean()
        user_features["total_hotel_discount_amount"] = grouped["hotel_discount_amount"].sum()

    # Bookings mit Konversionsraten
    if "flight_booked" in df.columns:
        user_features["total_flights_booked"] = grouped["flight_booked"].sum()
        user_features["flight_conversion_rate"] = grouped["flight_booked"].mean()
    if "hotel_booked" in df.columns:
        user_features["total_hotels_booked"] = grouped["hotel_booked"].sum()
        user_features["hotel_conversion_rate"] = grouped["hotel_booked"].mean()

    # Spending mit erweiterten Metriken
    if "base_fare_usd" in df.columns:
        user_features["avg_flight_fare_usd"] = grouped["base_fare_usd"].mean()
        user_features["total_flight_fare_usd"] = grouped["base_fare_usd"].sum()
        user_features["max_flight_fare"] = grouped["base_fare_usd"].max()
    if "hotel_total_spend_usd" in df.columns:
        user_features["avg_hotel_spend_usd"] = grouped["hotel_total_spend_usd"].mean()
        user_features["total_hotel_spend_usd"] = grouped["hotel_total_spend_usd"].sum()
        user_features["max_hotel_spend"] = grouped["hotel_total_spend_usd"].max()

    # Countries & Destinations
    if "country" in df.columns:
        user_features["n_unique_countries"] = grouped["country"].nunique()
    if "destination" in df.columns:
        user_features["n_unique_destinations"] = grouped["destination"].nunique()
    if "trip_type" in df.columns:
        user_features["n_unique_trip_types"] = grouped["trip_type"].nunique()

    # Dates & Aktivität
    if "booking_date" in df.columns:
        user_features["first_booking"] = grouped["booking_date"].min()
        user_features["last_booking"] = grouped["booking_date"].max()
        user_features["days_active"] = (user_features["last_booking"] - user_features["first_booking"]).dt.days
        user_features["bookings_per_month"] = user_features["total_flights_booked"] / (user_features["days_active"] / 30.44 + 1)
        user_features["recency_days"] = (datetime.now() - user_features["last_booking"]).dt.days

    # People per booking
    df["people_count"] = np.where(
        df["seats"] > 0,
        df["seats"],
        np.where(df["rooms"] > 0, df["rooms"], 1)  # Mindestens 1 Person
    )
    user_features["total_people"] = grouped["people_count"].sum()
    user_features["avg_people"] = grouped["people_count"].mean()

    # Neue kombinierte Features
    user_features["total_bookings"] = user_features.get("total_flights_booked", 0) + user_features.get("total_hotels_booked", 0)
    user_features["total_spend"] = user_features.get("total_flight_fare_usd", 0) + user_features.get("total_hotel_spend_usd", 0)
    user_features["avg_spend_per_booking"] = user_features["total_spend"] / (user_features["total_bookings"] + 1)
    user_features["sessions_per_booking"] = user_features["total_sessions"] / (user_features["total_bookings"] + 1)

    # Engagement Score (kombiniertes Feature)
    user_features["engagement_score"] = (
        user_features["total_sessions"] * 0.3 +
        user_features["total_clicks"] * 0.2 +
        user_features["total_session_duration"] * 0.2 +
        user_features["total_bookings"] * 0.3
    )

    # Value Score (kombiniertes Feature)
    user_features["value_score"] = (
        user_features["total_spend"] * 0.6 +
        user_features["avg_spend_per_booking"] * 0.4
    )

    # Fill NAs
    user_features = user_features.fillna(0)

    # Unendliche Werte vermeiden
    user_features = user_features.replace([np.inf, -np.inf], 0)

    return user_features