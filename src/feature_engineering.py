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