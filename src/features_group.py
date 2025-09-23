from src.setup import get_base
import pandas as pd
from datetime import datetime

def get_features(df):

    df_features = pd.DataFrame()
    df_features['age'] = (datetime.now() - df['birthdate']).dt.days // 365
    df_features['session_duration_min'] = (df['session_end'] - df['session_start']).dt.total_seconds() / 60
    df_features['hotel_total_spend_usd'] = df['hotel_price_per_room_night_usd'] * df['nights'].fillna(0) * df['rooms'].fillna(0)







    return df_features


import pandas as pd

def engineer_features(df, user_key="user_id"):
    grouped = df.groupby(user_key)
    feats = {}
    if "session_start" in df.columns and "session_end" in df.columns:
        df["session_duration"] = (df["session_end"] - df["session_start"]).dt.total_seconds() / 60.0
    else:
        df["session_duration"] = None  # falls Spalten fehlen

    # --- Rabattnutzung ---
    if "flight_discount" in df.columns:
        feats["avg_flight_discount"] = grouped["flight_discount"].mean()
        feats["total_flight_discount"] = grouped["flight_discount"].sum()
    if "hotel_discount" in df.columns:
        feats["avg_hotel_discount"] = grouped["hotel_discount"].mean()
        feats["total_hotel_discount"] = grouped["hotel_discount"].sum()

    # --- Useraktivität ---
    feats["total_sessions"] = grouped.size()
    if "session_duration" in df.columns:
        feats["avg_session_duration"] = grouped["session_duration"].mean()
        feats["total_session_duration"] = grouped["session_duration"].sum()

    # --- Buchungsverhalten ---
    if "flight_booked" in df.columns:
        feats["total_flights_booked"] = grouped["flight_booked"].sum()
    if "hotel_booked" in df.columns:
        feats["total_hotels_booked"] = grouped["hotel_booked"].sum()

    # --- Finanzielles Potenzial ---
    if "base_fare_usd" in df.columns:
        feats["avg_flight_fare_usd"] = grouped["base_fare_usd"].mean()
        feats["total_flight_fare_usd"] = grouped["base_fare_usd"].sum()
    if "hotel_total_spend_usd" in df.columns:
        feats["avg_hotel_spend_usd"] = grouped["hotel_total_spend_usd"].mean()
        feats["total_hotel_spend_usd"] = grouped["hotel_total_spend_usd"].sum()

    # --- Demografie ---
    if "age" in df.columns:
        feats["avg_age"] = grouped["age"].mean()
    if "country" in df.columns:
        feats["n_unique_countries"] = grouped["country"].nunique()

    # --- Reisepräferenz ---
    if "destination" in df.columns:
        feats["n_unique_destinations"] = grouped["destination"].nunique()
    if "trip_type" in df.columns:
        feats["n_unique_trip_types"] = grouped["trip_type"].nunique()

    # --- Zeitliche Muster ---
    if "booking_date" in df.columns:
        df["booking_date"] = pd.to_datetime(df["booking_date"], errors="coerce")
        feats["first_booking"] = grouped["booking_date"].min()
        feats["last_booking"] = grouped["booking_date"].max()
        feats["days_active"] = (grouped["booking_date"].max() - grouped["booking_date"].min()).dt.days
        feats["bookings_per_month"] = grouped["booking_date"].apply(lambda x: x.dt.to_period("M").nunique())

    user_features = pd.DataFrame(feats)
    return user_features.fillna(0)