import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build user-level features for clustering from session/trip data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or cleaned DataFrame with sessions, flights, and hotels.

    Returns
    -------
    features : pd.DataFrame
        Aggregated features at user_id level.
    """

    # --- Travel Behavior ---
    grouped = df.groupby("user_id")

    features = pd.DataFrame(index=grouped.size().index)

    features["num_trips"] = grouped["trip_id"].nunique()
    features["num_flights"] = grouped["flight_booked"].sum()
    features["num_hotels"] = grouped["hotel_booked"].sum()
    features["cancellation_rate"] = grouped["cancellation"].mean()

    # Avg km flown (exclude NaNs)
    if "flight_distance_km" in df.columns:
        features["avg_km_flown"] = grouped["flight_distance_km"].mean()

    # Trip duration
    if "check_in_time" in df.columns and "check_out_time" in df.columns:
        trip_durations = (df["check_out_time"] - df["check_in_time"]).dt.days
        features["avg_trip_duration_days"] = trip_durations.groupby(df["user_id"]).mean()

    # Multi-booking (flight+hotel in same trip)
    trip_bookings = df.groupby(["user_id", "trip_id"]).agg(
        flight=("flight_booked", "max"),
        hotel=("hotel_booked", "max")
    )
    multi = (trip_bookings["flight"] & trip_bookings["hotel"]).groupby("user_id").mean()
    features["multi_booking_rate"] = multi

    # --- Spending ---
    flight_spend = df["flight_discount_amount"].fillna(0)
    hotel_spend = (df["hotel_price_per_room_night_usd"].fillna(0) *
                   (df["check_out_time"] - df["check_in_time"]).dt.days.fillna(0))

    df["total_spend"] = flight_spend + hotel_spend

    features["total_flight_spend"] = df.groupby("user_id")["flight_discount_amount"].sum().fillna(0)
    features["total_hotel_spend"] = hotel_spend.groupby(df["user_id"]).sum().fillna(0)
    features["avg_spend_per_trip"] = features["total_flight_spend"].add(features["total_hotel_spend"]).div(features["num_trips"].replace(0, np.nan))

    features["hotel_vs_flight_ratio"] = features["total_hotel_spend"] / (features["total_flight_spend"] + features["total_hotel_spend"] + 1e-6)

    # --- Engagement ---
    session_duration = (df["session_end"] - df["session_start"]).dt.total_seconds() / 60
    features["num_sessions"] = grouped["session_id"].nunique()
    features["avg_session_duration"] = session_duration.groupby(df["user_id"]).mean()
    features["avg_clicks_per_session"] = grouped["page_clicks"].mean()
    features["click_to_booking_ratio"] = (df.groupby("user_id")["page_clicks"].sum() /
                                          (features["num_trips"] + 1e-6))

    # --- Booking Behavior ---
    if "departure_time" in df.columns and "session_start" in df.columns:
        lead_times = (df["departure_time"] - df["session_start"]).dt.days
        features["avg_lead_time_days"] = lead_times.groupby(df["user_id"]).mean()
        features["lead_time_variability"] = lead_times.groupby(df["user_id"]).std()

    # Recency of last trip
    if "check_out_time" in df.columns:
        features["last_trip_recency_days"] = (df["check_out_time"].max() - df.groupby("user_id")["check_out_time"].max()).dt.days

    # --- Demographics ---
    if "birthdate" in df.columns:
        age = (pd.Timestamp("today") - df["birthdate"]).dt.days / 365.25
        features["age"] = age.groupby(df["user_id"]).mean()
    if "married" in df.columns:
        features["married"] = grouped["married"].max()
    if "has_children" in df.columns:
        features["has_children"] = grouped["has_children"].max()

    # --- Discount / Loyalty ---
    features["flight_discount_usage_rate"] = grouped["flight_discount"].mean()
    features["hotel_discount_usage_rate"] = grouped["hotel_discount"].mean()
    features["avg_discount_amount"] = (df["flight_discount_amount"].fillna(0) + df["hotel_discount_amount"].fillna(0)).groupby(df["user_id"]).mean()

    # Reset index for clean merge
    features = features.reset_index()

    return features