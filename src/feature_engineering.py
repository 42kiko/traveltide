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
    # Zuerst einige Berechnungen auf dem originalen DataFrame, falls nötig
    df = df.copy()

    # Alter berechnen
    if 'birthdate' in df.columns:
        df['age'] = (datetime.now() - pd.to_datetime(df['birthdate'])).dt.days // 365

    # Session-Dauer, falls Session-Start und Ende vorhanden
    if "session_start" in df.columns and "session_end" in df.columns:
        df['session_duration'] = (pd.to_datetime(df['session_end']) - pd.to_datetime(df['session_start'])).dt.total_seconds() / 60
    else:
        df['session_duration'] = 0

    # Gruppierung nach User
    grouped = df.groupby(user_key)

    # Wir erstellen ein DataFrame für die User-Features
    user_features = pd.DataFrame(index=grouped.groups.keys())
    user_features.index.name = user_key

    # Demografische Merkmale (konstant pro User)
    if 'age' in df.columns:
        # Wir nehmen das letzte Alter des Users (könnte auch das erste sein, aber wenn es sich ändert, dann letztes)
        user_features['age'] = grouped['age'].last()

    if 'gender' in df.columns:
        # Gender: wir nehmen den ersten Eintrag pro User
        user_features['gender'] = grouped['gender'].first()
        # Umwandlung in numerisch: F=0, M=1
        user_features['gender'] = user_features['gender'].map({'F':0, 'M':1})

    if 'married' in df.columns:
        user_features['married'] = grouped['married'].first().astype(int)

    if 'has_children' in df.columns:
        user_features['has_children'] = grouped['has_children'].first().astype(int)

    # Klick-Verhalten
    if "page_clicks" in df.columns:
        user_features["avg_clicks"] = grouped["page_clicks"].mean()
        user_features["total_clicks"] = grouped["page_clicks"].sum()

    # Stornierungen
    if "cancellation" in df.columns:
        user_features["total_cancellation"] = grouped["cancellation"].sum()

    # Session-Dauer
    if "session_duration" in df.columns:
        user_features["avg_session_duration"] = grouped["session_duration"].mean()
        user_features["total_session_duration"] = grouped["session_duration"].sum()
        user_features["total_sessions"] = grouped.size()

    # Rabatte
    if "flight_discount" in df.columns:
        user_features["avg_flight_discount"] = grouped["flight_discount"].mean()
        user_features["total_flight_discount"] = grouped["flight_discount"].sum()
    if "hotel_discount" in df.columns:
        user_features["avg_hotel_discount"] = grouped["hotel_discount"].mean()
        user_features["total_hotel_discount"] = grouped["hotel_discount"].sum()

    # Buchungen
    if "flight_booked" in df.columns:
        user_features["total_flights_booked"] = grouped["flight_booked"].sum()
    if "hotel_booked" in df.columns:
        user_features["total_hotels_booked"] = grouped["hotel_booked"].sum()

    # Ausgaben
    if "base_fare_usd" in df.columns:
        user_features["avg_flight_fare_usd"] = grouped["base_fare_usd"].mean()
        user_features["total_flight_fare_usd"] = grouped["base_fare_usd"].sum()
    if "hotel_total_spend_usd" in df.columns:
        user_features["avg_hotel_spend_usd"] = grouped["hotel_total_spend_usd"].mean()
        user_features["total_hotel_spend_usd"] = grouped["hotel_total_spend_usd"].sum()

    # Demografie (andere)
    if "country" in df.columns:
        user_features["n_unique_countries"] = grouped["country"].nunique()

    # Reisepräferenz
    if "destination" in df.columns:
        user_features["n_unique_destinations"] = grouped["destination"].nunique()
    if "trip_type" in df.columns:
        user_features["n_unique_trip_types"] = grouped["trip_type"].nunique()

    # Zeitliche Muster
    if "booking_date" in df.columns:
        user_features["first_booking"] = grouped["booking_date"].min()
        user_features["last_booking"] = grouped["booking_date"].max()
        user_features["days_active"] = (user_features["last_booking"] - user_features["first_booking"]).dt.days
        # Buchungen pro Monat: Anzahl der Buchungen / Anzahl der Monate zwischen erster und letzter Buchung (in Monaten)
        # Achtung: Wenn nur eine Buchung, dann days_active=0, dann teilen durch 0 -> unendlich. Daher:
        user_features["bookings_per_month"] = user_features["total_flights_booked"] / (user_features["days_active"] / 30.44 + 1)  # Vermeidung von Division durch Null

    # Fehlende Werte: Wir füllen mit 0, aber Vorsicht bei kategorischen Merkmalen, die vielleicht nicht 0 sein sollten?
    user_features = user_features.fillna(0)

    return user_features