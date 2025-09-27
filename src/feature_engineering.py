import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd


def engineer_features(df, user_key="user_id"):

    grouped = df.groupby(user_key)
    feats = {}

    feats['age'] = (datetime.now() - df['birthdate']).dt.days // 365

    if "page_clicks" in df.columns:
        feats["avg_clicks"] = grouped["page_clicks"].mean()

    feats["total_cancellation"] = grouped["cancellation"].sum()

    # Gender F = 0, M = 1
    feats["gender"] = df["gender"].map({"F": 0, "M": 1})

    # Married Yes = 1, NO = 2
    feats["married"] = df["married"].astype(int)

    # Children Yes = 1, NO = 2
    df["has_children"] = df["has_children"].astype(int)

    if "session_start" in df.columns and "session_end" in df.columns:
        feats['session_duration_min'] = (df['session_end'] - df['session_start']).dt.total_seconds() / 60
    else:
        df["session_duration"] = None

    # --- Discount ---
    if "flight_discount" in df.columns:
        feats["avg_flight_discount"] = grouped["flight_discount"].mean()
        feats["total_flight_discount"] = grouped["flight_discount"].sum()
    if "hotel_discount" in df.columns:
        feats["avg_hotel_discount"] = grouped["hotel_discount"].mean()
        feats["total_hotel_discount"] = grouped["hotel_discount"].sum()

    # --- Sessions ---
    feats["total_sessions"] = grouped.size()
    if "session_duration" in df.columns:
        feats["avg_session_duration"] = grouped["session_duration"].mean()
        feats["total_session_duration"] = grouped["session_duration"].sum()

    # --- Bookings ---
    if "flight_booked" in df.columns:
        feats["total_flights_booked"] = grouped["flight_booked"].sum()
    if "hotel_booked" in df.columns:
        feats["total_hotels_booked"] = grouped["hotel_booked"].sum()

    # --- Financel ---
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



def build_pca_pipeline(n_components=2, random_state=42):
    """
    Pipeline: Skalierung + PCA (nur für Dimensionality Reduction)
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=random_state)),
    ])
    return pipeline


def build_cluster_pipeline(n_clusters=2, n_components=2, random_state=42):
    """
    Pipeline: Skalierung + PCA + KMeans (für Cluster)
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=random_state)),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state))
    ])
    return pipeline


def plot_pca_clusters(pca_data, labels, title="PCA Cluster Plot"):
    """
    Erstellt einen schönen 2D-Scatterplot der PCA-Daten mit Clustern.
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        pca_data[:, 0], pca_data[:, 1],
        c=labels, cmap="viridis", alpha=0.7, s=50, edgecolor="k"
    )
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()