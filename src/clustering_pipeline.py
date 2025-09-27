"""
clustering_pipeline.py

Generic, reproducible pipeline to:
- build user-level features from session-level data
- run multiple clustering algorithms and compare them
- save results, metrics, and diagnostic plots

Usage (as script):
    python clustering_pipeline.py --input data/base-data.csv --output output/ --methods kmeans,gmm,agg,dbscan

Usage (from notebook):
    from clustering_pipeline import run_full_pipeline
    results = run_full_pipeline(input_path, output_dir)

Everything is intentionally generic and robust to missing columns.
"""

import os
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score


# -----------------------------
# Feature builder (user-level)
# -----------------------------

def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate session/trip-level DataFrame to a user-level features DataFrame.
    This function is defensive: it checks for column existence and handles NaNs.
    """
    df = df.copy()
    # ensure datetime columns are parsed if strings
    for dt_col in ["session_start", "session_end", "departure_time", "return_time", "check_in_time", "check_out_time", "birthdate"]:
        if dt_col in df.columns and not np.issubdtype(df[dt_col].dtype, np.datetime64):
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")

    grouped = df.groupby("user_id")
    features = pd.DataFrame(index=grouped.size().index)

    # Travel counts
    features["num_trips"] = grouped["trip_id"].nunique().fillna(0)
    features["num_flights"] = grouped.get_group if False else grouped[["flight_booked"]].sum().fillna(0).squeeze() if "flight_booked" in df.columns else 0
    # The above line tries to produce a Series; ensure fallback
    if "flight_booked" in df.columns:
        features["num_flights"] = grouped["flight_booked"].sum().fillna(0)
    else:
        features["num_flights"] = 0

    if "hotel_booked" in df.columns:
        features["num_hotels"] = grouped["hotel_booked"].sum().fillna(0)
    else:
        features["num_hotels"] = 0

    # Cancellation
    if "cancellation" in df.columns:
        features["cancellation_rate"] = grouped["cancellation"].mean().fillna(0)
    else:
        features["cancellation_rate"] = 0

    # avg_km_flown
    if "flight_distance_km" in df.columns:
        features["avg_km_flown"] = grouped["flight_distance_km"].mean().fillna(0)

    # trip durations
    if {"check_in_time", "check_out_time"}.issubset(df.columns):
        trip_durations = (df["check_out_time"] - df["check_in_time"]).dt.days
        features["avg_trip_duration_days"] = trip_durations.groupby(df["user_id"]).mean().fillna(0)

    # multi booking rate (both flight+hotel in same trip)
    if {"flight_booked", "hotel_booked", "trip_id"}.issubset(df.columns):
        trip_bookings = df.groupby(["user_id", "trip_id"]).agg(
            flight=("flight_booked", "max"),
            hotel=("hotel_booked", "max")
        )
        multi = (trip_bookings["flight"] & trip_bookings["hotel"]).groupby("user_id").mean().fillna(0)
        features["multi_booking_rate"] = multi
    else:
        features["multi_booking_rate"] = 0

    # Spending
    # flight_discount_amount and hotel_price_per_room_night_usd used if available
    if "flight_discount_amount" in df.columns:
        features["total_flight_spend"] = grouped["flight_discount_amount"].sum().fillna(0)
    else:
        features["total_flight_spend"] = 0

    if "hotel_price_per_room_night_usd" in df.columns and {"check_in_time", "check_out_time"}.issubset(df.columns):
        stay_nights = (df["check_out_time"] - df["check_in_time"]).dt.days.fillna(0)
        hotel_spend = df["hotel_price_per_room_night_usd"].fillna(0) * stay_nights * df.get("rooms", 1)
        features["total_hotel_spend"] = hotel_spend.groupby(df["user_id"]).sum().fillna(0)
    else:
        features["total_hotel_spend"] = 0

    features["total_spend"] = features["total_flight_spend"] + features["total_hotel_spend"]
    features["avg_spend_per_trip"] = (features["total_spend"] / features["num_trips"].replace(0, np.nan)).fillna(0)
    features["hotel_vs_flight_ratio"] = (features["total_hotel_spend"] / (features["total_flight_spend"] + features["total_hotel_spend"] + 1e-9)).fillna(0)

    # Engagement
    session_duration = (df["session_end"] - df["session_start"]).dt.total_seconds() / 60 if {"session_end", "session_start"}.issubset(df.columns) else pd.Series(0, index=df.index)
    features["num_sessions"] = grouped["session_id"].nunique().fillna(0)
    features["avg_session_duration"] = session_duration.groupby(df["user_id"]).mean().fillna(0)
    features["avg_clicks_per_session"] = grouped["page_clicks"].mean().fillna(0) if "page_clicks" in df.columns else 0
    if "page_clicks" in df.columns:
        features["click_to_booking_ratio"] = (grouped["page_clicks"].sum() / (features["num_trips"] + 1e-9)).fillna(0)
    else:
        features["click_to_booking_ratio"] = 0

    # Booking behavior
    if {"departure_time", "session_start"}.issubset(df.columns):
        lead_times = (df["departure_time"] - df["session_start"]).dt.days
        features["avg_lead_time_days"] = lead_times.groupby(df["user_id"]).mean().fillna(0)
        features["lead_time_variability"] = lead_times.groupby(df["user_id"]).std().fillna(0)
    else:
        features["avg_lead_time_days"] = 0
        features["lead_time_variability"] = 0

    if "check_out_time" in df.columns:
        last_trip = grouped["check_out_time"].max()
        global_max = df["check_out_time"].max()
        features["last_trip_recency_days"] = (global_max - last_trip).dt.days.fillna(9999)
    else:
        features["last_trip_recency_days"] = 9999

    # Demographics
    if "birthdate" in df.columns:
        ages = ((pd.Timestamp("today") - df["birthdate"]).dt.days / 365.25).groupby(df["user_id"]).mean()
        features["age"] = ages.fillna(0)
    else:
        features["age"] = 0

    features["age_bucket_under_25"] = (features["age"] < 25).astype(int)
    features["age_bucket_25_40"] = ((features["age"] >= 25) & (features["age"] < 40)).astype(int)
    features["age_bucket_40_60"] = ((features["age"] >= 40) & (features["age"] < 60)).astype(int)
    features["age_bucket_60_plus"] = (features["age"] >= 60).astype(int)

    if "married" in df.columns:
        features["married"] = grouped["married"].max().fillna(0)
    else:
        features["married"] = 0

    if "has_children" in df.columns:
        features["has_children"] = grouped["has_children"].max().fillna(0)
    else:
        features["has_children"] = 0

    features["family_travel_intensity"] = features["has_children"] * features["num_trips"]

    # Discount behavior
    if "flight_discount" in df.columns:
        features["flight_discount_usage_rate"] = grouped["flight_discount"].mean().fillna(0)
    else:
        features["flight_discount_usage_rate"] = 0

    if "hotel_discount" in df.columns:
        features["hotel_discount_usage_rate"] = grouped["hotel_discount"].mean().fillna(0)
    else:
        features["hotel_discount_usage_rate"] = 0

    features["avg_discount_amount"] = (df["flight_discount_amount"].fillna(0) + df["hotel_discount_amount"].fillna(0)).groupby(df["user_id"]).mean().fillna(0)

    # Fill any remaining NaNs with 0
    features = features.fillna(0)

    # reset index
    features = features.reset_index()
    return features


# -----------------------------
# Clustering utilities
# -----------------------------

def standardize_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xs = pd.DataFrame(Xs, index=X.index, columns=X.columns)
    return Xs, scaler


def run_kmeans(X: pd.DataFrame, n_clusters: int) -> KMeans:
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(X)
    return model


def run_gmm(X: pd.DataFrame, n_components: int) -> GaussianMixture:
    model = GaussianMixture(n_components=n_components, random_state=42)
    model.fit(X)
    return model


def run_agglomerative(X: pd.DataFrame, n_clusters: int) -> AgglomerativeClustering:
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    return labels


def run_dbscan(X: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels


# -----------------------------
# Evaluation & plotting
# -----------------------------

def evaluate_clustering(X: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
    metrics = {}
    # need at least 2 clusters for silhouette
    unique_labels = len(set(labels)) - (1 if -1 in labels else 0)
    try:
        if unique_labels >= 2:
            metrics["silhouette"] = float(silhouette_score(X, labels))
        else:
            metrics["silhouette"] = np.nan
    except Exception:
        metrics["silhouette"] = np.nan

    try:
        if unique_labels >= 2:
            metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
        else:
            metrics["davies_bouldin"] = np.nan
    except Exception:
        metrics["davies_bouldin"] = np.nan

    metrics["n_clusters"] = int(unique_labels)
    return metrics


def plot_pca_clusters(X: pd.DataFrame, labels: np.ndarray, title: str, out_path: str):
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    dfz = pd.DataFrame(Z, columns=["PC1", "PC2"])
    dfz["label"] = labels

    plt.figure(figsize=(8,6))
    unique = np.unique(labels)
    for lab in unique:
        mask = dfz["label"] == lab
        plt.scatter(dfz.loc[mask, "PC1"], dfz.loc[mask, "PC2"], label=f"{lab}", alpha=0.6)
    plt.legend(title="cluster")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -----------------------------
# Orchestration
# -----------------------------

def run_full_pipeline(
    df: object,
    output_dir: str,
    clustering_methods: List[str] = None,
    n_clusters: int = 4,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5
) -> Dict[str, Any]:
    """
    Run the full pipeline: load -> build features -> standardize -> run clustering methods -> evaluate -> save outputs.
    Returns a dict with models, labels and metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build user-level features
    user_feats = build_user_features(df)
    user_feats.to_csv(os.path.join(output_dir, "user_features_raw.csv"), index=False)

    # Prepare matrix for clustering (drop id)
    X = user_feats.drop(columns=["user_id"], errors="ignore").set_index(user_feats["user_id"] if "user_id" in user_feats.columns else user_feats.index)

    # Keep feature names
    feat_names = X.columns.tolist()

    # Standardize
    Xs, scaler = standardize_features(X)
    Xs.to_csv(os.path.join(output_dir, "user_features_scaled.csv"))

    results = {}
    methods = clustering_methods or ["kmeans", "gmm", "agg", "dbscan"]

    for method in methods:
        if method == "kmeans":
            model = run_kmeans(Xs, n_clusters=n_clusters)
            labels = model.labels_
            metrics = evaluate_clustering(Xs, labels)
            plot_pca_clusters(Xs, labels, f"KMeans (k={n_clusters})", os.path.join(output_dir, "kmeans_pca.png"))
            results["kmeans"] = {"model": model, "labels": labels, "metrics": metrics}

        elif method == "gmm":
            model = run_gmm(Xs, n_components=n_clusters)
            labels = model.predict(Xs)
            metrics = evaluate_clustering(Xs, labels)
            plot_pca_clusters(Xs, labels, f"GMM (k={n_clusters})", os.path.join(output_dir, "gmm_pca.png"))
            results["gmm"] = {"model": model, "labels": labels, "metrics": metrics}

        elif method == "agg":
            labels = run_agglomerative(Xs, n_clusters=n_clusters)
            metrics = evaluate_clustering(Xs, labels)
            plot_pca_clusters(Xs, labels, f"Agglomerative (k={n_clusters})", os.path.join(output_dir, "agg_pca.png"))
            results["agg"] = {"labels": labels, "metrics": metrics}

        elif method == "dbscan":
            labels = run_dbscan(Xs, eps=dbscan_eps, min_samples=dbscan_min_samples)
            metrics = evaluate_clustering(Xs, labels)
            plot_pca_clusters(Xs, labels, f"DBSCAN (eps={dbscan_eps})", os.path.join(output_dir, "dbscan_pca.png"))
            results["dbscan"] = {"labels": labels, "metrics": metrics}

        else:
            print(f"Unknown method: {method}")

    # Save metrics summary
    metrics_df = []
    for k, v in results.items():
        m = v.get("metrics", {})
        m_row = {"method": k, **m}
        metrics_df.append(m_row)
    pd.DataFrame(metrics_df).to_csv(os.path.join(output_dir, "clustering_metrics.csv"), index=False)

    # Save label assignments
    for k, v in results.items():
        labels = v.get("labels")
        if labels is not None:
            out = pd.DataFrame({"user_id": X.index, f"cluster_{k}": labels})
            out.to_csv(os.path.join(output_dir, f"labels_{k}.csv"), index=False)

    # Save PCA projection for interactive plotting later
    pca = PCA(n_components=2)
    proj = pca.fit_transform(Xs)
    pd.DataFrame(proj, columns=["pc1", "pc2"], index=Xs.index).to_csv(os.path.join(output_dir, "pca_projection.csv"))

    # Return results dict for programmatic use
    return {"features": user_feats, "scaled": Xs, "results": results, "scaler": scaler}


# -----------------------------
# Command line interface
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run clustering pipeline")
    parser.add_argument("--input", required=True, help="Path to input CSV (session-level)")
    parser.add_argument("--output", required=True, help="Output directory to save results")
    parser.add_argument("--methods", default="kmeans,gmm,agg,dbscan", help="Comma-separated methods to run")
    parser.add_argument("--n_clusters", type=int, default=4, help="Number of clusters for kmeans/gmm/agg")
    parser.add_argument("--dbscan_eps", type=float, default=0.5, help="DBSCAN eps")
    parser.add_argument("--dbscan_min_samples", type=int, default=5, help="DBSCAN min samples")

    args = parser.parse_args()
    methods = args.methods.split(",") if args.methods else None

    run_full_pipeline(
        input_path=args.input,
        output_dir=args.output,
        clustering_methods=methods,
        n_clusters=args.n_clusters,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples
    )
