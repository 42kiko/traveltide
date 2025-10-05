"""
clustering.py
~~~~~~~~~~~~~

This module contains functionality to perform unsupervised clustering
on the engineered feature set. It provides tools to standardise
features, reduce dimensionality with PCA, select an appropriate
number of clusters using silhouette scores and train a KMeans model.

Functions
---------
scale_and_reduce(features_df: pd.DataFrame, n_components: int = 5) -> Tuple[np.ndarray, PCA, StandardScaler]
    Standardise the feature matrix and apply Principal Component Analysis.

find_optimal_clusters(X: np.ndarray, k_range: Iterable[int]) -> Dict[int, float]
    Compute silhouette scores for a range of cluster counts to aid
    selection of the optimal ``k``.

fit_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 42) -> KMeans
    Fit a KMeans model to the reduced feature matrix.

assign_clusters(model: KMeans, X: np.ndarray) -> np.ndarray
    Predict cluster labels for each sample.

compute_cluster_summary(features_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame
    Generate aggregate statistics per cluster to facilitate profiling.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def scale_and_reduce(features_df: pd.DataFrame, n_components: int = 5) -> Tuple[np.ndarray, PCA, StandardScaler]:
    """Standardise features and reduce dimensionality using PCA.

    Parameters
    ----------
    features_df : pd.DataFrame
        The user‑level feature matrix. All columns should be numeric.
    n_components : int, optional
        Number of principal components to retain. Defaults to 5.

    Returns
    -------
    np.ndarray
        The PCA‑transformed features array.
    PCA
        The fitted PCA instance.
    StandardScaler
        The scaler used to standardise the data.
    """
    # Ensure we only scale numeric columns
    X = features_df.select_dtypes(include=[np.number]).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)
    return X_reduced, pca, scaler


def find_optimal_clusters(X: np.ndarray, k_range: Iterable[int] = range(2, 8)) -> Dict[int, float]:
    """Calculate silhouette scores to determine optimal cluster number.

    Parameters
    ----------
    X : np.ndarray
        The feature matrix (ideally PCA‑reduced) used for clustering.
    k_range : iterable of int, optional
        Range of ``k`` values (number of clusters) to evaluate. Defaults
        to 2 through 7.

    Returns
    -------
    dict
        A mapping of ``k`` to its corresponding silhouette score.

    Notes
    -----
    Higher silhouette scores indicate better clustering structure.
    """
    scores = {}
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X)
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
        else:
            score = -1  # invalid clustering
        scores[k] = score
    return scores


def fit_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 42) -> KMeans:
    """Fit a KMeans clustering model.

    Parameters
    ----------
    X : np.ndarray
        The feature matrix (ideally PCA‑reduced) used for clustering.
    n_clusters : int
        Number of clusters to form.
    random_state : int, optional
        Seed used by the random number generator. Defaults to 42.

    Returns
    -------
    KMeans
        The fitted KMeans model.
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    model.fit(X)
    return model


def assign_clusters(model: KMeans, X: np.ndarray) -> np.ndarray:
    """Predict cluster labels using a trained clustering model.

    Parameters
    ----------
    model : KMeans
        A fitted KMeans model.
    X : np.ndarray
        The feature matrix used to predict cluster labels.

    Returns
    -------
    np.ndarray
        An array of cluster labels for each sample.
    """
    return model.predict(X)


def compute_cluster_summary(features_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Compute aggregate statistics for each cluster.

    Parameters
    ----------
    features_df : pd.DataFrame
        The DataFrame containing user‑level features.
    labels : np.ndarray
        Cluster labels for each user (aligned with ``features_df`` rows).

    Returns
    -------
    pd.DataFrame
        A DataFrame summarising key metrics for each cluster.
    """
    df = features_df.copy()
    df['cluster'] = labels
    agg_dict = {
        'cluster_size': ('cluster', 'count'),
        'booking_count_avg': ('booking_count', 'mean'),
        'flights_booked_avg': ('flights_booked_count', 'mean'),
        'hotels_booked_avg': ('hotels_booked_count', 'mean'),
        'avg_spend_per_booking_avg': ('avg_spend_per_booking', 'mean'),
        'cancellation_rate_avg': ('cancellation_rate', 'mean'),
        'discount_usage_rate_avg': ('discount_usage_rate', 'mean'),
        'age_avg': ('age_mean', 'mean'),
        'is_married_ratio': ('is_married', 'mean'),
        'has_children_ratio': ('has_children', 'mean'),
        'sign_up_year_avg': ('sign_up_year', 'mean'),
    }
    summary = df.groupby('cluster').agg(**agg_dict)
    summary = summary.reset_index().rename(columns={'cluster': 'cluster_id'})
    # Calculate revenue share if spend present
    total_revenue = df['base_fare_sum'].sum() + df['hotel_price_sum'].sum()
    if total_revenue > 0:
        summary['revenue_share'] = (
            df.groupby('cluster')[['base_fare_sum', 'hotel_price_sum']].sum().sum(axis=1) / total_revenue
        ).values
    return summary
