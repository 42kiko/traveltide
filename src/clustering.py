"""
Clustering utilities for the TravelTide project.

This module defines a `SegmentationPipeline` class that wraps common steps
required to cluster user-level feature matrices. It uses PCA for dimensionality
reduction and KMeans for clustering, and offers convenience methods for
evaluating and visualising results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# typing imports
from typing import Optional, Tuple, Dict



class SegmentationPipeline:
    """
    A simple pipeline for reducing feature dimensionality via PCA and
    subsequently clustering the data using KMeans.

    Parameters
    ----------
    variance_threshold : float, optional
        Proportion of total variance that PCA should retain. Defaults to 0.95.
    max_clusters : int, optional
        Maximum number of clusters to evaluate when searching for the optimal
        number of KMeans clusters. Defaults to 10.
    """

    def __init__(self, variance_threshold: float = 0.95, max_clusters: int = 10) -> None:
        self.variance_threshold = variance_threshold
        self.max_clusters = max_clusters
        self.scaler = StandardScaler()
        self.pca: Optional[PCA] = None
        self.components: Optional[np.ndarray] = None
        self.explained_var: Optional[np.ndarray] = None
        self.kmeans_model: Optional[KMeans] = None
        self.kmeans_scores: dict[int, float] = {}

    # -------------------
    # Data Preparation
    # -------------------
    def prepare_features(self, df: pd.DataFrame, drop_cols: Optional[list[str]] = None) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Drop specified columns, remove rows with missing values and scale features.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw features for clustering.
        drop_cols : list of str, optional
            Column names to drop before scaling. Defaults to ``None``.

        Returns
        -------
        tuple of (numpy.ndarray, pandas.DataFrame)
            Scaled features and the cleaned DataFrame.
        """
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")
        df = df.dropna()
        features_scaled = self.scaler.fit_transform(df)
        return features_scaled, df

    # -------------------
    # PCA
    # -------------------
    def run_pca(self, features: np.ndarray) -> tuple[np.ndarray, PCA, np.ndarray]:
        """
        Fit a PCA model and transform the feature matrix.

        Parameters
        ----------
        features : numpy.ndarray
            Scaled feature matrix.

        Returns
        -------
        components : numpy.ndarray
            The PCA-transformed components.
        pca : sklearn.decomposition.PCA
            The fitted PCA model.
        explained_var : numpy.ndarray
            Cumulative explained variance ratio.
        """
        self.pca = PCA(n_components=self.variance_threshold)
        self.components = self.pca.fit_transform(features)
        self.explained_var = np.cumsum(self.pca.explained_variance_ratio_)
        return self.components, self.pca, self.explained_var

    def plot_explained_variance(self) -> None:
        """Plot cumulative explained variance across PCA components."""
        if self.explained_var is None:
            raise ValueError("Run run_pca() before plotting explained variance.")
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(self.explained_var) + 1), self.explained_var, marker='o')
        plt.axhline(y=self.variance_threshold, color='r', linestyle='--')
        plt.title("Explained Variance by PCA Components")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.show()

    # -------------------
    # KMeans
    # -------------------
    def kmeans_clustering(self, n_clusters: Optional[int] = None) -> tuple[KMeans, dict[int, float]]:
        """
        Cluster the PCA components using KMeans.

        Parameters
        ----------
        n_clusters : int, optional
            If provided, fit a KMeans model with this fixed number of clusters.
            Otherwise, iterate through 2 up to ``max_clusters`` and select the
            number of clusters that maximises the silhouette score.

        Returns
        -------
        tuple
            The fitted KMeans model and a dictionary of silhouette scores keyed by
            ``k``.
        """
        if self.components is None:
            raise ValueError("Run run_pca() before clustering.")
        if n_clusters:
            # user-provided number of clusters
            self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = self.kmeans_model.fit_predict(self.components)
            score = silhouette_score(self.components, labels)
            self.kmeans_scores = {n_clusters: score}
        else:
            scores: dict[int, float] = {}
            for k in range(2, self.max_clusters + 1):
                model = KMeans(n_clusters=k, random_state=42)
                labels = model.fit_predict(self.components)
                score = silhouette_score(self.components, labels)
                scores[k] = score
            best_k = max(scores, key=scores.get)
            self.kmeans_model = KMeans(n_clusters=best_k, random_state=42).fit(self.components)
            self.kmeans_scores = scores
        return self.kmeans_model, self.kmeans_scores
    # -------------------
    # Optimized KMeans
    # -------------------
    def optimized_kmeans_clustering(self, n_clusters: Optional[int] = None, verbose: bool = False) -> tuple[KMeans, dict[int, float]]:
        """
        Fit KMeans with improved initialisation and multiple restarts.

        Parameters
        ----------
        n_clusters : int, optional
            If provided, fit KMeans with this fixed number of clusters.
        verbose : bool, optional
            Whether to print silhouette scores during the search. Defaults to ``False``.

        Returns
        -------
        tuple
            The fitted KMeans model and a dictionary of silhouette scores keyed by
            ``k``.
        """
        if self.components is None:
            raise ValueError("Run run_pca() before clustering.")
        kmeans_params = {
            'init': 'k-means++',
            'n_init': 20,
            'max_iter': 300,
            'random_state': 42,
        }
        if n_clusters:
            self.kmeans_model = KMeans(n_clusters=n_clusters, **kmeans_params)
            labels = self.kmeans_model.fit_predict(self.components)
            score = silhouette_score(self.components, labels)
            self.kmeans_scores = {n_clusters: score}
        else:
            scores: dict[int, float] = {}
            for k in range(2, self.max_clusters + 1):
                model = KMeans(n_clusters=k, **kmeans_params)
                labels = model.fit_predict(self.components)
                score = silhouette_score(self.components, labels)
                scores[k] = score
                if verbose:
                    print(f"KMeans with {k} clusters: {score:.4f}")
            best_k = max(scores, key=scores.get)
            self.kmeans_model = KMeans(n_clusters=best_k, **kmeans_params).fit(self.components)
            self.kmeans_scores = scores
        return self.kmeans_model, self.kmeans_scores

    # -------------------
    # Plotting
    # -------------------
    def plot_clusters(self) -> None:
        """Visualise the clustered PCA components as a 2D scatter plot."""
        if self.kmeans_model is None or self.components is None:
            raise ValueError("Run kmeans_clustering() first.")
        labels = self.kmeans_model.predict(self.components)
        plt.figure(figsize=(6, 6))
        sns.scatterplot(
            x=self.components[:, 1], y=self.components[:, 0],
            hue=labels, palette="Set2", s=50
        )
        plt.title(f"KMeans Clusters (k={self.kmeans_model.n_clusters})")
        plt.xlabel("PCA Component 2")
        plt.ylabel("PCA Component 1")
        plt.show()

    def plot_scores(self) -> None:
        """Plot silhouette scores for different numbers of clusters."""
        if not self.kmeans_scores:
            raise ValueError("Run kmeans_clustering() first.")
        plt.figure(figsize=(8, 5))
        ks = list(self.kmeans_scores.keys())
        scores = [self.kmeans_scores[k] for k in ks]
        plt.plot(ks, scores, marker='o')
        plt.title("Silhouette Scores for KMeans")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.show()

    # -------------------
    # Cluster Assignment
    # -------------------
    def assign_clusters(self, df: pd.DataFrame, user_key: str = "user_id") -> pd.DataFrame:
        """
        Add cluster labels back to the user-level feature DataFrame.

        This method accepts a ``user_key`` parameter for backward
        compatibility but it is ignored because clustering operates on the
        index order of the PCA components. The user-level DataFrame
        ``df`` should correspond to the features passed to the clustering
        pipeline in the same order.

        Parameters
        ----------
        df : pandas.DataFrame
            The user-level features that were clustered.
        user_key : str, optional
            Ignored. Kept for backward compatibility.

        Returns
        -------
        pandas.DataFrame
            Copy of ``df`` with an additional ``cluster`` column.
        """
        if self.kmeans_model is None or self.components is None:
            raise ValueError("Run kmeans_clustering() before assigning labels.")
        labels = self.kmeans_model.predict(self.components)
        df_with_clusters = df.copy()
        df_with_clusters["cluster"] = labels
        return df_with_clusters

    # -------------------
    # Cluster Profiles
    # -------------------
    def cluster_summary(self, df_with_clusters: pd.DataFrame) -> pd.DataFrame:
        """
        Compute descriptive statistics for each cluster.

        Parameters
        ----------
        df_with_clusters : pandas.DataFrame
            User-level features with a ``cluster`` column.

        Returns
        -------
        pandas.DataFrame
            Aggregated statistics (mean, median, min, max, count) per cluster.
        """
        return (
            df_with_clusters
            .groupby("cluster")
            .agg(["mean", "median", "min", "max", "count"])
        )

    def cluster_means(self, df_with_clusters: pd.DataFrame) -> pd.DataFrame:
        """Return only mean values per cluster (simpler view)."""
        return df_with_clusters.groupby("cluster").mean(numeric_only=True)

    def plot_cluster_feature(self, df_with_clusters: pd.DataFrame, feature: str) -> None:
        """
        Plot the distribution of a feature across clusters using a boxplot.

        Parameters
        ----------
        df_with_clusters : pandas.DataFrame
            User-level features with a ``cluster`` column.
        feature : str
            Column name of the feature to plot.
        """
        if feature not in df_with_clusters.columns:
            raise ValueError(f"{feature} not found in DataFrame")
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="cluster", y=feature, data=df_with_clusters, palette="Set2", hue="cluster")
        plt.title(f"Distribution of {feature} across clusters")
        plt.xlabel("Cluster")
        plt.ylabel(feature)
        plt.show()