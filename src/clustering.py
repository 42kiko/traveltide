import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


class SegmentationPipeline:
    def __init__(self, variance_threshold=0.95, max_clusters=10):
        self.variance_threshold = variance_threshold
        self.max_clusters = max_clusters
        self.scaler = StandardScaler()
        self.pca = None
        self.components = None
        self.explained_var = None
        self.kmeans_model = None
        self.kmeans_scores = {}

    # -------------------
    # Data Preparation
    # -------------------
    def prepare_features(self, df, drop_cols=None):
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")
        df = df.dropna()
        features_scaled = self.scaler.fit_transform(df)
        return features_scaled, df

    # -------------------
    # PCA
    # -------------------
    def run_pca(self, features):
        self.pca = PCA(n_components=self.variance_threshold)
        self.components = self.pca.fit_transform(features)
        self.explained_var = np.cumsum(self.pca.explained_variance_ratio_)
        return self.components, self.pca, self.explained_var

    def plot_explained_variance(self):
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(self.explained_var) + 1), self.explained_var, marker='o')
        plt.axhline(y=0.95, color='r', linestyle='--')
        plt.title("Explained Variance by PCA Components")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.show()

    # -------------------
    # KMeans
    # -------------------
    def kmeans_clustering(self, n_clusters=None):
        if n_clusters:  # explizit vorgegeben
            self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = self.kmeans_model.fit_predict(self.components)
            score = silhouette_score(self.components, labels)
            self.kmeans_scores[n_clusters] = score
        else:  # bestes K suchen
            scores = {}
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
    # Plotting
    # -------------------
    def plot_clusters(self):
        if not self.kmeans_model:
            raise ValueError("Run kmeans_clustering() first.")

        labels = self.kmeans_model.predict(self.components)
        plt.figure(figsize=(6, 6))
        sns.scatterplot(
            x=self.components[:, 1], y=self.components[:, 0],
            hue=labels, palette="Set2", s=50
        )
        plt.title(f"KMeans Clusters (k={self.kmeans_model.n_clusters})")
        plt.show()

    def plot_scores(self):
        if not self.kmeans_scores:
            raise ValueError("Run kmeans_clustering() first.")

        plt.figure(figsize=(8, 5))
        plt.plot(list(self.kmeans_scores.keys()), list(self.kmeans_scores.values()), marker='o')
        plt.title("Silhouette Scores for KMeans")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.show()

    # -------------------
    # Cluster Assignment
    # -------------------
    def assign_clusters(self, df, user_key="user_id"):
        """
        Add cluster labels back to the user DataFrame.
        """
        if not self.kmeans_model:
            raise ValueError("Run kmeans_clustering() first.")

        labels = self.kmeans_model.predict(self.components)
        df_with_clusters = df.copy()
        df_with_clusters["cluster"] = labels
        return df_with_clusters

    # -------------------
    # Cluster Profiles
    # -------------------
    def cluster_summary(self, df_with_clusters, user_key="user_id"):
        """
        Compute descriptive statistics per cluster.
        """
        return (
            df_with_clusters
            .groupby("cluster")
            .agg(["mean", "median", "min", "max", "count"])
        )

    def cluster_means(self, df_with_clusters):
        """
        Return only mean values per cluster (simpler view).
        """
        return df_with_clusters.groupby("cluster").mean(numeric_only=True)

    def plot_cluster_feature(self, df_with_clusters, feature):
        """
        Plot feature distribution across clusters.
        """
        if feature not in df_with_clusters.columns:
            raise ValueError(f"{feature} not in DataFrame")

        plt.figure(figsize=(8, 5))
        sns.boxplot(x="cluster", y=feature, data=df_with_clusters, palette="Set2", hue="cluster")
        plt.title(f"Distribution of {feature} across clusters")
        plt.show()