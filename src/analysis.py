# analysis.py
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def run_pca(df, n_components: int=2):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df.select_dtypes(include=['float64','int64']))
    explained = pca.explained_variance_ratio_
    pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n_components)])
    return pca_df, pca, explained

def run_kmeans(df, n_clusters: int=4):
    X = df.select_dtypes(include=['float64','int64'])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    sil = silhouette_score(X, labels)
    df_out = df.copy()
    df_out["cluster"] = labels
    return df_out, kmeans, sil

def plot_pca_clusters(pca_df, labels):
    plt.figure(figsize=(8,6))
    plt.scatter(pca_df["PC1"], pca_df["PC2"], c=labels, cmap="viridis", alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Clusters")
    plt.colorbar(label="Cluster")
    plt.show()


def find_optimal_k(df, k_min: int=2, k_max: int=10):
    X = df.select_dtypes(include=['float64','int64'])
    results = []
    for k in range(k_min, k_max+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        sil = silhouette_score(X, labels)
        results.append((k, sil))
    return results