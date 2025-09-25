import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def run_pca(df_scaled, n_components):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df_scaled)
    n = pca.components_.shape[0]
    explained = pca.explained_variance_ratio_
    pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n)])
    pca_df.index = df_scaled.index
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


def pca(df):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(X_scaled)
    df_scaled.index = df.index
    df_scaled.columns = df.columns

    pca = PCA(n_components = 0.95)
    df_pca = pca.fit_transform(df_scaled)
    component_matrix = pd.DataFrame(pca.components_).T
    component_matrix.columns = [f"pca_{i}"for i in range(component_matrix.shape[1])]
    component_matrix.index = df_scaled.columns
    return component_matrix


def pca_analysis(df):
    # Schritt 3: Führen Sie die PCA durch. Wir setzen n_components auf None,
    # um die Varianz aller möglichen Hauptkomponenten zu berechnen.
    pca = PCA(n_components=None)
    pca.fit(df)

    # Schritt 4: Kumulative Varianz berechnen
    # Die 'explained_variance_ratio_' gibt den Prozentsatz der Varianz an,
    # der von jeder Hauptkomponente erklärt wird.
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Schritt 5: Das Diagramm erstellen
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='b')
    plt.title('Kumulative Varianz erklärt durch die Hauptkomponenten')
    plt.xlabel('Anzahl der Hauptkomponenten')
    plt.ylabel('Kumulierte erklärte Varianz (%)')
    plt.grid(True)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Varianzschwelle')

    # Den Punkt finden, an dem die kumulierte Varianz 95% überschreitet
    n_components_95 = np.where(cumulative_variance >= 0.95)[0][0] + 1
    plt.axvline(x=n_components_95, color='g', linestyle='--', label=f'{n_components_95} Komponenten für 95%')
    plt.text(n_components_95, 0.5, f'{n_components_95} Komponenten', color='g', ha='right', va='center')

    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()