import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score


# ------------------------------
# PCA-Komponenten automatisch wählen
# ------------------------------

def choose_pca_components(data, variance_threshold=0.95, random_state=42):
    """
    Wählt automatisch die Anzahl an PCA-Komponenten,
    sodass mindestens variance_threshold (z.B. 95%) erklärt wird.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    pca = PCA(random_state=random_state)
    pca.fit(data_scaled)

    explained = pca.explained_variance_ratio_.cumsum()
    n_components = (explained < variance_threshold).sum() + 1

    print(f"PCA: {n_components} Komponenten erklären {explained[n_components-1]:.2%} der Varianz.")

    return n_components


# ------------------------------
# Pipelines
# ------------------------------

def build_cluster_pipeline(n_clusters=2, n_components=None, random_state=42):
    """
    Pipeline mit StandardScaler + PCA + KMeans.
    Falls n_components=None, wird automatisch 95% Varianz angestrebt.
    """
    steps = [("scaler", StandardScaler())]

    if n_components is None:
        pca = PCA(n_components=0.95, random_state=random_state)  # Auto: 95%
    else:
        pca = PCA(n_components=n_components, random_state=random_state)

    steps.append(("pca", pca))
    steps.append(("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state)))

    return Pipeline(steps)


# ------------------------------
# Analyse-Methoden
# ------------------------------

def silhouette_analysis(data, k_range=range(2, 11), n_components=None, random_state=42):
    scores = {}
    best_score = -1
    best_k = None
    best_labels = None
    best_pca_data = None
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    for k in k_range:
        pipeline = build_cluster_pipeline(k, n_components, random_state)
        pipeline.fit(data_scaled)

        labels = pipeline["kmeans"].labels_
        score = silhouette_score(data_scaled, labels)
        scores[k] = score

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_pca_data = pipeline["pca"].transform(data_scaled)

    # Plot Scores
    plt.figure(figsize=(8, 5))
    plt.plot(list(scores.keys()), list(scores.values()), marker="o", linestyle="-")
    plt.title("Silhouette Scores für verschiedene k")
    plt.xlabel("Anzahl Cluster (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    # Plot Beste Cluster
    plot_pca_clusters(best_pca_data, best_labels,
                      title=f"Beste Cluster (k={best_k}, Score={best_score:.3f})")

    return best_k, best_score, scores


def elbow_method(data, k_range=range(2, 11), n_components=None, random_state=42):
    inertias = {}
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    for k in k_range:
        pipeline = build_cluster_pipeline(k, n_components, random_state)
        pipeline.fit(data_scaled)
        inertias[k] = pipeline["kmeans"].inertia_

    # Plot Inertia
    plt.figure(figsize=(8, 5))
    plt.plot(list(inertias.keys()), list(inertias.values()), marker="o", linestyle="-")
    plt.title("Elbow-Methode (Inertia)")
    plt.xlabel("Anzahl Cluster (k)")
    plt.ylabel("Inertia")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    return inertias


# ------------------------------
# Plot-Helfer
# ------------------------------

def plot_pca_clusters(pca_data, labels, title="PCA Cluster Plot"):
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


# ------------------------------
# Report-Export
# ------------------------------

def export_cluster_report(data, labels, pca_data, silhouette_scores, inertias, filename="cluster_report.csv"):
    """
    Exportiert ein CSV mit Clusterlabels, PCA-Koordinaten und allen Scores.
    """
    report_df = pd.DataFrame(data).copy()
    report_df["cluster"] = labels
    report_df["pc1"] = pca_data[:, 0]
    report_df["pc2"] = pca_data[:, 1]

    # Scores als Extra-Tabelle
    score_summary = {
        "silhouette_scores": silhouette_scores,
        "inertias": inertias
    }
    score_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in score_summary.items()]))

    # Speichern
    report_df.to_csv(filename, index=False)
    score_df.to_csv("cluster_scores.csv")

    print(f"✅ Clusterreport gespeichert: {filename}")
    print(f"✅ Scores gespeichert: cluster_scores.csv")