import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
from sklearn.impute import SimpleImputer


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



class AdvancedSegmentationPipeline:
    def __init__(self, variance_threshold=0.95, max_clusters=10):
        self.variance_threshold = variance_threshold
        self.max_clusters = max_clusters
        self.scaler = StandardScaler()
        self.pca = None
        self.components = None
        self.explained_var = None
        self.kmeans_model = None
        self.kmeans_scores = {}
        self.best_method = None
        self.best_score = None
        self.best_labels = None

    # -------------------
    # Data Preparation
    # -------------------
    def prepare_features(self, df, drop_cols=None):
        """Vorbereitung der Features mit verbesserter Imputation"""
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")

        # Null-Werte behandeln
        imputer = SimpleImputer(strategy='median')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)

        # Skalierung
        features_scaled = self.scaler.fit_transform(df_imputed)
        return features_scaled, df_imputed

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
        plt.grid(True)
        plt.show()

    # -------------------
    # Optimized KMeans
    # -------------------
    def optimized_kmeans_clustering(self, n_clusters=None):
        """Optimiertes KMeans mit mehr Initialisierungen"""
        if n_clusters:
            # Explizite Cluster-Anzahl
            self.kmeans_model = KMeans(
                n_clusters=n_clusters,
                init='k-means++',
                n_init=20,
                max_iter=300,
                random_state=42
            )
            labels = self.kmeans_model.fit_predict(self.components)
            score = silhouette_score(self.components, labels)
            self.kmeans_scores[n_clusters] = score
        else:
            # Beste Cluster-Anzahl finden
            scores = {}
            for k in range(2, self.max_clusters + 1):
                model = KMeans(
                    n_clusters=k,
                    init='k-means++',
                    n_init=20,
                    max_iter=300,
                    random_state=42
                )
                labels = model.fit_predict(self.components)
                score = silhouette_score(self.components, labels)
                scores[k] = score
                print(f"KMeans with {k} clusters: {score:.4f}")

            best_k = max(scores, key=scores.get)
            self.kmeans_model = KMeans(
                n_clusters=best_k,
                init='k-means++',
                n_init=20,
                random_state=42
            ).fit(self.components)
            self.kmeans_scores = scores

        return self.kmeans_model, self.kmeans_scores

    # -------------------
    # Alternative Methods
    # -------------------
    def dbscan_clustering(self, eps=0.5, min_samples=5):
        """DBSCAN Clustering für dichtebasierte Cluster"""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.components)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1:
            score = silhouette_score(self.components, labels)
            print(f"DBSCAN found {n_clusters} clusters with score: {score:.4f}")
        else:
            score = -1
            print("DBSCAN found only one cluster")

        return labels, score

    def gmm_clustering(self, n_components_range=range(3, 8)):
        """Gaussian Mixture Models Clustering"""
        best_score = -1
        best_n = n_components_range[0]
        best_labels = None

        for n in n_components_range:
            gmm = GaussianMixture(n_components=n, random_state=42)
            labels = gmm.fit_predict(self.components)
            score = silhouette_score(self.components, labels)
            print(f"GMM with {n} components: {score:.4f}")

            if score > best_score:
                best_score = score
                best_n = n
                best_labels = labels

        return best_labels, best_score, best_n

    def ensemble_clustering(self, n_clusters_list=[3, 5, 7]):
        """Ensemble Clustering für robustere Ergebnisse"""
        all_labels = []

        for n_clusters in n_clusters_list:
            # Mehrere KMeans mit verschiedenen Random States
            for random_state in [42, 123, 456]:
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=random_state,
                    n_init=10
                )
                labels = kmeans.fit_predict(self.components)
                all_labels.append(labels)

        # Ensemble-Labels durch Mehrheitsentscheid
        all_labels_matrix = np.column_stack(all_labels)
        ensemble_labels, _ = mode(all_labels_matrix, axis=1)
        ensemble_labels = ensemble_labels.flatten()

        score = silhouette_score(self.components, ensemble_labels)
        print(f"Ensemble Clustering Score: {score:.4f}")

        return ensemble_labels, score

    # -------------------
    # Comprehensive Clustering
    # -------------------
    def run_comprehensive_clustering(self, target_clusters=None):
        """
        Führt umfassendes Clustering mit verschiedenen Methoden durch
        und wählt die beste Methode basierend auf Silhouette Score
        """
        methods_results = {}

        print("=== COMPREHENSIVE CLUSTERING ANALYSIS ===")

        # 1. Optimiertes KMeans
        print("\n1. OPTIMIZED KMEANS:")
        if target_clusters:
            kmeans_model, kmeans_scores = self.optimized_kmeans_clustering(n_clusters=target_clusters)
            kmeans_score = kmeans_scores[target_clusters]
            kmeans_labels = kmeans_model.labels_
        else:
            kmeans_model, kmeans_scores = self.optimized_kmeans_clustering()
            kmeans_score = max(kmeans_scores.values())
            kmeans_labels = kmeans_model.labels_

        methods_results['KMeans'] = (kmeans_labels, kmeans_score)

        # 2. Ensemble Clustering
        print("\n2. ENSEMBLE CLUSTERING:")
        ensemble_labels, ensemble_score = self.ensemble_clustering()
        methods_results['Ensemble'] = (ensemble_labels, ensemble_score)

        # 3. GMM Clustering
        print("\n3. GAUSSIAN MIXTURE MODELS:")
        gmm_labels, gmm_score, best_n = self.gmm_clustering()
        methods_results['GMM'] = (gmm_labels, gmm_score)

        # 4. DBSCAN (optional)
        print("\n4. DBSCAN:")
        dbscan_labels, dbscan_score = self.dbscan_clustering()
        if dbscan_score > 0:
            methods_results['DBSCAN'] = (dbscan_labels, dbscan_score)

        # Beste Methode auswählen
        self.best_method = max(methods_results.keys(),
                              key=lambda x: methods_results[x][1])
        self.best_labels, self.best_score = methods_results[self.best_method]

        print(f"\n=== BEST METHOD: {self.best_method} ===")
        print(f"Best Silhouette Score: {self.best_score:.4f}")

        return self.best_labels, self.best_score, methods_results

    # -------------------
    # Plotting
    # -------------------
    def plot_clusters(self, labels=None):
        """Plot der Cluster-Verteilung"""
        if labels is None:
            if self.best_labels is not None:
                labels = self.best_labels
            elif self.kmeans_model:
                labels = self.kmeans_model.labels_
            else:
                raise ValueError("Run clustering first or provide labels.")

        plt.figure(figsize=(10, 6))

        if self.components.shape[1] >= 2:
            # 2D Plot wenn möglich
            scatter = plt.scatter(
                self.components[:, 0], self.components[:, 1],
                c=labels, cmap='viridis', alpha=0.6, s=50
            )
            plt.colorbar(scatter)
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
        else:
            # 1D Histogram
            plt.hist(labels, bins=len(set(labels)), alpha=0.7, edgecolor='black')
            plt.xlabel('Cluster')
            plt.ylabel('Number of Users')

        plt.title(f'Cluster Distribution ({len(set(labels))} clusters)')
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_scores(self):
        """Plot der Silhouette Scores"""
        if not self.kmeans_scores:
            raise ValueError("Run KMeans clustering first.")

        plt.figure(figsize=(10, 6))
        plt.plot(list(self.kmeans_scores.keys()), list(self.kmeans_scores.values()),
                marker='o', linewidth=2, markersize=8)
        plt.title("Silhouette Scores for KMeans")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_comparison(self, methods_results):
        """Vergleich der verschiedenen Clustering-Methoden"""
        methods = list(methods_results.keys())
        scores = [methods_results[method][1] for method in methods]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])

        # Werte auf den Bars anzeigen
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.4f}', ha='center', va='bottom')

        plt.title('Comparison of Clustering Methods')
        plt.ylabel('Silhouette Score')
        plt.ylim(0, max(scores) + 0.1)
        plt.grid(True, alpha=0.3)
        plt.show()

    # -------------------
    # Cluster Assignment & Analysis
    # -------------------
    def assign_clusters(self, df, labels=None, user_key="user_id"):
        """Weist Cluster-Labels den Usern zu"""
        if labels is None:
            if self.best_labels is not None:
                labels = self.best_labels
            else:
                raise ValueError("Run clustering first or provide labels.")

        df_with_clusters = df.copy()
        df_with_clusters["cluster"] = labels
        return df_with_clusters

    def cluster_summary(self, df_with_clusters):
        """Detaillierte Cluster-Zusammenfassung"""
        return df_with_clusters.groupby("cluster").agg(["mean", "std", "count"])

    def cluster_means(self, df_with_clusters):
        """Vereinfachte Cluster-Mittelwerte"""
        return df_with_clusters.groupby("cluster").mean(numeric_only=True)

    def plot_cluster_feature(self, df_with_clusters, feature):
        """Feature-Verteilung über Cluster"""
        if feature not in df_with_clusters.columns:
            raise ValueError(f"Feature '{feature}' not in DataFrame")

        plt.figure(figsize=(10, 6))
        sns.boxplot(x="cluster", y=feature, data=df_with_clusters, palette="Set2")
        plt.title(f"Distribution of {feature} across clusters")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Behalten Sie Ihre ursprüngliche Klasse für Abwärtskompatibilität bei
class SegmentationPipeline(AdvancedSegmentationPipeline):
    """Abwärtskompatible Version Ihrer ursprünglichen Pipeline"""
    def kmeans_clustering(self, n_clusters=None):
        # Ruft die optimierte Version auf
        return self.optimized_kmeans_clustering(n_clusters)