import pandas as pd

def cluster_profiles(df_with_clusters, cluster_col="cluster"):
    # Numerische Features: Mittelwerte pro Cluster
    profile_num = df_with_clusters.groupby(cluster_col).mean(numeric_only=True)

    # Kategorische Features: Häufigkeiten pro Cluster
    profile_cat = {}
    for col in df_with_clusters.select_dtypes(include=["object", "category"]).columns:
        profile_cat[col] = df_with_clusters.groupby(cluster_col)[col].agg(lambda x: x.value_counts().index[0])

    profile_cat = pd.DataFrame(profile_cat)

    return profile_num, profile_cat