import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from preprocessing import prepare_clustering_features, CLUSTERING_FEATURES
import joblib

CLUSTER_MODEL = None
CLUSTER_LABELS = None
PCA_2D = None
CLUSTER_SCALER = None
CLUSTER_COLS = None
TRAINED = False
OPTIMAL_K = 5


def compute_elbow_silhouette(df, k_range=range(2, 11)):
    X_scaled, cols, scaler = prepare_clustering_features(df)
    wcss = []
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        wcss.append(km.inertia_)
        if k > 1:
            sil_scores.append(silhouette_score(X_scaled, labels))
        else:
            sil_scores.append(0)
    return list(k_range), wcss, sil_scores


def train_clustering(df, k=5):
    global CLUSTER_MODEL, CLUSTER_LABELS, PCA_2D, CLUSTER_SCALER, CLUSTER_COLS, TRAINED, OPTIMAL_K

    X_scaled, cols, scaler = prepare_clustering_features(df)
    CLUSTER_SCALER = scaler
    CLUSTER_COLS = cols
    OPTIMAL_K = k

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    CLUSTER_MODEL = km
    CLUSTER_LABELS = labels

    pca = PCA(n_components=2, random_state=42)
    PCA_2D = pca.fit_transform(X_scaled)

    joblib.dump(km, "cluster_model.pkl")
    joblib.dump(scaler, "cluster_scaler.pkl")
    joblib.dump({"cols": cols, "k": k}, "cluster_meta.pkl")

    TRAINED = True
    return km, labels, PCA_2D


def get_cluster_profiles(df):
    if CLUSTER_LABELS is None:
        return None
    df2 = df.copy()
    avail = [c for c in CLUSTERING_FEATURES if c in df2.columns]
    df2["cluster"] = CLUSTER_LABELS
    profile = df2.groupby("cluster")[avail].mean().round(2)
    if "persona_label" in df2.columns:
        persona_mode = df2.groupby("cluster")["persona_label"].agg(lambda x: x.mode()[0])
        profile["dominant_persona"] = persona_mode
    return profile


def validate_clusters_vs_personas(df):
    if CLUSTER_LABELS is None or "persona_id" not in df.columns:
        return None, None
    true_labels = df["persona_id"].values[:len(CLUSTER_LABELS)]
    ari = adjusted_rand_score(true_labels, CLUSTER_LABELS)
    ct = pd.crosstab(CLUSTER_LABELS, df["persona_label"].values[:len(CLUSTER_LABELS)],
                     rownames=["Cluster"], colnames=["Persona"])
    return ari, ct


def predict_cluster_new(df_new):
    import os
    if not os.path.exists("cluster_model.pkl"):
        return None
    km = joblib.load("cluster_model.pkl")
    scaler = joblib.load("cluster_scaler.pkl")
    meta = joblib.load("cluster_meta.pkl")
    cols = meta["cols"]

    from preprocessing import clean_dataframe
    df_clean = clean_dataframe(df_new)
    avail = [c for c in cols if c in df_clean.columns]
    X = df_clean[avail].copy().astype(float)
    for c in cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[cols]
    X_scaled = scaler.transform(X)
    return km.predict(X_scaled)


CLUSTER_PERSONA_MAP = {
    0: "Confused Explorer",
    1: "Focused Achiever",
    2: "Career Switcher",
    3: "Budget Conscious Learner",
    4: "Parent Driven Student"
}
