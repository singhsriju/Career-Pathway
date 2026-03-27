import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

NUMERIC_FEATURES = [
    "A_family_income_score", "B_prior_paid_score", "B_acad_performance_score",
    "B_self_learning_score", "C_confusion_level_score", "C_worry_frequency_score",
    "C_family_decision_score", "D_risk_tolerance_score", "D_motivation_score",
    "D_locus_of_control_score", "E_feat_career_match", "E_feat_roadmap",
    "E_feat_job_market", "E_feat_chatbot", "E_feat_college_rec",
    "E_feat_mock_interview", "E_feat_scholarship", "E_feat_peer_bench", "A_age"
]

CATEGORICAL_FEATURES = [
    "A_city_tier", "A_gender", "A_education_level", "A_stream",
    "A_institution_type", "C_career_clarity_stage", "C_family_decision_authority",
    "D_risk_tolerance", "D_motivation_type", "E_career_domain_1",
    "E_learning_preference", "E_primary_device"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CLUSTERING_FEATURES = [
    "C_confusion_level_score", "D_locus_of_control_score", "D_motivation_score",
    "D_risk_tolerance_score", "C_family_decision_score", "F_wtp_score_continuous",
    "B_self_learning_score", "B_acad_performance_score", "C_worry_frequency_score",
    "B_prior_paid_score"
]


def clean_dataframe(df):
    df = df.copy()
    # Fill numeric NAs with median
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
    # Fill categorical NAs with mode
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    if "F_wtp_score_continuous" in df.columns:
        df["F_wtp_score_continuous"] = pd.to_numeric(df["F_wtp_score_continuous"], errors="coerce")
        df["F_wtp_score_continuous"] = df["F_wtp_score_continuous"].fillna(df["F_wtp_score_continuous"].median())
    return df


def encode_features(df, fit=True, encoders=None, scaler=None):
    df = clean_dataframe(df)
    if encoders is None:
        encoders = {}
    X = pd.DataFrame()

    # Numeric features
    num_df = pd.DataFrame()
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            num_df[col] = df[col].astype(float)
        else:
            num_df[col] = 0.0

    if fit:
        scaler = StandardScaler()
        X_num = pd.DataFrame(scaler.fit_transform(num_df), columns=num_df.columns)
    else:
        if scaler is not None:
            X_num = pd.DataFrame(scaler.transform(num_df), columns=num_df.columns)
        else:
            X_num = num_df

    X = pd.concat([X, X_num], axis=1)

    # Categorical features
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            if fit:
                le = LabelEncoder()
                encoded = le.fit_transform(df[col].astype(str))
                encoders[col] = le
            else:
                le = encoders.get(col)
                if le is not None:
                    known = set(le.classes_)
                    df[col] = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
                    encoded = le.transform(df[col])
                else:
                    encoded = np.zeros(len(df), dtype=int)
            X[col + "_enc"] = encoded
        else:
            X[col + "_enc"] = 0

    return X, encoders, scaler


def prepare_clustering_features(df):
    df = clean_dataframe(df)
    avail = [c for c in CLUSTERING_FEATURES if c in df.columns]
    X = df[avail].copy().astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, avail, scaler


def get_feature_names(encoders):
    names = list(NUMERIC_FEATURES)
    for col in CATEGORICAL_FEATURES:
        names.append(col + "_enc")
    return names


def assign_pricing_tier(wtp_score):
    if wtp_score <= 1.5:
        return "Freemium (Free)"
    elif wtp_score <= 4.5:
        return "Basic (Rs199-Rs499)"
    elif wtp_score <= 7.5:
        return "Premium (Rs500-Rs999)"
    else:
        return "Enterprise (Rs1000+)"


def assign_marketing_tag(adoption_prob, wtp_score, persona):
    if adoption_prob >= 0.75 and wtp_score >= 6:
        return "High Priority - Premium Outreach"
    elif adoption_prob >= 0.75 and wtp_score < 6:
        return "High Priority - Basic Plan Push"
    elif adoption_prob >= 0.55:
        return "Nurture - Free Trial First"
    elif persona in ["Parent Driven Student", "Budget Conscious Learner"]:
        return "Parent-Targeted Campaign"
    elif adoption_prob >= 0.30:
        return "B2B School Channel"
    else:
        return "Awareness Only - Long Term"


def save_artifacts(encoders, scaler, cluster_scaler, filepath_prefix="model"):
    joblib.dump(encoders, filepath_prefix + "_encoders.pkl")
    joblib.dump(scaler, filepath_prefix + "_scaler.pkl")
    joblib.dump(cluster_scaler, filepath_prefix + "_cluster_scaler.pkl")


def load_artifacts(filepath_prefix="model"):
    encoders = joblib.load(filepath_prefix + "_encoders.pkl")
    scaler = joblib.load(filepath_prefix + "_scaler.pkl")
    cluster_scaler = joblib.load(filepath_prefix + "_cluster_scaler.pkl")
    return encoders, scaler, cluster_scaler
