import pandas as pd
import numpy as np
import os
import joblib
from preprocessing import (encode_features, assign_pricing_tier, assign_marketing_tag,
                            clean_dataframe, CLUSTERING_FEATURES)

CLUSTER_PERSONA_MAP = {
    0: "Confused Explorer",
    1: "Focused Achiever",
    2: "Career Switcher",
    3: "Budget Conscious Learner",
    4: "Parent Driven Student"
}


def validate_schema(df):
    required_cols = [
        "A_city_tier","A_gender","A_education_level","A_stream",
        "A_family_income_score","B_prior_paid_score","B_acad_performance_score",
        "B_self_learning_score","C_confusion_level_score","C_worry_frequency_score",
        "C_family_decision_score","D_risk_tolerance_score","D_motivation_score",
        "D_locus_of_control_score"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    present = [c for c in required_cols if c in df.columns]
    return missing, present


def impute_missing_columns(df):
    df = df.copy()
    defaults_numeric = {
        "A_family_income_score": 2.5, "B_prior_paid_score": 1.5,
        "B_acad_performance_score": 3.0, "B_self_learning_score": 2.0,
        "C_confusion_level_score": 3.0, "C_worry_frequency_score": 3.0,
        "C_family_decision_score": 3.0, "D_risk_tolerance_score": 1.5,
        "D_motivation_score": 3.0, "D_locus_of_control_score": 3.0,
        "A_age": 19,
        "E_feat_career_match": 3, "E_feat_roadmap": 3, "E_feat_job_market": 3,
        "E_feat_chatbot": 3, "E_feat_college_rec": 3, "E_feat_mock_interview": 3,
        "E_feat_scholarship": 3, "E_feat_peer_bench": 3,
    }
    defaults_cat = {
        "A_city_tier": "Tier 2", "A_gender": "Male",
        "A_education_level": "Undergraduate", "A_stream": "Science",
        "A_institution_type": "Private", "C_career_clarity_stage": "Have interests, dont know career match",
        "C_family_decision_authority": "Equal joint decision",
        "D_risk_tolerance": "Stable job", "D_motivation_type": "Balanced both",
        "E_career_domain_1": "Technology", "E_learning_preference": "Watching videos",
        "E_primary_device": "Smartphone Android"
    }
    for col, val in defaults_numeric.items():
        if col not in df.columns:
            df[col] = val
    for col, val in defaults_cat.items():
        if col not in df.columns:
            df[col] = val
    return df


def run_full_prediction(df_new, models_exist=True):
    df_new = impute_missing_columns(df_new)
    df_new = clean_dataframe(df_new)
    results = df_new.copy()

    # Classification prediction
    adoption_prob = np.full(len(df_new), 0.5)
    adoption_pred = np.zeros(len(df_new), dtype=int)
    try:
        if os.path.exists("clf_xgb.pkl"):
            clf = joblib.load("clf_xgb.pkl")
            enc = joblib.load("clf_encoders.pkl")
            scl = joblib.load("clf_scaler.pkl")
            X_enc, _, _ = encode_features(df_new, fit=False, encoders=enc, scaler=scl)
            adoption_prob = clf.predict_proba(X_enc)[:,1]
            adoption_pred = (adoption_prob >= 0.55).astype(int)
    except Exception as e:
        pass

    # Regression prediction
    wtp_pred = np.full(len(df_new), 3.0)
    try:
        if os.path.exists("reg_rf.pkl"):
            reg = joblib.load("reg_rf.pkl")
            enc_r = joblib.load("reg_encoders.pkl")
            scl_r = joblib.load("reg_scaler.pkl")
            X_enc_r, _, _ = encode_features(df_new, fit=False, encoders=enc_r, scaler=scl_r)
            wtp_pred = reg.predict(X_enc_r)
    except Exception as e:
        pass

    # Cluster assignment
    cluster_ids = np.zeros(len(df_new), dtype=int)
    try:
        if os.path.exists("cluster_model.pkl"):
            km = joblib.load("cluster_model.pkl")
            scaler_c = joblib.load("cluster_scaler.pkl")
            meta = joblib.load("cluster_meta.pkl")
            cols = meta["cols"]
            avail = [c for c in cols if c in df_new.columns]
            X_c = df_new[avail].copy().astype(float)
            for c in cols:
                if c not in X_c.columns:
                    X_c[c] = 0.0
            X_c = X_c[cols]
            X_scaled = scaler_c.transform(X_c)
            cluster_ids = km.predict(X_scaled)
    except Exception as e:
        pass

    persona_labels = [CLUSTER_PERSONA_MAP.get(int(c), "Unknown") for c in cluster_ids]
    pricing_tiers = [assign_pricing_tier(w) for w in wtp_pred]
    marketing_tags = [assign_marketing_tag(p, w, pers) for p, w, pers in zip(adoption_prob, wtp_pred, persona_labels)]

    results["predicted_adoption_prob"] = np.round(adoption_prob, 3)
    results["predicted_adoption"] = np.where(adoption_pred==1, "Yes", "No")
    results["predicted_wtp_score"] = np.round(wtp_pred, 2)
    results["assigned_cluster"] = cluster_ids
    results["assigned_persona"] = persona_labels
    results["pricing_tier"] = pricing_tiers
    results["marketing_tag"] = marketing_tags

    summary = {
        "total_uploaded": len(results),
        "high_priority": int((results["marketing_tag"].str.contains("High Priority")).sum()),
        "avg_wtp": float(np.round(wtp_pred.mean(), 2)),
        "predicted_adopters": int(adoption_pred.sum()),
        "adoption_rate": float(np.round(adoption_pred.mean()*100, 1))
    }

    return results, summary
