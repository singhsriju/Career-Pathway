import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve, confusion_matrix,
                              classification_report)
import xgboost as xgb
import joblib
from preprocessing import encode_features, get_feature_names

MODELS = {}
METRICS = {}
TRAINED = False
X_TRAIN = None
X_TEST = None
Y_TRAIN = None
Y_TEST = None
ENCODERS = None
SCALER = None
FEAT_NAMES = None


def train_classifiers(df):
    global MODELS, METRICS, TRAINED, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, ENCODERS, SCALER, FEAT_NAMES

    target = "F_will_use_binary"
    df_clean = df.dropna(subset=[target]).copy()
    df_clean[target] = df_clean[target].astype(int)

    X_enc, encoders, scaler = encode_features(df_clean, fit=True)
    y = df_clean[target].values

    ENCODERS = encoders
    SCALER = scaler
    FEAT_NAMES = get_feature_names(encoders)

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.2, random_state=42, stratify=y)

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = X_train, X_test, y_train, y_test

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss",
                                       scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                                       verbosity=0)
    }

    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {
            "model": clf,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "auc": round(roc_auc_score(y_test, y_prob), 4),
            "fpr": fpr,
            "tpr": tpr,
            "confusion_matrix": cm,
            "classification_report": cr,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    MODELS = results
    METRICS = {name: {k: v for k, v in r.items() if k != "model"} for name, r in results.items()}
    TRAINED = True

    joblib.dump(results["XGBoost"]["model"], "clf_xgb.pkl")
    joblib.dump(results["Random Forest"]["model"], "clf_rf.pkl")
    joblib.dump(results["Logistic Regression"]["model"], "clf_lr.pkl")
    joblib.dump(encoders, "clf_encoders.pkl")
    joblib.dump(scaler, "clf_scaler.pkl")

    return results, X_test, y_test, encoders, scaler


def get_feature_importance(model_name):
    if model_name not in MODELS:
        return None
    model = MODELS[model_name]["model"]
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        return pd.DataFrame({"feature": FEAT_NAMES, "importance": fi}).sort_values("importance", ascending=False)
    elif hasattr(model, "coef_"):
        fi = np.abs(model.coef_[0])
        return pd.DataFrame({"feature": FEAT_NAMES, "importance": fi}).sort_values("importance", ascending=False)
    return None


def predict_new(df_new):
    import os
    if not os.path.exists("clf_xgb.pkl"):
        return None
    model = joblib.load("clf_xgb.pkl")
    encoders = joblib.load("clf_encoders.pkl")
    scaler = joblib.load("clf_scaler.pkl")
    X_enc, _, _ = encode_features(df_new, fit=False, encoders=encoders, scaler=scaler)
    prob = model.predict_proba(X_enc)[:,1]
    pred = (prob >= 0.55).astype(int)
    return prob, pred
