import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocessing import encode_features, assign_pricing_tier, get_feature_names
import joblib

REG_MODELS = {}
REG_METRICS = {}
TRAINED = False
ENCODERS = None
SCALER = None
FEAT_NAMES = None


def train_regressors(df):
    global REG_MODELS, REG_METRICS, TRAINED, ENCODERS, SCALER, FEAT_NAMES

    target = "F_wtp_score_continuous"
    df_clean = df.dropna(subset=[target]).copy()
    df_clean[target] = pd.to_numeric(df_clean[target], errors="coerce")
    df_clean = df_clean.dropna(subset=[target])

    X_enc, encoders, scaler = encode_features(df_clean, fit=True)
    y = df_clean[target].values

    ENCODERS = encoders
    SCALER = scaler
    FEAT_NAMES = get_feature_names(encoders)

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.2, random_state=42)

    regressors = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }

    results = {}
    for name, reg in regressors.items():
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results[name] = {
            "model": reg,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "y_test": y_test,
            "y_pred": y_pred,
            "residuals": y_test - y_pred,
        }

    REG_MODELS = results
    REG_METRICS = {name: {k: v for k, v in r.items() if k != "model"} for name, r in results.items()}
    TRAINED = True

    best_model = results["Random Forest"]["model"]
    joblib.dump(best_model, "reg_rf.pkl")
    joblib.dump(encoders, "reg_encoders.pkl")
    joblib.dump(scaler, "reg_scaler.pkl")

    return results, X_test, y_test, encoders, scaler


def get_regression_feature_importance():
    if "Random Forest" not in REG_MODELS:
        return None
    model = REG_MODELS["Random Forest"]["model"]
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        return pd.DataFrame({"feature": FEAT_NAMES, "importance": fi}).sort_values("importance", ascending=False)
    return None


def predict_wtp_new(df_new):
    import os
    if not os.path.exists("reg_rf.pkl"):
        return None
    model = joblib.load("reg_rf.pkl")
    encoders = joblib.load("reg_encoders.pkl")
    scaler = joblib.load("reg_scaler.pkl")
    X_enc, _, _ = encode_features(df_new, fit=False, encoders=encoders, scaler=scaler)
    return model.predict(X_enc)


def assign_pricing_tiers_bulk(df):
    if ENCODERS is None:
        return df
    from preprocessing import encode_features as enc
    X_enc, _, _ = enc(df, fit=False, encoders=ENCODERS, scaler=SCALER)
    model = REG_MODELS.get("Random Forest", {}).get("model")
    if model is None:
        return df
    wtp_pred = model.predict(X_enc)
    df = df.copy()
    df["predicted_wtp"] = np.round(wtp_pred, 2)
    df["pricing_tier"] = [assign_pricing_tier(w) for w in wtp_pred]
    return df
