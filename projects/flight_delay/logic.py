# -*- coding: utf-8 -*-
"""
Flight Delay Predictor -- Core Logic
=====================================
Predicts US domestic flight delay probabilities (defined as ARR_DELAY >= 15 min)
using LightGBM classifier.
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


def preprocess_flight_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, Any]]:
    """
    Construct cyclical encodings for temporal variables and categorical dummies for airlines & airports.
    """
    df_clean = df.copy()

    y = df_clean["is_delayed"]

    # Cyclical representations
    hour_rad = 2 * np.pi * df_clean["dep_hour"] / 24.0
    dow_rad = 2 * np.pi * df_clean["day_of_week"] / 7.0
    month_rad = 2 * np.pi * (df_clean["month"] - 1) / 12.0

    cyclical = pd.DataFrame({
        "hour_sin": np.sin(hour_rad),
        "hour_cos": np.cos(hour_rad),
        "dow_sin": np.sin(dow_rad),
        "dow_cos": np.cos(dow_rad),
        "month_sin": np.sin(month_rad),
        "month_cos": np.cos(month_rad),
        "is_weekend": (df_clean["day_of_week"] >= 5).astype(float),
        "distance_miles": df_clean["distance_miles"],
    })

    # Dummies for airline and origin airport
    cats = pd.get_dummies(df_clean[["airline", "origin", "destination"]], drop_first=True, dtype=float)

    X = pd.concat([cyclical, cats], axis=1)
    feature_names = list(X.columns)

    meta = {
        "airlines": sorted(df_clean["airline"].unique()),
        "airports": sorted(df_clean["origin"].unique()),
        "base_delay_rate": float(y.mean()),
    }

    return X, y, feature_names, meta


def train_flight_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 180,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    seed: int = 42,
) -> lgb.LGBMClassifier:
    """Train LightGBM Classifier with class weight balanced."""
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=31,
        class_weight="balanced",
        random_state=seed,
        verbosity=-1,
        force_col_wise=True,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_flight_model(
    model: lgb.LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Compute classification metrics, ROC curve data, and confusion matrix."""
    prob_delayed = model.predict_proba(X_test)[:, 1]
    y_pred = (prob_delayed >= 0.50).astype(int)
    y_true = y_test.values

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, prob_delayed))

    fpr, tpr, _ = roc_curve(y_true, prob_delayed)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "metrics": {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "AUC-ROC": auc,
        },
        "roc_curve": {"fpr": fpr, "tpr": tpr},
        "confusion_matrix": cm,
        "prob_predictions": prob_delayed,
    }


def predict_flight_risk(
    model: lgb.LGBMClassifier,
    airline: str,
    origin: str,
    destination: str,
    month: int,
    day_of_week: int,
    dep_hour: int,
    distance_miles: float,
    feature_names: List[str],
) -> float:
    """Predict the delay probability (0.0 to 1.0) for a specific user flight configuration."""
    row = pd.DataFrame(0.0, index=[0], columns=feature_names)

    # Cyclicals
    hour_rad = 2 * np.pi * dep_hour / 24.0
    dow_rad = 2 * np.pi * day_of_week / 7.0
    month_rad = 2 * np.pi * (month - 1) / 12.0

    row.at[0, "hour_sin"] = np.sin(hour_rad)
    row.at[0, "hour_cos"] = np.cos(hour_rad)
    row.at[0, "dow_sin"] = np.sin(dow_rad)
    row.at[0, "dow_cos"] = np.cos(dow_rad)
    row.at[0, "month_sin"] = np.sin(month_rad)
    row.at[0, "month_cos"] = np.cos(month_rad)
    row.at[0, "is_weekend"] = 1.0 if day_of_week >= 5 else 0.0
    row.at[0, "distance_miles"] = float(distance_miles)

    # Categories
    airline_col = f"airline_{airline}"
    if airline_col in row.columns:
        row.at[0, airline_col] = 1.0

    origin_col = f"origin_{origin}"
    if origin_col in row.columns:
        row.at[0, origin_col] = 1.0

    dest_col = f"destination_{destination}"
    if dest_col in row.columns:
        row.at[0, dest_col] = 1.0

    prob = float(model.predict_proba(row)[0, 1])
    return prob


def get_feature_importances(
    model: lgb.LGBMClassifier,
    feature_names: List[str],
) -> pd.DataFrame:
    """Return top feature importances."""
    imp = model.feature_importances_
    return pd.DataFrame({"Feature": feature_names, "Importance": imp}).sort_values("Importance", ascending=False)


if __name__ == "__main__":
    from projects.flight_delay.data_loader import load_flight_dataset

    df_sample, is_real, msg = load_flight_dataset(sample_size=3000)
    X, y, f_names, meta = preprocess_flight_features(df_sample)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    mdl = train_flight_model(X_tr, y_tr, n_estimators=60)
    eval_res = evaluate_flight_model(mdl, X_te, y_te)
    print(f"Flight Model Metrics: {eval_res['metrics']}")
    assert eval_res["metrics"]["AUC-ROC"] > 0.60, "Flight model AUC too low!"

    p = predict_flight_risk(
        mdl,
        airline="United (UA)",
        origin="ORD",
        destination="SFO",
        month=7,
        day_of_week=4,
        dep_hour=18,
        distance_miles=1850.0,
        feature_names=f_names,
    )
    print(f"Sample Flight Delay Probability: {p:.1%}")
    print("Flight Delay Predictor Logic Test: PASSED")
