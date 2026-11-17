# -*- coding: utf-8 -*-
"""
Airbnb Pricing Engine -- Core Logic
====================================
Predicts nightly Airbnb prices using XGBoost and computes SHAP (SHapley Additive
exPlanations) values to interpret feature impacts on predictions.
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def preprocess_airbnb_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, Any]]:
    """
    Encode categorical features and prepare numerical feature matrices.
    """
    df_clean = df.copy()

    # Log transform target to handle skewness
    y = df_clean["price"]

    # One-hot encode categoricals
    cat_cols = ["neighbourhood", "room_type"]
    num_cols = [
        "accommodates",
        "bedrooms",
        "bathrooms",
        "distance_to_center_km",
        "number_of_reviews",
        "review_rating",
        "has_wifi",
        "has_kitchen",
        "has_ac",
        "is_superhost",
        "availability_365",
        "minimum_nights",
    ]

    # Filter to existing columns
    active_num = [c for c in num_cols if c in df_clean.columns]
    active_cat = [c for c in cat_cols if c in df_clean.columns]

    X = pd.get_dummies(df_clean[active_num + active_cat], columns=active_cat, drop_first=True, dtype=float)
    feature_names = list(X.columns)

    metadata = {
        "num_cols": active_num,
        "cat_cols": active_cat,
        "feature_names": feature_names,
    }

    return X, y, feature_names, metadata


def train_airbnb_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.08,
    seed: int = 42,
) -> xgb.XGBRegressor:
    """Train XGBoost Regressor with objective reg:squarederror."""
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        n_jobs=-1,
    )
    # Train on log(y) for stable proportional pricing
    model.fit(X_train, np.log(y_train))
    return model


def evaluate_airbnb_model(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Calculate MAE, RMSE, R2, and MAPE in real dollar terms."""
    log_preds = model.predict(X_test)
    preds = np.exp(log_preds)
    y_true = y_test.values

    mae = float(mean_absolute_error(y_true, preds))
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    r2 = float(r2_score(y_true, preds))
    mape = float(np.mean(np.abs((y_true - preds) / y_true))) * 100.0

    metrics = {
        "MAE ($)": mae,
        "RMSE ($)": rmse,
        "R2": r2,
        "MAPE (%)": mape,
    }
    return metrics, preds


def compute_shap_explanations(
    model: xgb.XGBRegressor,
    X_sample: pd.DataFrame,
) -> Tuple[Any, np.ndarray, float]:
    """
    Compute SHAP values using TreeExplainer.
    Returns (explainer, shap_values, base_value).
    """
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)
    base_val = float(explainer.expected_value)
    return explainer, shap_vals, base_val


def predict_single_listing(
    model: xgb.XGBRegressor,
    user_inputs: Dict[str, Any],
    feature_names: List[str],
) -> Tuple[float, np.ndarray, float]:
    """
    Predict nightly price and return local SHAP breakdown for a single input record.
    """
    input_row = pd.DataFrame(0.0, index=[0], columns=feature_names)

    # Fill numericals
    for k, v in user_inputs.items():
        if k in input_row.columns:
            input_row.at[0, k] = float(v)

    # Fill one-hot categoricals
    if "neighbourhood" in user_inputs:
        col = f"neighbourhood_{user_inputs['neighbourhood']}"
        if col in input_row.columns:
            input_row.at[0, col] = 1.0

    if "room_type" in user_inputs:
        col = f"room_type_{user_inputs['room_type']}"
        if col in input_row.columns:
            input_row.at[0, col] = 1.0

    log_pred = float(model.predict(input_row)[0])
    price_pred = float(np.exp(log_pred))

    explainer = shap.TreeExplainer(model)
    local_shap = explainer.shap_values(input_row)[0]
    base_val = float(explainer.expected_value)

    return price_pred, local_shap, base_val


if __name__ == "__main__":
    from projects.airbnb_pricing.data_loader import load_airbnb_dataset

    df_test, is_real, msg = load_airbnb_dataset(sample_size=1500)
    X, y, f_names, meta = preprocess_airbnb_features(df_test)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    mdl = train_airbnb_model(X_tr, y_tr, n_estimators=60)
    metrics, p = evaluate_airbnb_model(mdl, X_te, y_te)
    print(f"Airbnb Model Evaluation: {metrics}")
    assert metrics["R2"] > 0.6, "Airbnb R2 too low!"

    # Test single prediction
    sample_input = {
        "accommodates": 4,
        "bedrooms": 2,
        "bathrooms": 1,
        "distance_to_center_km": 3.0,
        "neighbourhood": "Manhattan - Midtown",
        "room_type": "Entire home/apt",
        "has_wifi": 1,
        "has_kitchen": 1,
        "has_ac": 1,
        "is_superhost": 1,
        "review_rating": 4.9,
        "number_of_reviews": 50,
        "availability_365": 200,
        "minimum_nights": 2,
    }
    pred_val, sh_val, bv = predict_single_listing(mdl, sample_input, f_names)
    print(f"Sample Predicted Price: ${pred_val:.2f}")
    print("Airbnb Pricing Logic Test: PASSED")
