# -*- coding: utf-8 -*-
"""
Traffic Forecaster -- Core Logic
=================================
Predicts highway traffic speeds 15 to 60 minutes ahead using LightGBM.
Implements:
- Multi-step lag features (t-5min, t-10min, ..., t-60min)
- Rolling window mean, min, and standard deviation
- Cyclical temporal embeddings (sin/cos of hour and day-of-week)
- Model training & multi-metric evaluation (MAE, RMSE, R2, MAPE)
- Tree split feature importance analysis
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def create_traffic_features(
    df: pd.DataFrame,
    sensor_id: str,
    horizon_minutes: int = 15,
    freq_minutes: int = 5,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Construct temporal and lag features for a specific sensor.

    Target = speed at t + horizon_minutes.
    """
    sub = df[df["sensor_id"] == sensor_id].sort_values("timestamp").copy()
    sub = sub.set_index("timestamp")

    steps_ahead = max(1, horizon_minutes // freq_minutes)
    speed = sub["speed_mph"]

    features = pd.DataFrame(index=sub.index)

    # 1. Autoregressive Lags (t-5, t-10, ..., t-60 min)
    for lag in range(1, 13):
        features[f"speed_lag_{lag * freq_minutes}m"] = speed.shift(lag)

    # 2. Rolling Statistics over past 30 & 60 mins
    features["rolling_mean_30m"] = speed.shift(1).rolling(6).mean()
    features["rolling_std_30m"] = speed.shift(1).rolling(6).std()
    features["rolling_min_30m"] = speed.shift(1).rolling(6).min()
    features["rolling_mean_60m"] = speed.shift(1).rolling(12).mean()

    # 3. Cyclical Temporal Encodings (sine / cosine)
    hours = sub.index.hour + sub.index.minute / 60.0
    features["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

    dows = sub.index.dayofweek
    features["dow_sin"] = np.sin(2 * np.pi * dows / 7.0)
    features["dow_cos"] = np.cos(2 * np.pi * dows / 7.0)
    features["is_weekend"] = (dows >= 5).astype(float)

    # Target: future speed
    target = speed.shift(-steps_ahead)

    # Align and clean NaNs
    valid_mask = ~(features.isna().any(axis=1) | target.isna())
    X = features[valid_mask]
    y = target[valid_mask]

    return X, y, list(X.columns)


def train_traffic_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 150,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    seed: int = 42,
) -> lgb.LGBMRegressor:
    """Train a LightGBM Regressor with time-series friendly hyperparameters."""
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbosity=-1,
        force_col_wise=True,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_traffic_model(
    model: lgb.LGBMRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Calculate MAE, RMSE, R2, and MAPE on holdout test set."""
    preds = model.predict(X_test)
    y_true = y_test.values

    mae = float(mean_absolute_error(y_true, preds))
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    r2 = float(r2_score(y_true, preds))
    mape = float(np.mean(np.abs((y_true - preds) / np.clip(y_true, 1e-3, None)))) * 100.0

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
    }
    return metrics, preds


def get_feature_importances(
    model: lgb.LGBMRegressor,
    feature_names: List[str],
) -> pd.DataFrame:
    """Extract and sort feature importances."""
    importances = model.feature_importances_
    df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from projects.traffic_forecaster.data_loader import load_traffic_dataset

    df_sample, is_real, msg = load_traffic_dataset(max_days=7)
    sensor = df_sample["sensor_id"].unique()[0]
    X, y, f_names = create_traffic_features(df_sample, sensor, horizon_minutes=15)
    split_idx = int(len(X) * 0.8)
    X_tr, y_tr = X.iloc[:split_idx], y.iloc[:split_idx]
    X_te, y_te = X.iloc[split_idx:], y.iloc[split_idx:]

    mdl = train_traffic_model(X_tr, y_tr)
    metrics, predictions = evaluate_traffic_model(mdl, X_te, y_te)
    print(f"Traffic Model Evaluation: {metrics}")
    assert metrics["R2"] > 0.5, "Traffic model R2 is too low!"
    print("Traffic Forecaster Logic Test: PASSED")
