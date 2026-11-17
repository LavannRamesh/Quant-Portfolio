# -*- coding: utf-8 -*-
"""
Traffic Forecaster -- Streamlit UI
===================================
Pure rendering layer. All computation lives in logic.py.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from common.theme import render_header, PLOTLY_TEMPLATE, COLORS
from projects.traffic_forecaster.data_loader import load_traffic_dataset
from projects.traffic_forecaster.logic import (
    create_traffic_features,
    train_traffic_model,
    evaluate_traffic_model,
    get_feature_importances,
)


def render() -> None:
    """Entry point called by the master app.py router."""
    render_header(
        "Traffic Speed Forecaster (PeMS / METR-LA)",
        "Predicting highway traffic velocities 15 to 60 minutes ahead with gradient-boosted lag models",
        icon="🚗",
    )

    # ── Sidebar Configuration ─────────────────────────────────────────
    st.sidebar.markdown("### Forecast Parameters")
    horizon = st.sidebar.select_slider(
        "Forecast Horizon",
        options=[15, 30, 45, 60],
        value=15,
        format_func=lambda x: f"{x} Minutes Ahead",
    )

    with st.spinner("Loading traffic sensor telemetry..."):
        df_raw, is_real, data_msg = load_traffic_dataset()

    if not is_real and data_msg:
        st.info(f"ℹ️ **Data Pipeline Notice**: {data_msg}")
    elif is_real:
        st.success("🟢 Operating on authentic METR-LA / PeMS traffic telemetry.")

    available_sensors = sorted(df_raw["sensor_id"].unique())
    selected_sensor = st.sidebar.selectbox("Highway Sensor", available_sensors, index=0)

    # ── Feature Pipeline & Training (Cached) ─────────────────────────
    @st.cache_data(show_spinner=False)
    def _pipeline(sensor: str, hz: int):
        X, y, f_names = create_traffic_features(df_raw, sensor, horizon_minutes=hz)
        split_idx = int(len(X) * 0.8)
        X_tr, y_tr = X.iloc[:split_idx], y.iloc[:split_idx]
        X_te, y_te = X.iloc[split_idx:], y.iloc[split_idx:]
        mdl = train_traffic_model(X_tr, y_tr)
        metrics, preds = evaluate_traffic_model(mdl, X_te, y_te)
        imp = get_feature_importances(mdl, f_names)
        return metrics, preds, y_te, imp

    with st.spinner("Training LightGBM Forecaster..."):
        metrics, preds, y_test, importances = _pipeline(selected_sensor, horizon)

    # ── Executive Metrics ─────────────────────────────────────────────
    st.markdown(f"### Performance at Horizon: **{horizon} Minutes**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{metrics['MAE']:.2f} mph")
    m2.metric("RMSE", f"{metrics['RMSE']:.2f} mph")
    m3.metric("R² Score", f"{metrics['R2']:.3f}")
    m4.metric("MAPE", f"{metrics['MAPE']:.2f}%")

    # ── Charts ────────────────────────────────────────────────────────
    # Chart 1: Time Series Forecast vs Ground Truth
    st.markdown("### Ground Truth vs. Predicted Speed (Out-of-Sample)")
    display_steps = min(288, len(y_test))  # Display last 24 hours (288 * 5min)
    test_dates = y_test.index[-display_steps:]
    y_actual = y_test.values[-display_steps:]
    y_pred = preds[-display_steps:]

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=test_dates, y=y_actual, mode="lines", name="Observed Speed",
        line=dict(color=COLORS["primary"], width=2),
    ))
    fig_ts.add_trace(go.Scatter(
        x=test_dates, y=y_pred, mode="lines", name=f"{horizon}-min LightGBM Forecast",
        line=dict(color=COLORS["secondary"], width=2, dash="dash"),
    ))
    fig_ts.update_layout(
        template=PLOTLY_TEMPLATE,
        xaxis_title="Timestamp",
        yaxis_title="Speed (mph)",
        height=400,
        margin=dict(t=30),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Chart 2: Feature Importance
        st.markdown("### Top Predictive Signals")
        top_imp = importances.head(10).sort_values("Importance", ascending=True)
        fig_imp = px.bar(
            top_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            template=PLOTLY_TEMPLATE,
            color="Importance",
            color_continuous_scale="Teal",
        )
        fig_imp.update_layout(
            height=380,
            margin=dict(t=20, l=10, r=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_right:
        # Chart 3: Speed Profile by Hour
        st.markdown("### Diurnal Velocity Profile")
        sensor_data = df_raw[df_raw["sensor_id"] == selected_sensor].copy()
        sensor_data["hour"] = sensor_data["timestamp"].dt.hour
        hourly = sensor_data.groupby("hour")["speed_mph"].agg(["mean", "std"]).reset_index()

        fig_profile = go.Figure()
        fig_profile.add_trace(go.Scatter(
            x=hourly["hour"], y=hourly["mean"], mode="lines+markers",
            name="Mean Speed", line=dict(color=COLORS["tertiary"], width=2)
        ))
        fig_profile.add_trace(go.Scatter(
            x=list(hourly["hour"]) + list(hourly["hour"])[::-1],
            y=list(hourly["mean"] + hourly["std"]) + list(hourly["mean"] - hourly["std"])[::-1],
            fill="toself",
            fillcolor="rgba(78, 205, 196, 0.15)",
            line=dict(width=0),
            name="±1 Std Dev",
        ))
        fig_profile.update_layout(
            template=PLOTLY_TEMPLATE,
            xaxis=dict(title="Hour of Day (0-23)", tickmode="linear", tick0=0, dtick=3),
            yaxis_title="Speed (mph)",
            height=380,
            margin=dict(t=20, l=10, r=10),
        )
        st.plotly_chart(fig_profile, use_container_width=True)
