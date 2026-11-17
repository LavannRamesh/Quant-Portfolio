# -*- coding: utf-8 -*-
"""
Flight Delay Predictor -- Streamlit UI
=======================================
Pure rendering layer. All computation lives in logic.py.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from common.theme import render_header, PLOTLY_TEMPLATE, COLORS
from projects.flight_delay.data_loader import load_flight_dataset
from projects.flight_delay.logic import (
    preprocess_flight_features,
    train_flight_model,
    evaluate_flight_model,
    predict_flight_risk,
    get_feature_importances,
)


def render() -> None:
    """Entry point called by the master app.py router."""
    render_header(
        "US Flight Delay Predictor (Bureau of Transportation Statistics)",
        "Estimate your real probability of encountering a 15+ minute delay across carriers, airports, and departure times",
        icon="✈️",
    )

    with st.spinner("Accessing US BTS aviation data..."):
        df_raw, is_real, data_msg = load_flight_dataset(sample_size=12000)

    if not is_real and data_msg:
        st.info(f"ℹ️ **Data Pipeline Notice**: {data_msg}")
    elif is_real:
        st.success("🟢 Operating on authentic US BTS TranStats aviation records.")

    # ── Pipeline & Caching ───────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def _train_pipeline():
        X, y, f_names, meta = preprocess_flight_features(df_raw)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        mdl = train_flight_model(X_tr, y_tr, n_estimators=180)
        eval_res = evaluate_flight_model(mdl, X_te, y_te)
        importances = get_feature_importances(mdl, f_names)
        return mdl, f_names, meta, eval_res, importances

    with st.spinner("Training LightGBM Classifier on historical flight records..."):
        model, f_names, meta, eval_res, importances = _train_pipeline()

    metrics = eval_res["metrics"]
    base_rate = meta["base_delay_rate"]

    # ── Navigation Tabs ──────────────────────────────────────────────
    tab_predict, tab_analytics = st.tabs([
        "🛫 Check Your Flight Delay Risk",
        "📊 Aviation Model Insights & ROC Analysis",
    ])

    with tab_predict:
        st.markdown("### Flight Itinerary Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            sel_airline = st.selectbox("Airline Carrier", meta["airlines"], index=0)
            sel_month = st.selectbox(
                "Month of Travel",
                options=list(range(1, 13)),
                format_func=lambda m: pd.to_datetime(f"2024-{m:02d}-01").strftime("%B"),
                index=6,
            )

        with col2:
            sel_origin = st.selectbox("Origin Airport", meta["airports"], index=0)
            dest_options = [a for a in meta["airports"] if a != sel_origin]
            sel_dest = st.selectbox("Destination Airport", dest_options, index=1)

        with col3:
            dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            sel_dow = st.selectbox("Day of Week", options=list(range(7)), format_func=lambda d: dow_names[d], index=4)
            sel_hour = st.slider("Scheduled Departure Hour", 5, 23, 17, format_func=lambda h: f"{h:02d}:00")

        # Estimate flight distance from typical hub pairs or slider
        est_distance = 1200.0

        user_prob = predict_flight_risk(
            model=model,
            airline=sel_airline,
            origin=sel_origin,
            destination=sel_dest,
            month=sel_month,
            day_of_week=sel_dow,
            dep_hour=sel_hour,
            distance_miles=est_distance,
            feature_names=f_names,
        )

        st.divider()

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            # Color code status
            if user_prob < 0.20:
                status_color = COLORS["positive"]
                status_text = "Low Delay Risk (Good to Fly)"
            elif user_prob < 0.35:
                status_color = "#FFD166"
                status_text = "Moderate Delay Risk (Possible Gate Hold)"
            else:
                status_color = COLORS["negative"]
                status_text = "Elevated Delay Risk (High Congestion Likely)"

            st.metric(
                label="Predicted Delay Probability (>= 15 min)",
                value=f"{user_prob:.1%}",
                delta=f"{user_prob - base_rate:+.1%} vs Industry Average ({base_rate:.1%})",
                delta_color="inverse",
            )
            st.markdown(
                f"""
                <div style='background-color: #1E2130; padding: 1rem; border-radius: 8px; border-left: 5px solid {status_color};'>
                    <h4 style='color: {status_color}; margin: 0;'>{status_text}</h4>
                    <p style='color: #8B8D97; margin-top: 0.5rem; font-size: 0.9rem;'>
                        Flights departing {sel_origin} in the late afternoon/evening hours, particularly on Fridays or holiday months, 
                        experience compounded downstream network propagation delays.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col2:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=user_prob * 100.0,
                title={"text": "Delay Probability (%)", "font": {"size": 18, "color": "white"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white"},
                    "bar": {"color": status_color},
                    "steps": [
                        {"range": [0, 20], "color": "rgba(0, 212, 170, 0.2)"},
                        {"range": [20, 35], "color": "rgba(255, 209, 102, 0.2)"},
                        {"range": [35, 100], "color": "rgba(255, 107, 107, 0.2)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.75,
                        "value": base_rate * 100.0,
                    },
                },
            ))
            fig_gauge.update_layout(
                template=PLOTLY_TEMPLATE,
                height=260,
                margin=dict(t=30, b=10, l=30, r=30),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

    with tab_analytics:
        st.markdown("### Model Diagnostics & Receiver Operating Characteristic")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
        c2.metric("Precision", f"{metrics['Precision']:.1%}")
        c3.metric("Recall", f"{metrics['Recall']:.1%}")
        c4.metric("F1 Score", f"{metrics['F1 Score']:.3f}")
        c5.metric("AUC-ROC", f"{metrics['AUC-ROC']:.3f}")

        row_a, row_b = st.columns(2)

        with row_a:
            # ROC Curve
            fpr = eval_res["roc_curve"]["fpr"]
            tpr = eval_res["roc_curve"]["tpr"]
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"LightGBM (AUC = {metrics['AUC-ROC']:.3f})",
                line=dict(color=COLORS["primary"], width=2.5),
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random Guess",
                line=dict(color="#8B8D97", dash="dash"),
            ))
            fig_roc.update_layout(
                title="Receiver Operating Characteristic (ROC)",
                template=PLOTLY_TEMPLATE,
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=380,
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with row_b:
            # Top Features
            top_f = importances.head(10).sort_values("Importance", ascending=True)
            fig_feat = px.bar(
                top_f,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
                template=PLOTLY_TEMPLATE,
                title="Top Predictive Features (Information Gain)",
            )
            fig_feat.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig_feat, use_container_width=True)

        # Diurnal delay curve
        st.markdown("### Empirical Delay Rate by Departure Hour")
        hourly_delays = df_raw.groupby("dep_hour")["is_delayed"].mean().reset_index()
        fig_hour = px.line(
            hourly_delays,
            x="dep_hour",
            y="is_delayed",
            markers=True,
            template=PLOTLY_TEMPLATE,
            title="Flight Delay Risk Accumulation Throughout the Day",
        )
        fig_hour.update_layout(
            xaxis=dict(title="Departure Hour (0-23)", tickmode="linear", dtick=2),
            yaxis=dict(title="Empirical Delay Rate", tickformat=".0%"),
            height=320,
        )
        st.plotly_chart(fig_hour, use_container_width=True)
