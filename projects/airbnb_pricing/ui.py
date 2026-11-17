# -*- coding: utf-8 -*-
"""
Airbnb Pricing Engine -- Streamlit UI
======================================
Pure rendering layer. All computation lives in logic.py.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from common.theme import render_header, PLOTLY_TEMPLATE, COLORS
from projects.airbnb_pricing.data_loader import load_airbnb_dataset
from projects.airbnb_pricing.logic import (
    preprocess_airbnb_features,
    train_airbnb_model,
    evaluate_airbnb_model,
    compute_shap_explanations,
    predict_single_listing,
)


def render() -> None:
    """Entry point called by the master app.py router."""
    render_header(
        "Airbnb Pricing & Revenue Engine (Inside Airbnb)",
        "XGBoost fair nightly price regression with SHAP explainability: discover which features leave money on the table",
        icon="🏠",
    )

    with st.spinner("Accessing Inside Airbnb market records..."):
        df_raw, is_real, data_msg = load_airbnb_dataset(sample_size=3500)

    if not is_real and data_msg:
        st.info(f"ℹ️ **Data Pipeline Notice**: {data_msg}")
    elif is_real:
        st.success("🟢 Operating on authentic Inside Airbnb NYC listings data.")

    # ── Pipeline & Caching ───────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def _train_pipeline():
        X, y, f_names, meta = preprocess_airbnb_features(df_raw)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        mdl = train_airbnb_model(X_tr, y_tr, n_estimators=160)
        metrics, preds = evaluate_airbnb_model(mdl, X_te, y_te)
        # Sample for SHAP
        X_shap_sample = X_te.sample(min(250, len(X_te)), random_state=42)
        explainer, shap_vals, base_val = compute_shap_explanations(mdl, X_shap_sample)
        return mdl, X, y, f_names, metrics, preds, y_te, X_shap_sample, shap_vals, base_val

    with st.spinner("Fitting XGBoost Pricing Model & Computing SHAP Values..."):
        (
            model,
            X_all,
            y_all,
            f_names,
            metrics,
            preds,
            y_test,
            X_shap_sample,
            shap_values,
            base_value,
        ) = _train_pipeline()

    # ── Navigation Tabs ──────────────────────────────────────────────
    tab_model, tab_predict = st.tabs([
        "📈 Model Performance & Global SHAP",
        "🎯 Host Price Calculator & Local Breakdown",
    ])

    with tab_model:
        # Metrics row
        st.markdown("### Out-of-Sample Regression Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE ($)", f"${metrics['MAE ($)']:.2f}")
        m2.metric("RMSE ($)", f"${metrics['RMSE ($)']:.2f}")
        m3.metric("R² Score", f"{metrics['R2']:.3f}")
        m4.metric("MAPE", f"{metrics['MAPE (%)']:.1f}%")

        # Chart 1: Predicted vs Actual Scatter
        st.markdown("### Ground Truth vs. Predicted Nightly Rates ($)")
        scatter_sample = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": preds,
        }).sample(min(400, len(y_test)), random_state=42)

        fig_pred = px.scatter(
            scatter_sample,
            x="Actual",
            y="Predicted",
            opacity=0.6,
            template=PLOTLY_TEMPLATE,
            trendline="ols",
            title="Model Predictions vs. Actual Listing Prices",
        )
        fig_pred.update_layout(
            xaxis_title="Actual Nightly Rate ($)",
            yaxis_title="Predicted Nightly Rate ($)",
            height=420,
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # Chart 2: Global SHAP Summary Plot
        st.markdown("### Global SHAP Feature Importance (Beeswarm)")
        st.caption(
            "Every dot represents a listing. Color indicates high (red) vs low (blue) feature values. "
            "Points on the right increase price; points on the left decrease price."
        )

        fig_shap, ax = plt.subplots(figsize=(9, 5))
        fig_shap.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#0E1117")
        shap.summary_plot(
            shap_values,
            X_shap_sample,
            max_display=12,
            show=False,
            color_bar=True,
        )
        # Style matplotlib axes for dark mode
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.close(fig_shap)

    with tab_predict:
        st.markdown("### Calculate Optimal Nightly Price for Your Listing")
        st.caption("Adjust your listing parameters below to receive an instant ML valuation and SHAP attribution.")

        col1, col2 = st.columns(2)
        with col1:
            neighbourhoods = [
                "Manhattan - Midtown",
                "Manhattan - Financial District",
                "Manhattan - Harlem",
                "Brooklyn - Williamsburg",
                "Brooklyn - DUMBO",
                "Brooklyn - Crown Heights",
                "Queens - Astoria",
                "Queens - Long Island City",
                "Bronx - South Bronx",
                "Staten Island - St. George",
            ]
            in_nb = st.selectbox("Neighbourhood", neighbourhoods, index=0)
            in_room = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"], index=0)
            in_acc = st.slider("Accommodates (Guests)", 1, 10, 4)
            in_bed = st.slider("Bedrooms", 1, 6, 2)
            in_bath = st.slider("Bathrooms", 1.0, 4.0, 1.0, 0.5)

        with col2:
            in_dist = st.slider("Distance to City Center (km)", 0.5, 20.0, 2.5, 0.5)
            in_rating = st.slider("Review Rating", 3.0, 5.0, 4.85, 0.05)
            in_reviews = st.number_input("Number of Reviews", 0, 500, 45)
            st.markdown("**Amenities & Host Status**")
            col_a, col_b = st.columns(2)
            with col_a:
                in_wifi = st.checkbox("High-Speed WiFi", value=True)
                in_kitchen = st.checkbox("Kitchen", value=True)
            with col_b:
                in_ac = st.checkbox("Air Conditioning", value=True)
                in_super = st.checkbox("Superhost Status", value=True)

        user_spec = {
            "neighbourhood": in_nb,
            "room_type": in_room,
            "accommodates": in_acc,
            "bedrooms": in_bed,
            "bathrooms": in_bath,
            "distance_to_center_km": in_dist,
            "review_rating": in_rating,
            "number_of_reviews": in_reviews,
            "has_wifi": int(in_wifi),
            "has_kitchen": int(in_kitchen),
            "has_ac": int(in_ac),
            "is_superhost": int(in_super),
            "availability_365": 200,
            "minimum_nights": 2,
        }

        pred_price, local_shap, exp_base = predict_single_listing(model, user_spec, f_names)

        st.divider()
        c_res1, c_res2 = st.columns([1, 2])
        with c_res1:
            st.metric(
                label="Recommended Nightly Price",
                value=f"${pred_price:.2f}",
                delta=f"${pred_price - np.exp(exp_base):+.2f} vs Market Baseline",
            )
            st.markdown(
                """
                > **Host Takeaway**:
                > Pricing right at or slightly above this value optimizes revenue while preserving high booking velocity.
                """
            )

        with c_res2:
            st.markdown("#### Feature Value Impact Breakdown (Local Attribution)")
            # Top positive and negative contributors
            impact_df = pd.DataFrame({
                "Feature": f_names,
                "SHAP Effect (log scale)": local_shap,
            }).sort_values("SHAP Effect (log scale)", key=abs, ascending=False).head(8)

            fig_water = px.bar(
                impact_df,
                x="SHAP Effect (log scale)",
                y="Feature",
                orientation="h",
                color="SHAP Effect (log scale)",
                color_continuous_scale="RdYlGn",
                template=PLOTLY_TEMPLATE,
                title="Features Adding (+) or Deducting (-) Price",
            )
            fig_water.update_layout(height=320, coloraxis_showscale=False, margin=dict(t=30, b=10))
            st.plotly_chart(fig_water, use_container_width=True)
