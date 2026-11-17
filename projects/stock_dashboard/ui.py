# -*- coding: utf-8 -*-
"""
Stock Returns & Volatility Dashboard -- Streamlit UI
=====================================================
Pure rendering layer. All computation lives in logic.py.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from common.theme import render_header, PLOTLY_TEMPLATE, COLORS
from projects.stock_dashboard.logic import (
    load_stock_data,
    compute_daily_returns,
    compute_normalized_prices,
    compute_rolling_volatility,
    compute_drawdowns,
    compute_portfolio_metrics,
    compute_correlation_matrix,
)


def render() -> None:
    """Entry point called by the master app.py router."""
    render_header(
        "Stock Returns & Volatility Dashboard",
        "Multi-asset price dynamics, rolling risk metrics, drawdowns, and cross-asset correlations",
        icon="📊",
    )

    # ── Sidebar Controls ─────────────────────────────────────────────
    st.sidebar.markdown("### Portfolio Configuration")
    default_tickers = ["SPY", "AAPL", "MSFT", "GOOGL", "TSLA"]
    selected_tickers = st.sidebar.multiselect(
        "Select Assets",
        options=["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B"],
        default=default_tickers,
    )
    if not selected_tickers:
        st.warning("Please select at least one ticker.")
        return

    period = st.sidebar.selectbox("Historical Window", ["1y", "2y", "5y"], index=1)
    rolling_window = st.sidebar.slider("Rolling Volatility Window (Days)", 10, 63, 21)

    # ── Load Data ────────────────────────────────────────────────────
    with st.spinner("Fetching market time-series..."):
        prices, is_real, warning_msg = load_stock_data(selected_tickers, period=period)

    if not is_real and warning_msg:
        st.info(f"ℹ️ **Data Notice**: {warning_msg}")
    elif is_real:
        st.success("🟢 Connected live to Yahoo Finance (real market data).")

    returns = compute_daily_returns(prices)
    normalized = compute_normalized_prices(prices)
    rolling_vol = compute_rolling_volatility(returns, window=rolling_window)
    drawdowns = compute_drawdowns(prices)
    metrics_df = compute_portfolio_metrics(returns, prices)
    corr_matrix = compute_correlation_matrix(returns)

    # ── Top Metric Cards ─────────────────────────────────────────────
    st.markdown("### Executive Risk & Return Summary")
    cols = st.columns(len(selected_tickers))
    for idx, ticker in enumerate(selected_tickers):
        with cols[idx % len(cols)]:
            cagr = metrics_df.loc[ticker, "CAGR"]
            vol = metrics_df.loc[ticker, "Ann. Vol"]
            sharpe = metrics_df.loc[ticker, "Sharpe (rf=2%)"]
            st.metric(
                label=ticker,
                value=f"{cagr:.1%}",
                delta=f"Sharpe: {sharpe:.2f} | Vol: {vol:.1%}",
            )

    # ── Tabs for Visualizations ──────────────────────────────────────
    tab_price, tab_risk, tab_corr, tab_dist = st.tabs([
        "📈 Performance & Drawdown",
        "⚡ Volatility Dynamics",
        "🔗 Correlation Structure",
        "📊 Return Distributions",
    ])

    with tab_price:
        # Chart 1: Normalized Performance
        fig_norm = go.Figure()
        for col in normalized.columns:
            fig_norm.add_trace(go.Scatter(
                x=normalized.index, y=normalized[col], mode="lines", name=col
            ))
        fig_norm.update_layout(
            title="Normalized Asset Growth (Base = 100)",
            template=PLOTLY_TEMPLATE,
            xaxis_title="Date",
            yaxis_title="Normalized Value",
            height=420,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_norm, use_container_width=True)

        # Chart 2: Underwater Drawdown
        fig_dd = go.Figure()
        for col in drawdowns.columns:
            fig_dd.add_trace(go.Scatter(
                x=drawdowns.index,
                y=drawdowns[col],
                mode="lines",
                name=f"{col} DD",
                fill="tozeroy",
            ))
        fig_dd.update_layout(
            title="Underwater Drawdowns (Peak-to-Trough)",
            template=PLOTLY_TEMPLATE,
            xaxis_title="Date",
            yaxis_title="Drawdown",
            yaxis=dict(tickformat=".1%"),
            height=350,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    with tab_risk:
        # Chart 3: Rolling Volatility
        fig_vol = go.Figure()
        for col in rolling_vol.columns:
            fig_vol.add_trace(go.Scatter(
                x=rolling_vol.index, y=rolling_vol[col], mode="lines", name=f"{col} {rolling_window}d Vol"
            ))
        fig_vol.update_layout(
            title=f"Annualized Rolling Volatility ({rolling_window}-Day Window)",
            template=PLOTLY_TEMPLATE,
            xaxis_title="Date",
            yaxis_title="Annualized Volatility",
            yaxis=dict(tickformat=".1%"),
            height=400,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        # Chart 4: Risk vs Return Scatter
        scatter_df = metrics_df.reset_index()
        fig_scatter = px.scatter(
            scatter_df,
            x="Ann. Vol",
            y="CAGR",
            text="Ticker",
            size=scatter_df["Sharpe (rf=2%)"].apply(lambda x: max(float(x), 0.1) * 20),
            color="Sharpe (rf=2%)",
            color_continuous_scale="Viridis",
            title="Risk vs. Return Frontier (Bubble size = Sharpe Ratio)",
            template=PLOTLY_TEMPLATE,
        )
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(
            xaxis_title="Annualized Volatility (Risk)",
            yaxis_title="CAGR (Expected Return)",
            xaxis=dict(tickformat=".1%"),
            yaxis=dict(tickformat=".1%"),
            height=450,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab_corr:
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Pairwise Asset Return Correlation Matrix",
            template=PLOTLY_TEMPLATE,
        )
        fig_corr.update_layout(height=450)
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab_dist:
        st.markdown("### Daily Return Distributions & Higher Moments")
        fig_hist = go.Figure()
        for col in returns.columns:
            fig_hist.add_trace(go.Histogram(
                x=returns[col],
                name=col,
                opacity=0.6,
                nbinsx=60,
            ))
        fig_hist.update_layout(
            barmode="overlay",
            title="Historical Return Histograms",
            template=PLOTLY_TEMPLATE,
            xaxis_title="Daily Return",
            xaxis=dict(tickformat=".1%"),
            yaxis_title="Frequency",
            height=380,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Detailed metrics table
        display_metrics = metrics_df.copy()
        display_metrics["CAGR"] = display_metrics["CAGR"].apply(lambda x: f"{x:.2%}")
        display_metrics["Ann. Vol"] = display_metrics["Ann. Vol"].apply(lambda x: f"{x:.2%}")
        display_metrics["Sharpe (rf=2%)"] = display_metrics["Sharpe (rf=2%)"].apply(lambda x: f"{x:.2f}")
        display_metrics["Max Drawdown"] = display_metrics["Max Drawdown"].apply(lambda x: f"{x:.2%}")
        display_metrics["Skewness"] = display_metrics["Skewness"].apply(lambda x: f"{x:.2f}")
        display_metrics["Kurtosis"] = display_metrics["Kurtosis"].apply(lambda x: f"{x:.2f}")
        st.dataframe(display_metrics, use_container_width=True)
