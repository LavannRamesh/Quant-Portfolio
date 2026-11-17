# -*- coding: utf-8 -*-
"""
Options Pricer -- Streamlit UI
===============================
Pure rendering layer. All computation lives in logic.py.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from common.theme import render_header, PLOTLY_TEMPLATE, COLORS
from projects.options_pricer.logic import (
    black_scholes,
    monte_carlo_price,
    price_sensitivity_surface,
)


def render() -> None:
    """Entry point called by the master app.py router."""

    render_header(
        "Options Pricer",
        "Black-Scholes closed-form pricing validated with 1M Monte Carlo paths",
        icon="\U0001f9ee",
    )

    # ── Sidebar controls ─────────────────────────────────────────────
    st.sidebar.markdown("### Options Parameters")
    S = st.sidebar.number_input("Spot price (S)", 10.0, 1000.0, 100.0, 5.0)
    K = st.sidebar.number_input("Strike price (K)", 10.0, 1000.0, 100.0, 5.0)
    T = st.sidebar.slider("Time to expiry (years)", 0.1, 3.0, 1.0, 0.1)
    r = st.sidebar.slider("Risk-free rate", 0.0, 0.15, 0.05, 0.005)
    sigma = st.sidebar.slider("Volatility (sigma)", 0.05, 1.0, 0.20, 0.01)
    option_type = st.sidebar.selectbox("Option type", ["call", "put"])
    n_mc = st.sidebar.select_slider(
        "MC paths", options=[100_000, 250_000, 500_000, 1_000_000], value=500_000
    )

    # ── Black-Scholes calculation ────────────────────────────────────
    bs = black_scholes(S, K, T, r, sigma, option_type)

    # ── Monte Carlo calculation (cached) ─────────────────────────────
    @st.cache_data(show_spinner=False)
    def _mc(s, k, t, rf, sig, n, ot):
        return monte_carlo_price(s, k, t, rf, sig, n, ot)

    with st.spinner("Running Monte Carlo simulation..."):
        mc = _mc(S, K, T, r, sigma, n_mc, option_type)

    error_abs = abs(mc["price"] - bs["price"])
    error_pct = error_abs / max(bs["price"], 1e-8) * 100

    # ── Formulas ─────────────────────────────────────────────────────
    st.markdown("### Black-Scholes Formula")
    if option_type == "call":
        st.latex(r"C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)")
    else:
        st.latex(r"P = K \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)")
    st.latex(
        r"d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad"
        r"d_2 = d_1 - \sigma\sqrt{T}"
    )

    # ── Price comparison ─────────────────────────────────────────────
    st.markdown("### Price Comparison")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BS Price", f"${bs['price']:.4f}")
    c2.metric("MC Price", f"${mc['price']:.4f}")
    c3.metric("Abs Error", f"${error_abs:.4f}")
    c4.metric("Rel Error", f"{error_pct:.3f}%")

    # ── Greeks ───────────────────────────────────────────────────────
    st.markdown("### Greeks")
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Delta", f"{bs['delta']:.4f}")
    g2.metric("Gamma", f"{bs['gamma']:.6f}")
    g3.metric("Theta (daily)", f"{bs['theta']:.4f}")
    g4.metric("Vega (1%)", f"{bs['vega']:.4f}")
    g5.metric("Rho (1%)", f"{bs['rho']:.4f}")

    # ── Chart 1: MC Convergence ──────────────────────────────────────
    st.markdown("### Monte Carlo Convergence")
    conv = mc["convergence"]

    fig_conv = go.Figure()
    fig_conv.add_trace(
        go.Scatter(
            x=conv["n_paths"],
            y=conv["mc_price"] + 2 * conv["std_error"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig_conv.add_trace(
        go.Scatter(
            x=conv["n_paths"],
            y=conv["mc_price"] - 2 * conv["std_error"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0, 212, 170, 0.15)",
            name="+/- 2 SE",
        )
    )
    fig_conv.add_trace(
        go.Scatter(
            x=conv["n_paths"],
            y=conv["mc_price"],
            mode="lines",
            name="MC Price",
            line=dict(color=COLORS["primary"], width=2),
        )
    )
    fig_conv.add_hline(
        y=bs["price"],
        line_dash="dash",
        line_color=COLORS["secondary"],
        annotation_text=f"BS = ${bs['price']:.4f}",
    )
    fig_conv.update_layout(
        template=PLOTLY_TEMPLATE,
        xaxis_title="Number of Paths",
        yaxis_title=f"{option_type.title()} Price ($)",
        height=450,
        margin=dict(t=30),
    )
    st.plotly_chart(fig_conv, use_container_width=True)

    # ── Chart 2: Price sensitivity surface ───────────────────────────
    st.markdown("### Price Sensitivity (Spot vs Volatility)")

    @st.cache_data(show_spinner=False)
    def _surface(k, t, rf, ot):
        return price_sensitivity_surface(k, t, rf, option_type=ot)

    with st.spinner("Computing surface..."):
        surf = _surface(K, T, r, option_type)

    fig_surf = go.Figure(
        data=[
            go.Surface(
                x=surf["S"][0],
                y=surf["sigma"][:, 0],
                z=surf["prices"],
                colorscale="Viridis",
                colorbar=dict(title="Price ($)"),
            )
        ]
    )
    fig_surf.update_layout(
        template=PLOTLY_TEMPLATE,
        scene=dict(
            xaxis_title="Spot Price (S)",
            yaxis_title="Volatility (sigma)",
            zaxis_title=f"{option_type.title()} Price ($)",
        ),
        height=550,
        margin=dict(t=30, l=0, r=0),
    )
    st.plotly_chart(fig_surf, use_container_width=True)

    st.info(
        "**Convergence rate:** Monte Carlo converges at O(1/sqrt(N)) by the "
        "Central Limit Theorem. Doubling accuracy requires 4x the paths."
    )
