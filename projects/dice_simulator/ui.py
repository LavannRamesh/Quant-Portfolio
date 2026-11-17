# -*- coding: utf-8 -*-
"""
Dice Game Simulator -- Streamlit UI
====================================
Pure rendering layer. All computation lives in logic.py.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from common.theme import render_header, PLOTLY_TEMPLATE, COLORS
from projects.dice_simulator.logic import simulate_dice_game, manual_expected_value


def render() -> None:
    """Entry point called by the master app.py router."""

    render_header(
        "Dice Game Simulator",
        "100 000+ Monte Carlo runs verifying Expected Value against closed-form calculation",
        icon="🎲",
    )

    # ── Sidebar controls ─────────────────────────────────────────────
    st.sidebar.markdown("### Dice Simulator Settings")
    n_sims = st.sidebar.slider(
        "Number of simulations",
        min_value=1_000,
        max_value=500_000,
        value=100_000,
        step=1_000,
    )
    seed = st.sidebar.number_input("Random seed", value=42, min_value=0)

    # ── Run simulation (cached) ──────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def _run(n: int, s: int):
        return simulate_dice_game(n, s)

    with st.spinner("Simulating..."):
        result = _run(n_sims, seed)

    # ── Analytical derivation ────────────────────────────────────────
    st.markdown("### Mathematical Derivation")
    st.latex(
        r"E[X] = \sum_{i=1}^{6} P(i) \cdot \text{payoff}(i)"
        r" = \frac{2}{6}(-1) + \frac{2}{6}(0) + \frac{1}{6}(1) + \frac{1}{6}(2)"
        r" = \frac{1}{6} \approx 0.1667"
    )

    # ── Metric cards ─────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Analytical E[X]", f"{result['analytical_ev']:.5f}")
    c2.metric("Simulated E[X]", f"{result['simulated_ev']:.5f}")
    c3.metric("Error %", f"{result['error_pct']:.4f}%")
    c4.metric("Simulations", f"{result['n_simulations']:,}")

    # ── Chart 1: Convergence ─────────────────────────────────────────
    st.markdown("### Convergence of Simulated Mean")
    cm = result["cumulative_mean"]
    # Subsample for performance (max 2000 points)
    step = max(1, len(cm) // 2000)
    x = np.arange(1, len(cm) + 1)[::step]
    y = cm[::step]

    fig_conv = go.Figure()
    fig_conv.add_trace(
        go.Scatter(x=x, y=y, mode="lines", name="Simulated Mean",
                   line=dict(color=COLORS["primary"], width=2))
    )
    fig_conv.add_hline(
        y=result["analytical_ev"],
        line_dash="dash",
        line_color=COLORS["secondary"],
        annotation_text=f"Analytical E[X] = {result['analytical_ev']:.5f}",
    )
    fig_conv.update_layout(
        template=PLOTLY_TEMPLATE,
        xaxis_title="Number of Simulations",
        yaxis_title="Cumulative Mean Payoff ($)",
        height=420,
        margin=dict(t=30),
    )
    st.plotly_chart(fig_conv, use_container_width=True)

    # ── Chart 2: Outcome histogram ───────────────────────────────────
    st.markdown("### Outcome Distribution")
    hist = result["histogram"]
    payoff_labels = sorted(hist.keys())
    counts = [hist[k] for k in payoff_labels]

    colour_map = {
        -1: COLORS["negative"],
        0: COLORS["text_muted"],
        1: COLORS["primary"],
        2: COLORS["tertiary"],
    }
    bar_colours = [colour_map.get(p, COLORS["primary"]) for p in payoff_labels]

    fig_hist = go.Figure(
        go.Bar(
            x=[f"${p:+d}" for p in payoff_labels],
            y=counts,
            marker_color=bar_colours,
            text=[f"{c / n_sims * 100:.1f}%" for c in counts],
            textposition="outside",
        )
    )
    fig_hist.update_layout(
        template=PLOTLY_TEMPLATE,
        xaxis_title="Payoff",
        yaxis_title="Count",
        height=380,
        margin=dict(t=30),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Payoff table ─────────────────────────────────────────────────
    with st.expander("Payoff Rules"):
        st.markdown(
            """
            | Die Roll | Payoff |
            |----------|--------|
            | 1 or 2   | -$1    |
            | 3 or 4   |  $0    |
            | 5        | +$1    |
            | 6        | +$2    |
            """
        )
