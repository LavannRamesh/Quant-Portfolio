# -*- coding: utf-8 -*-
"""
Kelly Criterion Bet Sizing Simulator -- Streamlit UI
=====================================================
Pure rendering layer. All computation lives in logic.py.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from common.theme import render_header, PLOTLY_TEMPLATE, COLORS
from projects.kelly_criterion.logic import kelly_fraction, simulate_kelly_paths


# Colour palette for different fractions
FRAC_COLORS = ["#636EFA", "#00CC96", "#AB63FA", "#FFA15A", "#EF553B", "#19D3F3"]


def render() -> None:
    """Entry point called by the master app.py router."""

    render_header(
        "Kelly Criterion Simulator",
        "Optimal bet sizing -- why over-betting destroys wealth even with an edge",
        icon="\U0001f4b0",
    )

    # ── Sidebar controls ─────────────────────────────────────────────
    st.sidebar.markdown("### Kelly Settings")
    win_prob = st.sidebar.slider("Win probability", 0.50, 0.90, 0.60, 0.01)
    odds = st.sidebar.slider("Net odds (b)", 0.5, 5.0, 1.0, 0.1)
    n_bets = st.sidebar.slider("Bets per path", 50, 1000, 500, 50)
    n_paths = st.sidebar.slider("Number of paths", 100, 5000, 1000, 100)

    f_star = kelly_fraction(win_prob, odds)

    # ── Kelly formula display ────────────────────────────────────────
    st.markdown("### The Kelly Formula")
    st.latex(
        r"f^* = \frac{b \cdot p - q}{b}"
        r" = \frac{"
        + f"{odds:.1f}"
        + r" \times "
        + f"{win_prob:.2f}"
        + r" - "
        + f"{1 - win_prob:.2f}"
        + r"}{"
        + f"{odds:.1f}"
        + r"}"
        + f" = {f_star:.4f}"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal Kelly f*", f"{f_star:.2%}")
    c2.metric("Win Probability", f"{win_prob:.0%}")
    c3.metric("Net Odds", f"{odds:.1f}:1")

    # ── Run simulation ───────────────────────────────────────────────
    fractions = sorted(set([0.05, 0.20, round(f_star, 4), 0.60, 1.00]))

    @st.cache_data(show_spinner=False)
    def _simulate(wp, o, fracs, nb, np_):
        return simulate_kelly_paths(wp, o, list(fracs), nb, np_)

    with st.spinner("Simulating wealth paths..."):
        result = _simulate(win_prob, odds, tuple(fractions), n_bets, n_paths)

    # ── Chart 1: Median wealth paths with confidence bands ───────────
    st.markdown("### Wealth Trajectories (Log Scale)")
    fig_paths = go.Figure()
    x_axis = np.arange(n_bets + 1)

    for i, frac in enumerate(result["fractions"]):
        data = result["fraction_results"][frac]
        color = FRAC_COLORS[i % len(FRAC_COLORS)]
        is_kelly = abs(frac - f_star) < 1e-6
        label = f"f={frac:.0%}" + (" (Kelly*)" if is_kelly else "")

        # 5th-95th percentile band
        fig_paths.add_trace(
            go.Scatter(
                x=np.concatenate([x_axis, x_axis[::-1]]),
                y=np.concatenate([data["p95_path"], data["p5_path"][::-1]]),
                fill="toself",
                fillcolor=color.replace(")", ", 0.1)").replace("rgb", "rgba")
                if "rgb" in color
                else color + "1A",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Median path
        fig_paths.add_trace(
            go.Scatter(
                x=x_axis,
                y=data["median_path"],
                mode="lines",
                name=label,
                line=dict(
                    color=color,
                    width=3 if is_kelly else 1.5,
                    dash="solid" if is_kelly else "solid",
                ),
            )
        )

    fig_paths.update_layout(
        template=PLOTLY_TEMPLATE,
        yaxis_type="log",
        yaxis_title="Bankroll ($, log scale)",
        xaxis_title="Bet Number",
        height=500,
        margin=dict(t=30),
        legend=dict(x=0.02, y=0.98),
    )
    fig_paths.add_hline(y=1.0, line_dash="dot", line_color="red",
                        annotation_text="Ruin threshold ($1)")
    st.plotly_chart(fig_paths, use_container_width=True)

    # ── Chart 2: Terminal wealth box plots ────────────────────────────
    st.markdown("### Terminal Wealth Distribution")
    fig_box = go.Figure()
    for i, frac in enumerate(result["fractions"]):
        data = result["fraction_results"][frac]
        # Clip for log display
        tw = np.clip(data["terminal_wealth"], 1e-2, None)
        is_kelly = abs(frac - f_star) < 1e-6
        fig_box.add_trace(
            go.Box(
                y=np.log10(tw),
                name=f"f={frac:.0%}" + ("*" if is_kelly else ""),
                marker_color=FRAC_COLORS[i % len(FRAC_COLORS)],
            )
        )
    fig_box.update_layout(
        template=PLOTLY_TEMPLATE,
        yaxis_title="log10(Terminal Wealth)",
        height=400,
        margin=dict(t=30),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # ── Metrics table ────────────────────────────────────────────────
    st.markdown("### Summary Metrics")
    rows = []
    for frac in result["fractions"]:
        data = result["fraction_results"][frac]
        is_kelly = abs(frac - f_star) < 1e-6
        rows.append(
            {
                "Fraction": f"{frac:.0%}" + (" (Kelly*)" if is_kelly else ""),
                "Median Terminal $": f"${data['median_terminal']:,.2f}",
                "Ruin Probability": f"{data['ruin_prob']:.1%}",
                "Growth Rate": f"{data['growth_rate']:.5f}",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Insight box ──────────────────────────────────────────────────
    st.info(
        "**Key insight:** The Kelly fraction maximises long-run geometric growth. "
        "Betting more than Kelly (e.g. 60% or 100%) dramatically increases ruin "
        "probability even when you have a genuine edge. In practice, many "
        "professionals use 'Half-Kelly' (f*/2) for a smoother ride."
    )
