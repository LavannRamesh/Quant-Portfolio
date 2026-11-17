# -*- coding: utf-8 -*-
"""
Moving Average Backtest -- Streamlit UI
========================================
Pure rendering layer. All computation lives in logic.py.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common.theme import render_header, PLOTLY_TEMPLATE, COLORS
from projects.moving_average_backtest.logic import generate_sample_data, run_backtest


def render() -> None:
    """Entry point called by the master app.py router."""

    render_header(
        "Moving Average Crossover Backtest",
        "Signal generation with look-ahead bias prevention, transaction costs, and risk metrics",
        icon="\U0001f4c8",
    )

    # ── Look-ahead bias warning ──────────────────────────────────────
    st.warning(
        "**Look-Ahead Bias Prevention:**  The trading signal is computed from "
        "bar *t*'s close prices, but the position is only entered at bar *t+1* "
        "via `position = signal.shift(1)`.  This ensures the backtest never "
        "uses information that wouldn't have been available in real-time."
    )
    with st.expander("See the critical code"):
        st.code(
            """
# Signal: 1 when fast SMA > slow SMA (computed from today's close)
signal = (sma_fast > sma_slow).astype(int)

# CRITICAL: shift by 1 day so we trade TOMORROW, not today
position = signal.shift(1)   # <-- PREVENTS LOOK-AHEAD BIAS
            """,
            language="python",
        )

    # ── Sidebar controls ─────────────────────────────────────────────
    st.sidebar.markdown("### Backtest Settings")
    fast_w = st.sidebar.slider("Fast SMA window", 5, 50, 20)
    slow_w = st.sidebar.slider("Slow SMA window", 20, 200, 50)
    cost_bps = st.sidebar.slider("Transaction cost (bps)", 0, 50, 10)

    if fast_w >= slow_w:
        st.error("Fast window must be smaller than slow window.")
        return

    # ── Generate data & run backtest ─────────────────────────────────
    @st.cache_data(show_spinner=False)
    def _backtest(fw, sw, cost):
        df = generate_sample_data()
        return run_backtest(df, fw, sw, cost)

    with st.spinner("Running backtest..."):
        result = _backtest(fast_w, slow_w, cost_bps)

    data = result["data"]
    m = result["metrics"]

    # ── Metrics row ──────────────────────────────────────────────────
    st.markdown("### Performance Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Strategy Sharpe",
        f"{m['strategy_sharpe']:.3f}",
        delta=f"vs {m['benchmark_sharpe']:.3f} (B&H)",
    )
    c2.metric("Strategy Max DD", f"{m['strategy_max_dd']:.2%}")
    c3.metric("Strategy CAGR", f"{m['strategy_cagr']:.2%}")
    c4.metric("Win Rate", f"{m['win_rate']:.1%}")
    c5.metric("Total Trades", f"{m['total_trades']}")

    # ── Chart 1: Price + SMAs ────────────────────────────────────────
    st.markdown("### Price Chart with SMA Overlays")
    valid = data.dropna(subset=["SMA_Slow"])

    fig_price = go.Figure()
    fig_price.add_trace(
        go.Scatter(x=valid["Date"], y=valid["Close"], name="Close",
                   line=dict(color=COLORS["text_muted"], width=1))
    )
    fig_price.add_trace(
        go.Scatter(x=valid["Date"], y=valid["SMA_Fast"],
                   name=f"SMA {fast_w}", line=dict(color=COLORS["primary"], width=2))
    )
    fig_price.add_trace(
        go.Scatter(x=valid["Date"], y=valid["SMA_Slow"],
                   name=f"SMA {slow_w}", line=dict(color=COLORS["secondary"], width=2))
    )
    fig_price.update_layout(
        template=PLOTLY_TEMPLATE, height=400, margin=dict(t=30),
        xaxis_title="Date", yaxis_title="Price ($)",
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # ── Chart 2: Equity curves ───────────────────────────────────────
    st.markdown("### Equity Curves")
    fig_eq = go.Figure()
    fig_eq.add_trace(
        go.Scatter(x=data["Date"], y=data["Strategy_Equity"],
                   name="Strategy", line=dict(color=COLORS["primary"], width=2))
    )
    fig_eq.add_trace(
        go.Scatter(x=data["Date"], y=data["Benchmark_Equity"],
                   name="Buy & Hold", line=dict(color=COLORS["text_muted"], width=2))
    )
    fig_eq.update_layout(
        template=PLOTLY_TEMPLATE, height=380, margin=dict(t=30),
        xaxis_title="Date", yaxis_title="Equity ($1 start)",
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # ── Chart 3: Drawdown ────────────────────────────────────────────
    st.markdown("### Drawdown")
    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(
            x=data["Date"], y=data["Strategy_Drawdown"],
            name="Strategy DD", fill="tozeroy",
            line=dict(color=COLORS["secondary"], width=1),
            fillcolor="rgba(255, 107, 107, 0.3)",
        )
    )
    fig_dd.add_trace(
        go.Scatter(
            x=data["Date"], y=data["Benchmark_Drawdown"],
            name="B&H DD",
            line=dict(color=COLORS["text_muted"], width=1, dash="dot"),
        )
    )
    fig_dd.update_layout(
        template=PLOTLY_TEMPLATE, height=300, margin=dict(t=30),
        xaxis_title="Date", yaxis_title="Drawdown",
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # ── Comparison table ─────────────────────────────────────────────
    st.markdown("### Strategy vs Benchmark")
    import pandas as pd
    comp = pd.DataFrame({
        "Metric": ["Total Return", "CAGR", "Sharpe Ratio", "Max Drawdown"],
        "Strategy": [
            f"{m['strategy_total_return']:.2%}",
            f"{m['strategy_cagr']:.2%}",
            f"{m['strategy_sharpe']:.3f}",
            f"{m['strategy_max_dd']:.2%}",
        ],
        "Buy & Hold": [
            f"{m['benchmark_total_return']:.2%}",
            f"{m['benchmark_cagr']:.2%}",
            f"{m['benchmark_sharpe']:.3f}",
            f"{m['benchmark_max_dd']:.2%}",
        ],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)
