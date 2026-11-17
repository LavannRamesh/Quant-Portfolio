"""
app.py — Quant & Data Science Master Portfolio
================================================
A unified Streamlit web portal containing 10 projects spanning
quantitative finance, machine learning, and mathematical computing.

Launch:  streamlit run app.py
"""

import streamlit as st
import sys
import os

# Ensure project root is on the path so sub-packages resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.theme import inject_css, COLORS

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quant & Data Science Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
inject_css()

# ── Sidebar navigation ──────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div style='text-align: center; padding: 1rem 0;'>
        <h2 style='margin: 0; color: #00D4AA;'>📊 Quant Portfolio</h2>
        <p style='color: #8B8D97; font-size: 0.85rem; margin-top: 0.3rem;'>
            Data Science • ML • Mathematics
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()

# Navigation categories
PAGES = {
    "🏠 Home": "home",
    "── Quant Finance ──": None,  # Section header
    "🎲 Dice Game Simulator": "dice_simulator",
    "💰 Kelly Criterion": "kelly_criterion",
    "📈 Moving Avg Backtest": "moving_average_backtest",
    "🧮 Options Pricer": "options_pricer",
    "📊 Stock Dashboard": "stock_dashboard",
    "── Machine Learning ──": None,  # Section header
    "🚗 Traffic Forecaster": "traffic_forecaster",
    "🏠 Airbnb Pricing": "airbnb_pricing",
    "✈️ Flight Delay Predictor": "flight_delay",
    "── Mathematics ──": None,  # Section header
    "🌊 Wave Equation": "wave_equation",
    "🧠 Neural Net from Scratch": "neural_net_scratch",
}

# Build sidebar radio (skip section headers)
selectable = [k for k, v in PAGES.items() if v is not None]

# Add section headers as disabled markdown
for label, module in PAGES.items():
    if module is None:
        st.sidebar.markdown(
            f"<p style='color: #8B8D97; font-size: 0.75rem; "
            f"font-weight: 600; margin: 1rem 0 0.3rem 0; "
            f"text-transform: uppercase; letter-spacing: 0.05em;'>{label}</p>",
            unsafe_allow_html=True,
        )

selected = st.sidebar.radio(
    "Navigate",
    selectable,
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown(
    "<p style='color: #8B8D97; font-size: 0.75rem; text-align: center;'>"
    "Built with ❤️ using Streamlit & Python</p>",
    unsafe_allow_html=True,
)

# ── Page routing ─────────────────────────────────────────────────────
page_key = PAGES.get(selected, "home")

if page_key == "home":
    # ── Home / Landing Page ──────────────────────────────────────────
    st.markdown(
        """
        # 📊 Quant & Data Science Master Portfolio

        Welcome to a unified collection of **10 production-quality projects** spanning
        quantitative finance, machine learning, and mathematical computing.

        ---

        ### 🎯 What's Inside

        | Category | Projects | Stack |
        |----------|----------|-------|
        | **Quant Finance** | Dice Simulator, Kelly Criterion, MA Backtest, Options Pricer, Stock Dashboard | NumPy, Pandas, Plotly, yfinance |
        | **Machine Learning** | Traffic Forecaster, Airbnb Pricing, Flight Delay Predictor | XGBoost, LightGBM, SHAP, scikit-learn |
        | **Mathematics** | 2D Wave Equation, Neural Network from Scratch | NumPy, Matplotlib, Plotly |

        ---

        ### 🚀 Getting Started

        👈 **Select a project from the sidebar** to explore interactive simulations,
        real-time dashboards, and ML-powered predictions.

        ---

        ### ✨ Highlights

        - **Mathematically rigorous** — every formula is derived, documented, and validated
        - **Look-ahead bias prevention** — backtests use `.shift(1)` to avoid peeking at future data
        - **Monte Carlo validation** — Black-Scholes prices are verified against 1M simulation paths
        - **SHAP explanations** — ML models include visual feature importance analysis
        - **Interactive visualizations** — Plotly charts with hover, zoom, and real-time parameter tuning
        """
    )

    # Feature cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Projects", "10", delta="3 categories")
    with col2:
        st.metric("Simulations", "1M+", delta="Monte Carlo paths")
    with col3:
        st.metric("ML Models", "3", delta="with SHAP explanations")

elif page_key == "dice_simulator":
    from projects.dice_simulator.ui import render
    render()

elif page_key == "kelly_criterion":
    from projects.kelly_criterion.ui import render
    render()

elif page_key == "moving_average_backtest":
    from projects.moving_average_backtest.ui import render
    render()

elif page_key == "options_pricer":
    from projects.options_pricer.ui import render
    render()

elif page_key == "stock_dashboard":
    from projects.stock_dashboard.ui import render
    render()

elif page_key == "traffic_forecaster":
    from projects.traffic_forecaster.ui import render
    render()

elif page_key == "airbnb_pricing":
    from projects.airbnb_pricing.ui import render
    render()

elif page_key == "flight_delay":
    from projects.flight_delay.ui import render
    render()

elif page_key == "wave_equation":
    from projects.wave_equation.ui import render
    render()

elif page_key == "neural_net_scratch":
    from projects.neural_net_scratch.ui import render
    render()
