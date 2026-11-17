"""
common/theme.py
===============
Shared Streamlit theming, CSS injection, and layout constants for the
Quant & Data Science Master Portfolio.
"""

import streamlit as st

# ── Colour palette (dark theme) ──────────────────────────────────────
COLORS = {
    "primary": "#00D4AA",       # Teal accent
    "secondary": "#FF6B6B",     # Coral for warnings / losses
    "tertiary": "#4ECDC4",      # Light teal
    "background": "#0E1117",    # Streamlit dark bg
    "surface": "#1E2130",       # Card surface
    "text": "#FAFAFA",          # Primary text
    "text_muted": "#8B8D97",    # Muted text
    "positive": "#00D4AA",      # Green / profit
    "negative": "#FF6B6B",      # Red / loss
}

# ── Plotly template ──────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"

# ── Category icons ───────────────────────────────────────────────────
CATEGORIES = {
    "🎲 Dice Simulator": "dice_simulator",
    "💰 Kelly Criterion": "kelly_criterion",
    "📈 Moving Avg Backtest": "moving_average_backtest",
    "🧮 Options Pricer": "options_pricer",
    "📊 Stock Dashboard": "stock_dashboard",
    "🚗 Traffic Forecaster": "traffic_forecaster",
    "🏠 Airbnb Pricing": "airbnb_pricing",
    "✈️ Flight Delay Predictor": "flight_delay",
    "🌊 Wave Equation": "wave_equation",
    "🧠 Neural Net from Scratch": "neural_net_scratch",
}


def inject_css():
    """Inject custom CSS for a polished dark-theme look."""
    st.markdown(
        """
        <style>
        /* ── Global ────────────────────────────────────── */
        .main .block-container {
            padding-top: 2rem;
            max-width: 1200px;
        }

        /* ── Sidebar ───────────────────────────────────── */
        [data-testid="stSidebar"] {
            background-color: #161B22;
        }
        [data-testid="stSidebar"] .css-1d391kg {
            padding-top: 1rem;
        }

        /* ── Metric cards ──────────────────────────────── */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 700;
        }
        [data-testid="stMetricDelta"] > div {
            font-size: 0.9rem;
        }

        /* ── Headers ───────────────────────────────────── */
        h1, h2, h3 {
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }

        /* ── Code blocks ───────────────────────────────── */
        .stCodeBlock {
            border-radius: 8px;
        }

        /* ── Dividers ──────────────────────────────────── */
        hr {
            border-color: #30363D;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, description: str, icon: str = "📊"):
    """Render a consistent project header with title and description."""
    st.markdown(f"# {icon} {title}")
    st.markdown(f"*{description}*")
    st.divider()
