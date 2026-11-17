# 📊 Quant & Data Science Master Portfolio

A unified, professional **Streamlit web portal** containing 10 production-quality projects
spanning quantitative finance, machine learning, and mathematical computing.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Quick Start

### Option 1: One-Click Launch

**Windows:**
```bash
double-click run.bat
```

**Mac:**
```bash
chmod +x run.command && ./run.command
```

### Option 2: Manual

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Option 3: Docker

```bash
docker-compose up --build
```

Then open [http://localhost:8501](http://localhost:8501).

---

## 📁 Projects

### Category 1: Quant Finance & Simulation

| # | Project | Description | Stack |
|---|---------|-------------|-------|
| 1 | **🎲 Dice Game Simulator** | 100K simulated runs verifying E[X] against manual calculation | NumPy |
| 2 | **💰 Kelly Criterion** | Optimal bet sizing — simulate the ruin of over-betting | NumPy |
| 3 | **📈 Moving Avg Backtest** | Signal generation with look-ahead bias prevention, Sharpe, drawdown | Pandas, NumPy |
| 4 | **🧮 Options Pricer** | Black-Scholes validated with 1M Monte Carlo paths | SciPy, NumPy |
| 5 | **📊 Stock Dashboard** | Real-time rolling volatility, drawdowns, risk vs return | yfinance, Plotly |

### Category 2: Machine Learning & Forecasting

| # | Project | Description | Stack |
|---|---------|-------------|-------|
| 6 | **🚗 Traffic Forecaster** | Predict highway speeds 15-60 min ahead | LightGBM |
| 7 | **🏠 Airbnb Pricing** | Predict nightly price + SHAP explainability | XGBoost, SHAP |
| 8 | **✈️ Flight Delay** | Predict P(delay > 15 min) by airline/airport/time | LightGBM |

### Category 3: Python Mathematics

| # | Project | Description | Stack |
|---|---------|-------------|-------|
| 9 | **🌊 Wave Equation** | 2D damped wave equation via finite differences, 3D animated | NumPy, Plotly |
| 10 | **🧠 Neural Net from Scratch** | Pure NumPy neural net with 3D loss landscape | NumPy, Matplotlib |

---

## 🏗️ Architecture

```
├── app.py                    ← Master Streamlit portal with sidebar navigation
├── requirements.txt
├── run.bat / run.command     ← One-click launchers
├── Dockerfile / docker-compose.yml
├── common/
│   ├── theme.py              ← Shared theming & CSS
│   └── utils.py              ← Shared utilities (Sharpe, drawdown, etc.)
├── projects/
│   ├── dice_simulator/       ← logic.py + ui.py + README.md
│   ├── kelly_criterion/
│   ├── moving_average_backtest/
│   ├── options_pricer/
│   ├── stock_dashboard/
│   ├── traffic_forecaster/
│   ├── airbnb_pricing/
│   ├── flight_delay/
│   ├── wave_equation/
│   └── neural_net_scratch/
└── data/                     ← Auto-generated synthetic data
```

---

## ✨ Key Technical Highlights

- **No look-ahead bias** — Backtest signals use `.shift(1)` to prevent peeking at future data
- **Monte Carlo validation** — Options prices verified: BS vs 1M MC paths converge within 0.5%
- **CFL stability** — Wave equation solver asserts Courant number ≤ 1/√2 before running
- **Gradient checking** — Neural network validates analytic gradients against numerical (ε = 1e-7)
- **SHAP explanations** — ML models include TreeExplainer for feature importance
- **Offline-capable** — All ML projects use synthetic data generators; real data optional

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
