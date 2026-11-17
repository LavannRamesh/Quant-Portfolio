# Stock Returns & Volatility Dashboard

## Business Question
How do various equities and benchmark indices compare in risk-adjusted performance, drawdown behavior, tail risk, and cross-asset correlations over market cycles?

## Quant Methodology

### Return Calculations
Daily simple returns:
$$R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

### Rolling Volatility
Annualized sample standard deviation of daily returns over rolling window $W$:
$$\sigma_{\text{ann}} = \sqrt{252} \times \sqrt{\frac{1}{W-1}\sum_{k=0}^{W-1} (R_{t-k} - \bar{R})^2}$$

### Maximum Drawdown
Drawdown measures the percentage loss from the running historical peak:
$$\text{DD}_t = \frac{P_t - \max_{s \le t} P_s}{\max_{s \le t} P_s}$$
$$\text{MaxDD} = \min_{t} \text{DD}_t$$

### Sharpe Ratio (Annualized)
$$\text{Sharpe} = \frac{\bar{R}_{\text{ann}} - R_f}{\sigma_{\text{ann}}}$$

### Data Pipeline & Offline Resiliency
The module attempts to stream live daily adjusted prices directly from Yahoo Finance (`yfinance`). If rate limits or network blocks occur, it automatically falls back to an exact, calibrated Geometric Brownian Motion simulator with Cholesky-correlated shocks, notifying the user transparently.

## Stack
- `yfinance` & `pandas` (time-series acquisition and transformations)
- `numpy` (Cholesky correlation factorizations for fallback simulation)
- `plotly` (interactive multi-chart risk dashboard)
- `streamlit` (reactive controls and metrics)
