# Moving Average Crossover Backtest

## Business Question
Can a simple dual-SMA crossover strategy beat buy-and-hold after accounting for transaction costs and look-ahead bias?

## Look-Ahead Bias (The #1 Backtesting Pitfall)
Look-ahead bias occurs when a backtest uses information from the future. The most common mistake:

```python
# WRONG: trades on the same bar the signal was computed
position = (sma_fast > sma_slow).astype(int)

# CORRECT: trades one bar AFTER the signal
signal = (sma_fast > sma_slow).astype(int)
position = signal.shift(1)   # <-- CRITICAL
```

Without `.shift(1)`, the strategy pretends it knew the closing price before the bar closed.

## Methodology
- **Signal**: Long when SMA_fast > SMA_slow, flat otherwise
- **Position entry**: Next bar after signal (via `.shift(1)`)
- **Transaction costs**: Applied each time position changes (default 10 bps)

## Risk Metrics
$$\text{Sharpe} = \frac{\mu_{excess}}{\sigma} \times \sqrt{252}$$
$$\text{MaxDD} = \max_t\left(1 - \frac{\text{Equity}_t}{\max_{s \leq t} \text{Equity}_s}\right)$$
$$\text{CAGR} = \left(\frac{\text{Equity}_T}{\text{Equity}_0}\right)^{252/N} - 1$$

## Stack
- Pandas (signal generation, vectorised backtest)
- NumPy (GBM data generation)
- Plotly (equity curves, drawdown charts)
- Streamlit (interactive UI)
