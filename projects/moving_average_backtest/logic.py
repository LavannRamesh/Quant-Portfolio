# -*- coding: utf-8 -*-
"""
Moving Average Crossover Backtest -- Core Logic
================================================
Implements a dual-SMA crossover strategy with EXPLICIT look-ahead bias
prevention, realistic transaction costs, and standard risk metrics.

LOOK-AHEAD BIAS PREVENTION (CRITICAL)
--------------------------------------
The signal is computed from SMA values at time t, but positions are only
entered at time t+1.  This is implemented via:

    signal = (sma_fast > sma_slow).astype(int)
    position = signal.shift(1)   # <-- THE CRITICAL LINE

Without the shift, the backtest would use information from the CURRENT
bar's close to compute the signal and pretend it traded at the SAME
bar's close -- which is impossible in reality because you only know the
close after the bar has closed.

Transaction Cost Model
-----------------------
Each time the position changes (0->1 or 1->0), a one-way cost is applied:

    trades = |position - position_prev|
    cost   = trades * cost_per_trade        (default 10 bps = 0.001)

Risk Metrics
------------
    Sharpe = (mean_excess / std) * sqrt(252)
    MaxDD  = max(1 - equity / running_max)
    CAGR   = (final / initial) ^ (252 / n_days) - 1
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any


def generate_sample_data(
    ticker: str = "SPY",
    n_days: int = 2520,
    seed: int = 42,
    mu: float = 0.08,
    sigma: float = 0.18,
    S0: float = 300.0,
) -> pd.DataFrame:
    """
    Generate synthetic stock data via geometric Brownian motion.

    dS/S = mu*dt + sigma*dW

    Returns DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    n = n_days

    # Daily log returns
    log_returns = (mu - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    prices = S0 * np.exp(np.cumsum(log_returns))
    prices = np.insert(prices, 0, S0)[:n]  # n data points

    dates = pd.bdate_range(start="2015-01-02", periods=n, freq="B")

    # Synthetic OHLCV
    df = pd.DataFrame({"Date": dates, "Close": prices})
    df["Open"] = df["Close"].shift(1).fillna(S0)
    noise = rng.uniform(0.002, 0.015, n)
    df["High"] = df[["Open", "Close"]].max(axis=1) * (1 + noise)
    df["Low"] = df[["Open", "Close"]].min(axis=1) * (1 - noise)
    df["Volume"] = rng.integers(5_000_000, 50_000_000, n)
    df["Ticker"] = ticker

    return df.reset_index(drop=True)


def run_backtest(
    prices: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 50,
    cost_bps: float = 10.0,
    risk_free_rate: float = 0.02,
) -> Dict[str, Any]:
    """
    Run a dual-SMA crossover backtest with look-ahead bias prevention.

    Parameters
    ----------
    prices      : DataFrame with at least a 'Close' column
    fast_window : short SMA window
    slow_window : long SMA window
    cost_bps    : round-trip transaction cost in basis points
    risk_free_rate : annual risk-free rate for Sharpe calculation

    Returns
    -------
    dict with keys:
        data      : DataFrame with signals, positions, equity curves
        metrics   : dict of performance metrics
    """
    df = prices.copy()
    close = df["Close"]

    # ── Compute SMAs ─────────────────────────────────────────────────
    df["SMA_Fast"] = close.rolling(window=fast_window, min_periods=fast_window).mean()
    df["SMA_Slow"] = close.rolling(window=slow_window, min_periods=slow_window).mean()

    # ── Generate signal ──────────────────────────────────────────────
    # Signal = 1 when fast SMA > slow SMA (bullish), else 0
    df["Signal"] = (df["SMA_Fast"] > df["SMA_Slow"]).astype(int)

    # ══════════════════════════════════════════════════════════════════
    # LOOK-AHEAD BIAS PREVENTION:
    # The signal computed from bar t's close is acted on at bar t+1.
    # This .shift(1) ensures we never trade on information we couldn't
    # have known in real-time.
    # ══════════════════════════════════════════════════════════════════
    df["Position"] = df["Signal"].shift(1)  # <-- CRITICAL: prevents look-ahead bias

    # Fill NaN positions (first slow_window bars have no signal)
    df["Position"] = df["Position"].fillna(0)

    # ── Daily returns ────────────────────────────────────────────────
    df["Daily_Return"] = close.pct_change().fillna(0)

    # ── Transaction costs ────────────────────────────────────────────
    cost_per_trade = cost_bps / 10_000
    df["Trades"] = df["Position"].diff().abs().fillna(0)
    df["Transaction_Cost"] = df["Trades"] * cost_per_trade

    # ── Strategy returns ─────────────────────────────────────────────
    df["Strategy_Return"] = (
        df["Position"] * df["Daily_Return"] - df["Transaction_Cost"]
    )

    # ── Equity curves (starting at $1) ───────────────────────────────
    df["Strategy_Equity"] = (1 + df["Strategy_Return"]).cumprod()
    df["Benchmark_Equity"] = (1 + df["Daily_Return"]).cumprod()

    # ── Drawdowns ────────────────────────────────────────────────────
    strat_max = df["Strategy_Equity"].cummax()
    df["Strategy_Drawdown"] = (df["Strategy_Equity"] - strat_max) / strat_max

    bench_max = df["Benchmark_Equity"].cummax()
    df["Benchmark_Drawdown"] = (df["Benchmark_Equity"] - bench_max) / bench_max

    # ── Compute metrics ──────────────────────────────────────────────
    # Drop NaN rows for clean metric calculation
    valid = df.dropna(subset=["SMA_Slow"])
    strat_ret = valid["Strategy_Return"]
    bench_ret = valid["Daily_Return"]
    n_trading_days = len(valid)

    metrics = compute_metrics(strat_ret, bench_ret, n_trading_days, risk_free_rate)
    metrics["total_trades"] = int(df["Trades"].sum())
    metrics["cost_bps"] = cost_bps

    return {"data": df, "metrics": metrics}


def compute_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    n_trading_days: int,
    risk_free_rate: float = 0.02,
) -> Dict[str, float]:
    """Compute standard performance metrics."""
    rf_daily = risk_free_rate / 252

    # Sharpe ratios
    strat_excess = strategy_returns - rf_daily
    bench_excess = benchmark_returns - rf_daily

    strat_sharpe = (
        float(strat_excess.mean() / strat_excess.std() * np.sqrt(252))
        if strat_excess.std() > 0
        else 0.0
    )
    bench_sharpe = (
        float(bench_excess.mean() / bench_excess.std() * np.sqrt(252))
        if bench_excess.std() > 0
        else 0.0
    )

    # Equity curves
    strat_equity = (1 + strategy_returns).cumprod()
    bench_equity = (1 + benchmark_returns).cumprod()

    # CAGR
    years = n_trading_days / 252
    strat_cagr = float(strat_equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0
    bench_cagr = float(bench_equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0

    # Max Drawdown
    strat_max_dd = float(
        ((strat_equity - strat_equity.cummax()) / strat_equity.cummax()).min()
    )
    bench_max_dd = float(
        ((bench_equity - bench_equity.cummax()) / bench_equity.cummax()).min()
    )

    # Total return
    strat_total = float(strat_equity.iloc[-1] - 1)
    bench_total = float(bench_equity.iloc[-1] - 1)

    # Win rate
    win_rate = float((strategy_returns > 0).sum() / max(len(strategy_returns), 1))

    return {
        "strategy_sharpe": strat_sharpe,
        "benchmark_sharpe": bench_sharpe,
        "strategy_cagr": strat_cagr,
        "benchmark_cagr": bench_cagr,
        "strategy_max_dd": strat_max_dd,
        "benchmark_max_dd": bench_max_dd,
        "strategy_total_return": strat_total,
        "benchmark_total_return": bench_total,
        "win_rate": win_rate,
    }


# ── Self-test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_sample_data()
    print(f"Generated {len(df)} rows of synthetic data")

    result = run_backtest(df, fast_window=20, slow_window=50, cost_bps=10)
    m = result["metrics"]
    print(f"\nStrategy Sharpe:  {m['strategy_sharpe']:.3f}")
    print(f"Benchmark Sharpe: {m['benchmark_sharpe']:.3f}")
    print(f"Strategy MaxDD:   {m['strategy_max_dd']:.2%}")
    print(f"Strategy CAGR:    {m['strategy_cagr']:.2%}")
    print(f"Total trades:     {m['total_trades']}")
    print(f"Win rate:         {m['win_rate']:.2%}")

    # Verify look-ahead bias prevention: Position at t should equal Signal at t-1
    data = result["data"].dropna(subset=["SMA_Slow"]).iloc[1:]
    assert (data["Position"] == result["data"]["Signal"].shift(1).loc[data.index]).all(), \
        "LOOK-AHEAD BIAS DETECTED: Position != Signal.shift(1)"
    print("\nLook-ahead bias check PASSED.")
    print("All checks passed.")
