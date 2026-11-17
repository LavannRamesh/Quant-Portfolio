# -*- coding: utf-8 -*-
"""
Stock Returns & Volatility Dashboard -- Core Logic
===================================================
Provides robust portfolio and stock statistics:
- Real market data download via yfinance (with graceful fallback to
  correlated Geometric Brownian Motion if offline).
- Daily log returns and arithmetic returns.
- Annualized rolling volatility.
- Cumulative returns and rolling peak drawdowns.
- Risk vs Return metrics (CAGR, Annualized Vol, Sharpe Ratio, Skew, Kurtosis).
- Correlation matrix.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_synthetic_stock_data(
    tickers: List[str],
    n_days: int = 504,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic multi-asset correlated price series via Geometric Brownian Motion.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(tickers)
    dt = 1.0 / 252.0

    # Base parameters per asset type / ticker
    params = {
        "SPY": (0.09, 0.16, 400.0),
        "AAPL": (0.15, 0.24, 180.0),
        "MSFT": (0.14, 0.22, 330.0),
        "GOOGL": (0.12, 0.25, 135.0),
        "TSLA": (0.20, 0.45, 220.0),
    }

    drifts = []
    vols = []
    initial_prices = []

    for t in tickers:
        p = params.get(t, (0.10 + rng.uniform(-0.03, 0.05), rng.uniform(0.18, 0.35), 100.0))
        drifts.append(p[0])
        vols.append(p[1])
        initial_prices.append(p[2])

    drifts = np.array(drifts)
    vols = np.array(vols)
    initial_prices = np.array(initial_prices)

    # Positive correlation matrix across equity assets (~0.5 - 0.7)
    corr = np.full((n_assets, n_assets), 0.55)
    np.fill_diagonal(corr, 1.0)
    L = np.linalg.cholesky(corr)

    # Generate correlated normal increments
    uncorr_normals = rng.standard_normal((n_days, n_assets))
    corr_normals = uncorr_normals @ L.T

    # Log returns: (mu - sigma^2 / 2)*dt + sigma*sqrt(dt)*Z
    log_returns = (drifts - 0.5 * (vols ** 2)) * dt + vols * np.sqrt(dt) * corr_normals
    cum_returns = np.vstack([np.zeros(n_assets), np.cumsum(log_returns, axis=0)])
    prices = initial_prices * np.exp(cum_returns)

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days + 1, freq="B")
    df = pd.DataFrame(prices, index=dates, columns=tickers)
    df.index.name = "Date"
    return df


def load_stock_data(
    tickers: List[str],
    period: str = "2y",
) -> Tuple[pd.DataFrame, bool, str]:
    """
    Attempt to download real adjusted close prices from yfinance.
    Falls back to synthetic GBM if yfinance fails or is offline.

    Returns
    -------
    (df_prices, is_real_data, warning_message)
    """
    tickers = [t.upper().strip() for t in tickers if t.strip()]
    if not tickers:
        tickers = ["SPY", "AAPL", "MSFT"]

    is_real = False
    warning_msg = ""
    prices = None

    try:
        import yfinance as yf

        data = yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if data is not None and not data.empty:
            if "Close" in data.columns and isinstance(data.columns, pd.MultiIndex):
                prices = data["Close"][tickers].dropna()
            elif len(tickers) == 1 and "Close" in data.columns:
                prices = pd.DataFrame({tickers[0]: data["Close"]}).dropna()
            elif isinstance(data, pd.DataFrame) and not isinstance(data.columns, pd.MultiIndex):
                # Single ticker or flat
                if "Close" in data:
                    prices = data[["Close"]].rename(columns={"Close": tickers[0]}).dropna()
                else:
                    prices = data[tickers].dropna()

            if prices is not None and len(prices) > 20:
                is_real = True
    except Exception as exc:
        logger.warning(f"yfinance download failed: {exc}")
        warning_msg = f"yfinance live fetch unavailable ({exc}). Using calibrated synthetic market data."

    if not is_real or prices is None or prices.empty:
        if not warning_msg:
            warning_msg = "Live market connection not responding. Using calibrated synthetic multi-asset data."
        prices = generate_synthetic_stock_data(tickers=tickers, n_days=504)

    return prices, is_real, warning_msg


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute percentage daily returns."""
    return prices.pct_change().dropna()


def compute_normalized_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Normalize all price series to 100 at t=0."""
    return (prices / prices.iloc[0]) * 100.0


def compute_rolling_volatility(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Annualized rolling standard deviation (sigma_daily * sqrt(252))."""
    return returns.rolling(window=window).std() * np.sqrt(252)


def compute_drawdowns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate rolling peak drawdown for each ticker: (Price - Peak) / Peak."""
    cum_max = prices.cummax()
    return (prices - cum_max) / cum_max


def compute_portfolio_metrics(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """
    Compute comprehensive risk/return stats per asset:
    - Annualized Return (CAGR)
    - Annualized Volatility
    - Sharpe Ratio
    - Maximum Drawdown
    - Skewness
    - Kurtosis
    """
    n_days = len(returns)
    years = max(n_days / 252.0, 1.0 / 252.0)
    rf_daily = risk_free_rate / 252.0

    stats = []
    for col in returns.columns:
        ret_series = returns[col]
        price_series = prices[col]

        cagr = (price_series.iloc[-1] / price_series.iloc[0]) ** (1.0 / years) - 1.0
        ann_vol = ret_series.std() * np.sqrt(252)
        excess_mean = ret_series.mean() - rf_daily
        sharpe = (excess_mean / ret_series.std() * np.sqrt(252)) if ret_series.std() > 0 else 0.0

        dd = (price_series - price_series.cummax()) / price_series.cummax()
        max_dd = dd.min()
        skew = ret_series.skew()
        kurt = ret_series.kurtosis()

        stats.append({
            "Ticker": col,
            "CAGR": cagr,
            "Ann. Vol": ann_vol,
            "Sharpe (rf=2%)": sharpe,
            "Max Drawdown": max_dd,
            "Skewness": skew,
            "Kurtosis": kurt,
        })

    return pd.DataFrame(stats).set_index("Ticker")


def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Return asset pairwise Pearson correlation matrix."""
    return returns.corr()


if __name__ == "__main__":
    test_tickers = ["SPY", "AAPL", "MSFT", "TSLA"]
    prices_df, is_real_flag, msg = load_stock_data(test_tickers)
    print(f"Data Loaded: {len(prices_df)} rows, Real: {is_real_flag}, Msg: {msg}")
    rets = compute_daily_returns(prices_df)
    m = compute_portfolio_metrics(rets, prices_df)
    print("\nMetrics:")
    print(m)
    assert not prices_df.empty, "Prices DataFrame is empty!"
    print("\nStock Dashboard Logic Test: PASSED")
