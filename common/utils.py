"""
common/utils.py
===============
Shared utility functions used across multiple projects.
"""

import numpy as np
import pandas as pd


def format_pct(value: float, decimals: int = 2) -> str:
    """Format a float as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, decimals: int = 2) -> str:
    """Format a float as USD currency."""
    return f"${value:,.{decimals}f}"


def annualize_sharpe(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Compute the annualized Sharpe ratio from daily returns.

    Sharpe = (mean_excess_return / std_return) * sqrt(252)
    """
    excess = daily_returns - risk_free_rate / 252
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(252))


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Compute the maximum drawdown from an equity curve.

    MaxDD = max(1 - equity / running_max)
    """
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return float(drawdown.min())


def rolling_drawdown(equity_curve: pd.Series) -> pd.Series:
    """Compute the rolling drawdown series."""
    running_max = equity_curve.cummax()
    return (equity_curve - running_max) / running_max
