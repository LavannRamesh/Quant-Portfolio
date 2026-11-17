# -*- coding: utf-8 -*-
"""
Options Pricer -- Core Logic
=============================
Black-Scholes closed-form pricing validated against Monte Carlo simulation.

Black-Scholes Formulas (European Options)
------------------------------------------
    d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    Call = S * N(d1) - K * exp(-r*T) * N(d2)
    Put  = K * exp(-r*T) * N(-d2) - S * N(-d1)

Greeks
------
    Delta (call) = N(d1)
    Gamma        = phi(d1) / (S * sigma * sqrt(T))
    Theta (call) = -(S * phi(d1) * sigma) / (2*sqrt(T)) - r*K*exp(-rT)*N(d2)
    Vega         = S * phi(d1) * sqrt(T)
    Rho   (call) = K * T * exp(-r*T) * N(d2)

Monte Carlo Pricing
--------------------
    S_T = S * exp((r - sigma^2/2)*T + sigma*sqrt(T)*Z),  Z ~ N(0,1)
    Call_MC = exp(-r*T) * mean(max(S_T - K, 0))

Convergence rate: O(1/sqrt(N)) by CLT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Any


def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> Dict[str, float]:
    """
    Black-Scholes closed-form price and Greeks.

    Parameters
    ----------
    S     : current stock price
    K     : strike price
    T     : time to expiry (years)
    r     : risk-free rate (annualised)
    sigma : volatility (annualised)
    option_type : 'call' or 'put'

    Returns
    -------
    dict with keys: price, delta, gamma, theta, vega, rho
    """
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Standard normal CDF and PDF
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    phi_d1 = norm.pdf(d1)

    discount = np.exp(-r * T)

    if option_type == "call":
        price = S * Nd1 - K * discount * Nd2
        delta = Nd1
        theta = -(S * phi_d1 * sigma) / (2 * sqrt_T) - r * K * discount * Nd2
        rho = K * T * discount * Nd2
    else:  # put
        Nmd1 = norm.cdf(-d1)
        Nmd2 = norm.cdf(-d2)
        price = K * discount * Nmd2 - S * Nmd1
        delta = Nd1 - 1.0
        theta = -(S * phi_d1 * sigma) / (2 * sqrt_T) + r * K * discount * Nmd2
        rho = -K * T * discount * Nmd2

    gamma = phi_d1 / (S * sigma * sqrt_T)
    vega = S * phi_d1 * sqrt_T  # per 1.0 change in sigma

    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta / 365),  # daily theta
        "vega": float(vega / 100),    # per 1% change in vol
        "rho": float(rho / 100),      # per 1% change in rate
    }


def monte_carlo_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 1_000_000,
    option_type: str = "call",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Monte Carlo option pricing with convergence tracking.

    S_T = S * exp((r - sigma^2/2)*T + sigma*sqrt(T)*Z)

    Returns
    -------
    dict with keys: price, std_error, convergence (DataFrame)
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)

    # Simulate terminal stock prices
    S_T = S * np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * Z)

    # Payoffs
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    discount = np.exp(-r * T)
    discounted = discount * payoffs

    # Overall price
    mc_price = float(discounted.mean())
    mc_std_error = float(discounted.std() / np.sqrt(n_paths))

    # Convergence tracking: compute running mean at checkpoints
    checkpoints = np.unique(
        np.concatenate([
            np.arange(100, min(10_000, n_paths), 100),
            np.arange(10_000, min(100_000, n_paths), 1_000),
            np.arange(100_000, n_paths + 1, 10_000),
        ])
    )
    checkpoints = checkpoints[checkpoints <= n_paths]

    cum_sum = np.cumsum(discounted)
    conv_prices = cum_sum[checkpoints - 1] / checkpoints
    conv_stderr = np.array([
        discounted[:cp].std() / np.sqrt(cp) for cp in checkpoints
    ])

    convergence = pd.DataFrame({
        "n_paths": checkpoints,
        "mc_price": conv_prices,
        "std_error": conv_stderr,
    })

    return {
        "price": mc_price,
        "std_error": mc_std_error,
        "convergence": convergence,
    }


def price_sensitivity_surface(
    K: float,
    T: float,
    r: float,
    S_range: tuple = (50, 200),
    sigma_range: tuple = (0.05, 0.80),
    n_points: int = 30,
    option_type: str = "call",
) -> Dict[str, np.ndarray]:
    """
    Compute option price over a grid of (S, sigma) for surface plotting.
    """
    S_vals = np.linspace(S_range[0], S_range[1], n_points)
    sigma_vals = np.linspace(sigma_range[0], sigma_range[1], n_points)
    S_grid, sigma_grid = np.meshgrid(S_vals, sigma_vals)

    prices = np.zeros_like(S_grid)
    for i in range(n_points):
        for j in range(n_points):
            result = black_scholes(S_grid[i, j], K, T, r, sigma_grid[i, j], option_type)
            prices[i, j] = result["price"]

    return {
        "S": S_grid,
        "sigma": sigma_grid,
        "prices": prices,
    }


# ── Self-test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test BS pricing
    bs = black_scholes(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
    print(f"BS Call Price: ${bs['price']:.4f}")
    print(f"  Delta={bs['delta']:.4f}  Gamma={bs['gamma']:.6f}  "
          f"Theta={bs['theta']:.4f}  Vega={bs['vega']:.4f}  Rho={bs['rho']:.4f}")

    # Test MC pricing
    mc = monte_carlo_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2,
                           n_paths=500_000, option_type="call")
    print(f"\nMC Call Price: ${mc['price']:.4f} +/- {mc['std_error']:.4f}")

    error_pct = abs(mc["price"] - bs["price"]) / bs["price"] * 100
    print(f"BS vs MC error: {error_pct:.3f}%")
    assert error_pct < 1.0, f"MC diverged too far from BS: {error_pct:.3f}%"
    print("All checks passed.")
