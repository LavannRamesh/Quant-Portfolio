# -*- coding: utf-8 -*-
"""
Kelly Criterion Bet Sizing Simulator -- Core Logic
=====================================================
Demonstrates optimal bet sizing via the Kelly Criterion and shows
how over-betting leads to ruin even with a positive-edge game.

Kelly Formula
-------------
    f* = (b * p - q) / b

where
    b = net decimal odds (payout per $1 wagered, not counting stake)
    p = probability of winning
    q = 1 - p = probability of losing
    f* = optimal fraction of bankroll to wager

All simulation is NumPy-vectorised across paths AND bets simultaneously.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Any


def kelly_fraction(win_prob: float, odds: float) -> float:
    """
    Compute the optimal Kelly fraction.

    Parameters
    ----------
    win_prob : float  -- probability of winning (0 < p < 1)
    odds     : float  -- net payout per $1 wagered (b)

    Returns
    -------
    f* : float -- optimal fraction of bankroll to bet
    """
    q = 1.0 - win_prob
    f_star = (odds * win_prob - q) / odds
    return max(f_star, 0.0)  # never negative (don't bet if no edge)


def simulate_kelly_paths(
    win_prob: float = 0.6,
    odds: float = 1.0,
    fractions: List[float] | None = None,
    n_bets: int = 500,
    n_paths: int = 1_000,
    initial_bankroll: float = 100.0,
    ruin_threshold: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Simulate wealth trajectories for several bet-sizing fractions.

    Parameters
    ----------
    win_prob         : win probability per bet
    odds             : net payout per $1 wagered
    fractions        : list of fractions to compare (default: 5%, 20%, f*, 60%, 100%)
    n_bets           : number of sequential bets per path
    n_paths          : number of independent wealth paths per fraction
    initial_bankroll : starting wealth
    ruin_threshold   : wealth below which a path is considered "ruined"
    seed             : random seed

    Returns
    -------
    dict with keys per fraction:
        wealth_paths  : (n_paths, n_bets+1) array of wealth over time
        median_path   : (n_bets+1,) median wealth at each step
        p5_path       : 5th percentile path
        p95_path      : 95th percentile path
        terminal_wealth : (n_paths,) final wealth values
        median_terminal : float
        ruin_prob     : float in [0, 1]
        growth_rate   : float, median log-growth per bet
    Also:
        kelly_f       : the optimal Kelly fraction
        fractions     : list of fractions tested
    """
    rng = np.random.default_rng(seed)
    f_star = kelly_fraction(win_prob, odds)

    if fractions is None:
        fractions = sorted(set([0.05, 0.20, round(f_star, 4), 0.60, 1.00]))

    # Pre-generate all outcomes: shape (n_paths, n_bets), True = win
    outcomes = rng.random((n_paths, n_bets)) < win_prob

    results: Dict[str, Any] = {
        "kelly_f": f_star,
        "fractions": fractions,
        "fraction_results": {},
    }

    for frac in fractions:
        # Multiplicative returns per bet:
        #   win  -> bankroll *= (1 + frac * odds)
        #   loss -> bankroll *= (1 - frac)
        win_mult = 1.0 + frac * odds
        loss_mult = 1.0 - frac

        # Build multiplier matrix: shape (n_paths, n_bets)
        multipliers = np.where(outcomes, win_mult, loss_mult)

        # Wealth evolution via cumulative product
        # Prepend a column of 1s for initial bankroll
        cum_prod = np.cumprod(multipliers, axis=1)
        wealth = np.empty((n_paths, n_bets + 1))
        wealth[:, 0] = initial_bankroll
        wealth[:, 1:] = initial_bankroll * cum_prod

        # Clamp near-zero values to avoid log issues
        wealth = np.maximum(wealth, 1e-10)

        terminal = wealth[:, -1]
        ruin_count = np.sum(np.any(wealth <= ruin_threshold, axis=1))

        # Log growth rate: median of mean(log(multiplier)) per path
        log_mults = np.log(multipliers)
        mean_log_growth = np.mean(log_mults, axis=1)
        median_growth = float(np.median(mean_log_growth))

        results["fraction_results"][frac] = {
            "wealth_paths": wealth,
            "median_path": np.median(wealth, axis=0),
            "p5_path": np.percentile(wealth, 5, axis=0),
            "p95_path": np.percentile(wealth, 95, axis=0),
            "terminal_wealth": terminal,
            "median_terminal": float(np.median(terminal)),
            "ruin_prob": float(ruin_count / n_paths),
            "growth_rate": median_growth,
        }

    return results


# ── Self-test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    f = kelly_fraction(0.6, 1.0)
    print(f"Kelly f* = {f:.4f}  (should be 0.2000)")
    assert abs(f - 0.2) < 1e-10

    res = simulate_kelly_paths(win_prob=0.6, odds=1.0, n_bets=300, n_paths=500)
    print(f"Fractions tested: {res['fractions']}")
    for frac, data in res["fraction_results"].items():
        print(
            f"  f={frac:.2f}  median_terminal=${data['median_terminal']:.2f}  "
            f"ruin={data['ruin_prob']:.2%}  growth={data['growth_rate']:.5f}"
        )
    print("All checks passed.")
