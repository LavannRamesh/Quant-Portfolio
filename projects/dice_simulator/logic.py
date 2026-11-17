# -*- coding: utf-8 -*-
"""
Dice Game Simulator -- Core Logic
==================================
Simulates a fair 6-sided die with asymmetric payoffs and verifies the
simulated Expected Value against the closed-form analytical solution.

Payoff Rules
------------
  Roll 1 or 2  ->  lose $1
  Roll 3 or 4  ->  push  $0
  Roll 5       ->  win  $1
  Roll 6       ->  win  $2

Analytical E[X]
----------------
  E[X] = (2/6)(-1) + (2/6)(0) + (1/6)(1) + (1/6)(2)
       = -2/6 + 0 + 1/6 + 2/6
       = 1/6
       ~ 0.16667

All simulation is NumPy-vectorised (no Python loops on the hot path).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any

# ── Payoff table (index 0 is unused; indices 1-6 map die faces) ──────
PAYOFF_TABLE = np.array([0, -1, -1, 0, 0, 1, 2], dtype=np.float64)

# ── Face probabilities (uniform, but kept explicit for clarity) ──────
FACE_PROBS = np.array([1 / 6] * 6)

# ── Analytical expected value (exact) ────────────────────────────────
PAYOFFS = np.array([-1, -1, 0, 0, 1, 2], dtype=np.float64)


def manual_expected_value() -> float:
    """
    Compute E[X] analytically.

    E[X] = sum_i P(face_i) * payoff(face_i)
         = (1/6)*(-1) + (1/6)*(-1) + (1/6)*(0)
           + (1/6)*(0) + (1/6)*(1) + (1/6)*(2)
         = 1/6
    """
    return float(np.dot(FACE_PROBS, PAYOFFS))


def simulate_dice_game(
    n_simulations: int = 100_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run *n_simulations* of the dice game using vectorised NumPy.

    Returns
    -------
    dict with keys:
        outcomes        : 1-D array of payoffs per trial
        cumulative_mean : running mean of payoffs (for convergence plot)
        histogram       : dict {payoff_value: count}
        simulated_ev    : float, overall mean payoff
        analytical_ev   : float, exact E[X]
        error_pct       : float, |(sim - exact) / exact| * 100
    """
    rng = np.random.default_rng(seed)

    # Roll N dice at once (values 1-6)
    rolls = rng.integers(1, 7, size=n_simulations)

    # Map each roll to its payoff via the lookup table
    outcomes = PAYOFF_TABLE[rolls]

    # Cumulative mean for convergence visualisation
    cumulative_mean = np.cumsum(outcomes) / np.arange(1, n_simulations + 1)

    # Histogram counts per unique payoff
    unique, counts = np.unique(outcomes, return_counts=True)
    histogram = dict(zip(unique.astype(int), counts.astype(int)))

    simulated_ev = float(outcomes.mean())
    analytical_ev = manual_expected_value()
    error_pct = abs(simulated_ev - analytical_ev) / abs(analytical_ev) * 100

    return {
        "outcomes": outcomes,
        "cumulative_mean": cumulative_mean,
        "histogram": histogram,
        "simulated_ev": simulated_ev,
        "analytical_ev": analytical_ev,
        "error_pct": error_pct,
        "n_simulations": n_simulations,
    }


# ── Quick self-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    ev = manual_expected_value()
    print(f"Analytical E[X] = {ev:.6f}  (should be ~0.16667)")

    result = simulate_dice_game(100_000, seed=42)
    print(f"Simulated  E[X] = {result['simulated_ev']:.6f}")
    print(f"Error           = {result['error_pct']:.4f}%")
    print(f"Histogram       = {result['histogram']}")

    assert result["error_pct"] < 2.0, "Simulated EV diverged too far from analytical!"
    print("All checks passed.")
