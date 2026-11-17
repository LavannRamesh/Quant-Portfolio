# -*- coding: utf-8 -*-
"""
2D Damped Wave Equation Simulator -- Core Logic
================================================
Solves the damped 2D wave equation on a square membrane:
    d^2 u / dt^2 + gamma * du / dt = c^2 * (d^2 u / dx^2 + d^2 u / dy^2)

Using explicit 2nd-order Central Finite Differences in time and space with
Dirichlet fixed boundary conditions (u = 0 on boundary).

Numerical Stability:
    Courant-Friedrichs-Lewy (CFL) condition:
        r = c * dt / dx <= 1 / sqrt(2)  (~ 0.7071)
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any
import numpy as np


def check_cfl_stability(c: float, dt: float, dx: float) -> Tuple[bool, float, float]:
    """
    Check Courant-Friedrichs-Lewy stability condition for 2D wave equation.

    Returns (is_stable, courant_number, max_stable_dt).
    """
    courant = (c * dt) / dx
    max_courant = 1.0 / np.sqrt(2.0)
    max_stable_dt = (max_courant * dx) / c
    is_stable = courant <= (max_courant + 1e-9)
    return is_stable, courant, max_stable_dt


def generate_initial_condition(
    Nx: int,
    Ny: int,
    ic_type: str = "gaussian_pluck",
) -> np.ndarray:
    """
    Generate initial displacement field u(x, y, 0).
    Grid is normalized to [0, 1] x [0, 1].
    """
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y)

    if ic_type == "gaussian_pluck":
        # Smooth Gaussian strike near center (0.45, 0.55)
        u0 = np.exp(-((X - 0.45) ** 2 + (Y - 0.55) ** 2) / (2 * 0.04 ** 2))
    elif ic_type == "double_drop":
        # Two droplets creating constructive/destructive interference
        u1 = np.exp(-((X - 0.35) ** 2 + (Y - 0.40) ** 2) / (2 * 0.035 ** 2))
        u2 = 0.8 * np.exp(-((X - 0.65) ** 2 + (Y - 0.60) ** 2) / (2 * 0.035 ** 2))
        u0 = u1 + u2
    elif ic_type == "circular_drum_mode":
        # Eigenmode (m=2, n=2) pattern
        u0 = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    elif ic_type == "corner_tsunami":
        # Corner energy impulse
        u0 = np.exp(-((X - 0.15) ** 2 + (Y - 0.15) ** 2) / (2 * 0.05 ** 2))
    else:
        u0 = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * 0.04 ** 2))

    # Enforce zero boundary conditions
    u0[0, :] = 0.0
    u0[-1, :] = 0.0
    u0[:, 0] = 0.0
    u0[:, -1] = 0.0

    return u0


def solve_damped_wave_2d(
    Nx: int = 60,
    Ny: int = 60,
    c: float = 1.0,
    gamma: float = 0.25,
    T_max: float = 2.0,
    n_save_frames: int = 40,
    ic_type: str = "gaussian_pluck",
) -> Dict[str, Any]:
    """
    Simulate the damped 2D wave equation using finite differences.

    Parameters
    ----------
    Nx, Ny        : grid spatial resolution
    c             : wave speed (m/s)
    gamma         : damping coefficient
    T_max         : total simulated physical time
    n_save_frames : number of snapshots to save for interactive visualization
    ic_type       : initial displacement shape

    Returns
    -------
    dict containing:
        frames       : list of 2D numpy arrays (u at each saved time step)
        times        : list of timestamps
        cfl_number   : Courant number
        dt           : numerical time step
        dx           : spatial step
        energy_decay : list of total mechanical energy over time
    """
    dx = 1.0 / (Nx - 1)
    dy = 1.0 / (Ny - 1)
    assert abs(dx - dy) < 1e-9, "Grid must be square for symmetric Laplacian"

    # Auto-calculate stable dt via CFL
    cfl_target = 0.65  # safely below 1/sqrt(2) ~ 0.7071
    dt = cfl_target * dx / (c * np.sqrt(2.0))
    is_stable, courant, _ = check_cfl_stability(c, dt, dx)
    if not is_stable:
        raise ValueError(f"CFL condition violated: courant = {courant:.4f} > 0.7071")

    n_steps = int(np.ceil(T_max / dt))
    r2 = (c * dt / dx) ** 2

    # Initial displacement
    u0 = generate_initial_condition(Nx, Ny, ic_type)

    # First time step (from initial velocity du/dt = 0)
    # u^1 = u^0 + 0.5 * r^2 * Laplacian(u^0)
    laplacian_0 = np.zeros_like(u0)
    laplacian_0[1:-1, 1:-1] = (
        u0[2:, 1:-1] + u0[:-2, 1:-1] + u0[1:-1, 2:] + u0[1:-1, :-2] - 4.0 * u0[1:-1, 1:-1]
    )
    u1 = u0 + 0.5 * r2 * laplacian_0
    # Dirichlet boundaries
    u1[0, :] = u1[-1, :] = u1[:, 0] = u1[:, -1] = 0.0

    # Pre-allocate frames
    save_interval = max(1, n_steps // n_save_frames)
    frames: List[np.ndarray] = [u0.copy()]
    times: List[float] = [0.0]
    energies: List[float] = [float(np.sum(u0**2))]

    u_prev = u0.copy()
    u_curr = u1.copy()
    u_next = np.zeros_like(u0)

    # Precompute damping weights
    damp_factor = gamma * dt / 2.0
    denom = 1.0 + damp_factor
    w_curr = 2.0 / denom
    w_prev = (1.0 - damp_factor) / denom
    w_lap = r2 / denom

    for step in range(1, n_steps + 1):
        # 5-point discrete Laplacian on interior points
        lap = (
            u_curr[2:, 1:-1]
            + u_curr[:-2, 1:-1]
            + u_curr[1:-1, 2:]
            + u_curr[1:-1, :-2]
            - 4.0 * u_curr[1:-1, 1:-1]
        )

        # Update interior
        u_next[1:-1, 1:-1] = (
            w_curr * u_curr[1:-1, 1:-1]
            - w_prev * u_prev[1:-1, 1:-1]
            + w_lap * lap
        )

        # Dirichlet zero boundary conditions
        u_next[0, :] = 0.0
        u_next[-1, :] = 0.0
        u_next[:, 0] = 0.0
        u_next[:, -1] = 0.0

        if step % save_interval == 0 or step == n_steps:
            frames.append(u_next.copy())
            times.append(step * dt)
            # Energy proxy (potential + kinetic displacement magnitude)
            energies.append(float(np.sum(u_next**2)))

        # Cycle pointers
        u_prev[:, :] = u_curr
        u_curr[:, :] = u_next

    return {
        "frames": frames,
        "times": times,
        "cfl_number": courant,
        "dt": dt,
        "dx": dx,
        "n_steps": n_steps,
        "energies": energies,
        "Nx": Nx,
        "Ny": Ny,
    }


if __name__ == "__main__":
    res = solve_damped_wave_2d(Nx=40, Ny=40, c=1.0, gamma=0.2, T_max=1.0, n_save_frames=20)
    print(f"Wave Simulator: {len(res['frames'])} frames generated. CFL={res['cfl_number']:.4f}")
    assert len(res["frames"]) > 10, "Failed to generate simulation frames!"
    assert res["cfl_number"] <= 1.0 / np.sqrt(2.0), "CFL condition failed!"
    # Ensure damping dissipates energy
    assert res["energies"][-1] < res["energies"][0], "Energy did not dissipate under damping!"
    print("Wave Equation Logic Test: PASSED")
