# -*- coding: utf-8 -*-
"""
2D Damped Wave Equation Simulator -- Streamlit UI
==================================================
Pure rendering layer. All computation lives in logic.py.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from common.theme import render_header, PLOTLY_TEMPLATE, COLORS
from projects.wave_equation.logic import (
    solve_damped_wave_2d,
    check_cfl_stability,
)


def render() -> None:
    """Entry point called by the master app.py router."""
    render_header(
        "2D Damped Wave Equation Simulator (Drum Membrane)",
        "Finite difference numerical PDE solution of vibrating membranes, wave diffraction, damping dissipation, and 3D surface animation",
        icon="🌊",
    )

    # ── Sidebar Controls ─────────────────────────────────────────────
    st.sidebar.markdown("### Physical Parameters")
    c_speed = st.sidebar.slider("Wave Speed (c)", 0.5, 2.5, 1.0, 0.1)
    gamma_damp = st.sidebar.slider("Damping Coefficient (gamma)", 0.0, 1.0, 0.25, 0.05)
    t_max = st.sidebar.slider("Simulation Time (sec)", 0.5, 3.0, 1.5, 0.25)

    st.sidebar.markdown("### Numerical Grid Settings")
    grid_size = st.sidebar.select_slider("Grid Resolution (N x N)", options=[35, 50, 65, 80], value=50)
    ic_type = st.sidebar.selectbox(
        "Initial Membrane Disturbance",
        options=[
            ("gaussian_pluck", "Central Gaussian Pluck (Drum Stick)"),
            ("double_drop", "Dual Droplet Interference"),
            ("circular_drum_mode", "Modal Eigenmode (m=2, n=2)"),
            ("corner_tsunami", "Corner Energy Pulse"),
        ],
        format_func=lambda x: x[1],
        index=0,
    )[0]

    # ── Mathematical Formulation Expander ────────────────────────────
    with st.expander("📐 Mathematical PDE Derivation & CFL Stability Condition", expanded=False):
        st.markdown(
            r"""
            The continuous 2D damped wave equation on a fixed square membrane $\Omega = [0, 1]^2$:
            """
        )
        st.latex(
            r"\frac{\partial^2 u}{\partial t^2} + \gamma \frac{\partial u}{\partial t} = c^2 \nabla^2 u = c^2 \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)"
        )
        st.markdown(
            r"""
            **Explicit Central Finite Difference Discretization ($O(\Delta t^2, \Delta x^2)$):**
            """
        )
        st.latex(
            r"u_{i,j}^{n+1} = \frac{2u_{i,j}^n - \left(1 - \frac{\gamma \Delta t}{2}\right)u_{i,j}^{n-1} + r^2 \left(u_{i+1,j}^n + u_{i-1,j}^n + u_{i,j+1}^n + u_{i,j-1}^n - 4u_{i,j}^n\right)}{1 + \frac{\gamma \Delta t}{2}}"
        )
        st.markdown(
            """
            **Von Neumann CFL Stability Criterion:**
            """
        )
        st.latex(
            r"r = \frac{c \Delta t}{\Delta x} \le \frac{1}{\sqrt{2}} \approx 0.7071"
        )

    # ── Simulation Computation (Cached) ──────────────────────────────
    @st.cache_data(show_spinner=False)
    def _run_pde(n, c, g, t, ic):
        return solve_damped_wave_2d(Nx=n, Ny=n, c=c, gamma=g, T_max=t, n_save_frames=45, ic_type=ic)

    with st.spinner("Solving hyperbolic PDE via finite differences..."):
        sim_res = _run_pde(grid_size, c_speed, gamma_damp, t_max, ic_type)

    frames = sim_res["frames"]
    times = sim_res["times"]
    cfl = sim_res["cfl_number"]
    dt = sim_res["dt"]
    dx = sim_res["dx"]

    # ── Status Banner ─────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grid Mesh", f"{grid_size} x {grid_size}")
    c2.metric("CFL Number", f"{cfl:.3f}", delta="<= 0.707 (Stable)")
    c3.metric("Time Step (dt)", f"{dt:.5f} s")
    c4.metric("Saved Frames", f"{len(frames)}")

    # ── Interactive Frame Playback ────────────────────────────────────
    st.markdown("### Interactive 3D Wavefront Playback")
    frame_idx = st.slider(
        "Simulation Time Slider",
        min_value=0,
        max_value=len(frames) - 1,
        value=min(8, len(frames) - 1),
        format_func=lambda idx: f"t = {times[idx]:.3f}s (Frame {idx + 1}/{len(frames)})",
    )

    current_u = frames[frame_idx]

    # Chart 1: 3D Surface
    x_coords = np.linspace(0, 1, grid_size)
    y_coords = np.linspace(0, 1, grid_size)

    # Fixed z limits for consistent camera visualization
    max_amp = max(float(np.max(np.abs(frames[0]))), 0.1)

    fig_3d = go.Figure(data=[
        go.Surface(
            z=current_u,
            x=x_coords,
            y=y_coords,
            colorscale="Plasma",
            cmin=-max_amp * 0.7,
            cmax=max_amp * 0.7,
            colorbar=dict(title="Amplitude u"),
        )
    ])
    fig_3d.update_layout(
        title=f"Membrane Displacement u(x, y, t) at t = {times[frame_idx]:.3f} s",
        template=PLOTLY_TEMPLATE,
        scene=dict(
            xaxis=dict(title="X (m)", range=[0, 1]),
            yaxis=dict(title="Y (m)", range=[0, 1]),
            zaxis=dict(title="Displacement u", range=[-max_amp * 1.1, max_amp * 1.1]),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
        height=550,
        margin=dict(t=40, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # ── Lower Analysis Tabs ──────────────────────────────────────────
    tab_2d, tab_energy = st.tabs([
        "🗺️ 2D Contour & Wave Diffraction Heatmap",
        "⚡ Kinetic/Potential Energy Dissipation Curve",
    ])

    with tab_2d:
        fig_2d = px.imshow(
            current_u,
            color_continuous_scale="Viridis",
            zmin=-max_amp * 0.7,
            zmax=max_amp * 0.7,
            title=f"Planar Wavefield Projection (t = {times[frame_idx]:.3f} s)",
            template=PLOTLY_TEMPLATE,
        )
        fig_2d.update_layout(height=420)
        st.plotly_chart(fig_2d, use_container_width=True)

    with tab_energy:
        fig_energy = go.Figure()
        fig_energy.add_trace(go.Scatter(
            x=times,
            y=sim_res["energies"],
            mode="lines+markers",
            line=dict(color=COLORS["primary"], width=2.5),
            name="Membrane Energy",
        ))
        fig_energy.update_layout(
            title="Total Mechanical Energy Decay under Viscous Damping",
            template=PLOTLY_TEMPLATE,
            xaxis_title="Time t (seconds)",
            yaxis_title="Total Energy (arbitrary units)",
            height=380,
            margin=dict(t=30),
        )
        st.plotly_chart(fig_energy, use_container_width=True)
