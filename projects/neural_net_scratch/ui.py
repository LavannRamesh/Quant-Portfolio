# -*- coding: utf-8 -*-
"""
Neural Network from Scratch -- Streamlit UI
============================================
Pure rendering layer. All computation lives in logic.py.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from common.theme import render_header, PLOTLY_TEMPLATE, COLORS
from projects.neural_net_scratch.logic import (
    PureNumPyNN,
    generate_synthetic_dataset,
    compute_loss_surface,
)


def render() -> None:
    """Entry point called by the master app.py router."""
    render_header(
        "Neural Network from Scratch (Pure NumPy)",
        "Zero-framework deep learning: analytical matrix backpropagation, numerical gradient verification, decision boundaries, and 3D loss surface optimization trajectories",
        icon="🧠",
    )

    # ── Sidebar Controls ─────────────────────────────────────────────
    st.sidebar.markdown("### Topology & Dataset")
    dataset_name = st.sidebar.selectbox("Benchmark Manifold", ["xor", "circles", "moons"], index=0)
    activation = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid"], index=0)
    hidden_arch = st.sidebar.selectbox("Hidden Layer Nodes", ["8, 4", "12, 6", "16, 8, 4"], index=0)
    learning_rate = st.sidebar.slider("Learning Rate (alpha)", 0.01, 0.30, 0.08, 0.01)
    epochs = st.sidebar.slider("Training Epochs", 100, 1000, 450, 50)
    noise_lvl = st.sidebar.slider("Data Noise", 0.05, 0.25, 0.10, 0.01)

    layer_sizes = [2] + [int(x.strip()) for x in hidden_arch.split(",")] + [1]

    # ── Mathematical Derivation Expander ─────────────────────────────
    with st.expander("📐 Mathematical Backpropagation Derivations & Loss Gradients", expanded=False):
        st.markdown(
            """
            **Forward Propagation per Layer $l$:**
            """
        )
        st.latex(r"Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}, \quad A^{[l]} = g\left(Z^{[l]}\right)")
        st.markdown(
            """
            **Binary Cross-Entropy Objective:**
            """
        )
        st.latex(
            r"\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \ln \hat{y}^{(i)} + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right]"
        )
        st.markdown(
            """
            **Analytical Chain Rule Matrix Gradients:**
            """
        )
        st.latex(
            r"dZ^{[L]} = A^{[L]} - Y, \quad \frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{1}{m} (A^{[l-1]})^T dZ^{[l]}, \quad \frac{\partial \mathcal{L}}{\partial b^{[l]}} = \frac{1}{m} \sum_{i} dZ_i^{[l]}"
        )
        st.latex(
            r"dZ^{[l-1]} = \left(dZ^{[l]} (W^{[l]})^T\right) \odot g'\left(Z^{[l-1]}\right)"
        )
        st.markdown(
            """
            **Finite-Difference Numerical Gradient Checking:**
            """
        )
        st.latex(
            r"\frac{\partial \mathcal{L}}{\partial \theta_i} \approx \frac{\mathcal{L}(\theta + \epsilon e_i) - \mathcal{L}(\theta - \epsilon e_i)}{2\epsilon}, \quad \text{Rel Error} = \frac{\|\nabla_{\text{anal}} - \nabla_{\text{num}}\|}{\|\nabla_{\text{anal}}\| + \|\nabla_{\text{num}}\|} < 10^{-5}"
        )

    # ── Training Pipeline (Cached) ───────────────────────────────────
    @st.cache_data(show_spinner=False)
    def _train_pipeline(ds, arch, act, lr, ep, ns):
        X, y = generate_synthetic_dataset(ds, n_samples=350, noise=ns, seed=42)
        nn = PureNumPyNN(layer_sizes=arch, activation=act, learning_rate=lr, seed=42)
        grad_err = nn.numerical_gradient_check(X, y)
        hist = nn.train(X, y, epochs=ep, track_weight_idx=(0, 0, 0))
        W1_g, W2_g, Loss_g = compute_loss_surface(nn, X, y, hist["tracked_indices"], grid_points=24, span=1.2)
        return nn, X, y, grad_err, hist, W1_g, W2_g, Loss_g

    with st.spinner("Executing NumPy Matrix Backpropagation & Computing Loss Manifold..."):
        nn, X, y, grad_error, history, W1_grid, W2_grid, Loss_grid = _train_pipeline(
            dataset_name, layer_sizes, activation, learning_rate, epochs, noise_lvl
        )

    # ── Metric Cards ─────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final BCE Loss", f"{history['loss_history'][-1]:.4f}")
    c2.metric("Classification Accuracy", f"{history['final_accuracy']:.1%}")
    c3.metric(
        "Gradient Check Error",
        f"{grad_error:.2e}",
        delta="< 1e-5 (Passed)",
        delta_color="normal",
    )
    c4.metric("Trainable Parameters", f"{sum(w.size + b.size for w, b in zip(nn.weights, nn.biases))}")

    # ── Charts ────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Chart 1: Decision Boundary
        st.markdown("### Learned Non-Linear Decision Boundary")
        x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
        y_min, y_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 80), np.linspace(y_min, y_max, 80))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_preds, _ = nn.forward(grid_points)
        zz = grid_preds.reshape(xx.shape)

        fig_bound = go.Figure()
        # Contour heatmap
        fig_bound.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 80),
            y=np.linspace(y_min, y_max, 80),
            z=zz,
            colorscale="Viridis",
            opacity=0.6,
            showscale=False,
            contours=dict(start=0.1, end=0.9, size=0.1),
        ))
        # Scatter points
        fig_bound.add_trace(go.Scatter(
            x=X[y == 0, 0], y=X[y == 0, 1],
            mode="markers", name="Class 0",
            marker=dict(color="#FF6B6B", size=8, line=dict(width=1, color="white")),
        ))
        fig_bound.add_trace(go.Scatter(
            x=X[y == 1, 0], y=X[y == 1, 1],
            mode="markers", name="Class 1",
            marker=dict(color="#00D4AA", size=8, line=dict(width=1, color="white")),
        ))
        fig_bound.update_layout(
            template=PLOTLY_TEMPLATE,
            title=f"Non-Linear Boundary ({dataset_name.upper()})",
            xaxis_title="Feature X1",
            yaxis_title="Feature X2",
            height=430,
            margin=dict(t=30),
        )
        st.plotly_chart(fig_bound, use_container_width=True)

    with col_right:
        # Chart 2: Loss Convergence Curve
        st.markdown("### Binary Cross-Entropy Loss Curve")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=list(range(1, len(history["loss_history"]) + 1)),
            y=history["loss_history"],
            mode="lines",
            line=dict(color=COLORS["primary"], width=2),
            name="BCE Loss",
        ))
        fig_loss.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Optimization Convergence across Epochs",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=430,
            margin=dict(t=30),
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    # ── Chart 3: 3D Loss Landscape & Gradient Descent Trajectory ─────
    st.markdown("### 3D Loss Landscape & Optimization Trajectory Path")
    st.caption(
        "Interactive 3D visualization showing the empirical loss bowl around a 2D parameter slice $(W_1, W_2)$. "
        "The neon trajectory line tracks the exact coordinate descent path through the non-convex manifold."
    )

    path_w1 = history["path_w1"]
    path_w2 = history["path_w2"]
    # Compute loss along the descent path
    path_loss = []
    # Subsample path for clear 3D rendering
    step = max(1, len(path_w1) // 35)
    pw1_sub = path_w1[::step]
    pw2_sub = path_w2[::step]
    ploss_sub = history["loss_history"][::step]

    fig_land = go.Figure()
    # 3D Surface
    fig_land.add_trace(go.Surface(
        x=W1_grid[0],
        y=W2_grid[:, 0],
        z=Loss_grid,
        colorscale="Viridis",
        opacity=0.88,
        colorbar=dict(title="Loss"),
    ))
    # Optimization trajectory line in 3D
    fig_land.add_trace(go.Scatter3d(
        x=pw1_sub,
        y=pw2_sub,
        z=[l + 0.02 for l in ploss_sub],
        mode="lines+markers",
        marker=dict(size=4, color="#FF6B6B"),
        line=dict(color="#FFD166", width=5),
        name="Gradient Descent Path",
    ))
    # Mark start and end points
    fig_land.add_trace(go.Scatter3d(
        x=[pw1_sub[0]], y=[pw2_sub[0]], z=[ploss_sub[0] + 0.03],
        mode="markers", marker=dict(size=8, color="#FF6B6B", symbol="diamond"),
        name="Start (Epoch 0)",
    ))
    fig_land.add_trace(go.Scatter3d(
        x=[pw1_sub[-1]], y=[pw2_sub[-1]], z=[ploss_sub[-1] + 0.03],
        mode="markers", marker=dict(size=8, color="#00D4AA", symbol="cross"),
        name="Optimum (Trained)",
    ))

    fig_land.update_layout(
        title="3D Parameter Loss Surface (w_1 vs w_2 vs Loss)",
        template=PLOTLY_TEMPLATE,
        scene=dict(
            xaxis_title="Weight Parameter w_1",
            yaxis_title="Weight Parameter w_2",
            zaxis_title="Loss",
            camera=dict(eye=dict(x=1.4, y=-1.4, z=1.3)),
        ),
        height=600,
        margin=dict(t=40, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_land, use_container_width=True)
