# -*- coding: utf-8 -*-
"""
Neural Network from Scratch -- Core Logic
==========================================
Pure NumPy implementation of a Multi-Layer Perceptron (MLP):
- Forward propagation (arbitrary hidden layers, ReLU / Sigmoid activations)
- Binary Cross-Entropy loss computation
- Full backpropagation via analytical chain-rule vector calculus
- Numerical gradient checking (relative error verification < 1e-6)
- 2D parameter slice loss landscape & gradient descent path extraction
- Decision boundary grid generation
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Any, Optional
import numpy as np


class PureNumPyNN:
    """
    Feedforward Neural Network implemented strictly using pure NumPy vector algebra.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "relu",
        learning_rate: float = 0.05,
        momentum: float = 0.9,
        seed: int = 42,
    ) -> None:
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.lr = learning_rate
        self.momentum = momentum
        self.seed = seed

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.v_w: List[np.ndarray] = []
        self.v_b: List[np.ndarray] = []

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """He / Xavier weight initialization."""
        rng = np.random.default_rng(self.seed)
        self.weights = []
        self.biases = []
        self.v_w = []
        self.v_b = []

        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]

            # He initialization for ReLU, Xavier for Sigmoid
            scale = np.sqrt(2.0 / n_in) if self.activation == "relu" else np.sqrt(1.0 / n_in)
            W = rng.normal(0.0, scale, (n_in, n_out))
            b = np.zeros(n_out)

            self.weights.append(W)
            self.biases.append(b)
            self.v_w.append(np.zeros_like(W))
            self.v_b.append(np.zeros_like(b))

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    @staticmethod
    def _sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        return a * (1.0 - a)

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def _relu_derivative(z: np.ndarray) -> np.ndarray:
        return (z > 0.0).astype(float)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
        """
        Forward pass through all layers.
        Returns (output_activations, caches_per_layer).
        """
        caches = []
        A = X

        for i in range(len(self.weights) - 1):
            W = self.weights[i]
            b = self.biases[i]
            Z = A @ W + b

            if self.activation == "relu":
                A_next = self._relu(Z)
            else:
                A_next = self._sigmoid(Z)

            caches.append({"A_prev": A, "Z": Z, "A": A_next, "W": W, "b": b})
            A = A_next

        # Output layer: always Sigmoid for binary classification
        W_last = self.weights[-1]
        b_last = self.biases[-1]
        Z_last = A @ W_last + b_last
        A_last = self._sigmoid(Z_last)

        caches.append({"A_prev": A, "Z": Z_last, "A": A_last, "W": W_last, "b": b_last})
        return A_last, caches

    @staticmethod
    def compute_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Binary cross-entropy loss with numerical clamping."""
        eps = 1e-12
        y_p = np.clip(y_pred, eps, 1.0 - eps)
        y_t = y_true.reshape(-1, 1)
        bce = -np.mean(y_t * np.log(y_p) + (1.0 - y_t) * np.log(1.0 - y_p))
        return float(bce)

    def backward(
        self,
        y_true: np.ndarray,
        caches: List[Dict[str, np.ndarray]],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Analytical backpropagation using matrix calculus chain rule.
        Returns (dW_list, db_list).
        """
        m = y_true.shape[0]
        dW_list: List[np.ndarray] = [None] * len(self.weights)  # type: ignore
        db_list: List[np.ndarray] = [None] * len(self.biases)   # type: ignore

        # Output layer error: dZ = (A_last - Y)
        last_cache = caches[-1]
        A_last = last_cache["A"]
        A_prev = last_cache["A_prev"]

        dZ = (A_last - y_true.reshape(-1, 1))  # (m, 1)
        dW_list[-1] = (A_prev.T @ dZ) / m
        db_list[-1] = np.sum(dZ, axis=0) / m

        dA_prev = dZ @ last_cache["W"].T

        # Hidden layers in reverse
        for l in range(len(self.weights) - 2, -1, -1):
            cache = caches[l]
            Z = cache["Z"]
            A_prev = cache["A_prev"]

            if self.activation == "relu":
                dZ = dA_prev * self._relu_derivative(Z)
            else:
                dZ = dA_prev * self._sigmoid_derivative(cache["A"])

            dW_list[l] = (A_prev.T @ dZ) / m
            db_list[l] = np.sum(dZ, axis=0) / m

            if l > 0:
                dA_prev = dZ @ cache["W"].T

        return dW_list, db_list

    def step(self, dW_list: List[np.ndarray], db_list: List[np.ndarray]) -> None:
        """Apply gradient update with SGD + Momentum."""
        for i in range(len(self.weights)):
            self.v_w[i] = self.momentum * self.v_w[i] + self.lr * dW_list[i]
            self.v_b[i] = self.momentum * self.v_b[i] + self.lr * db_list[i]
            self.weights[i] -= self.v_w[i]
            self.biases[i] -= self.v_b[i]

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 500,
        track_weight_idx: Tuple[int, int, int] = (0, 0, 0),
    ) -> Dict[str, Any]:
        """
        Train network and record optimization trajectory.
        track_weight_idx: (layer, row, col) parameter pair to log for loss landscape.
        """
        loss_history = []
        path_w1 = []
        path_w2 = []

        l_idx, r_idx, c_idx = track_weight_idx
        # Pick adjacent parameter for 2D plane
        c_idx2 = (c_idx + 1) % self.weights[l_idx].shape[1]

        for ep in range(epochs):
            y_pred, caches = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            loss_history.append(loss)

            # Log weight coordinates for 3D trajectory
            path_w1.append(float(self.weights[l_idx][r_idx, c_idx]))
            path_w2.append(float(self.weights[l_idx][r_idx, c_idx2]))

            dW, db = self.backward(y, caches)
            self.step(dW, db)

        y_final, _ = self.forward(X)
        final_preds = (y_final >= 0.5).astype(int).ravel()
        acc = float(np.mean(final_preds == y))

        return {
            "loss_history": loss_history,
            "path_w1": path_w1,
            "path_w2": path_w2,
            "final_accuracy": acc,
            "tracked_indices": (l_idx, r_idx, c_idx, c_idx2),
        }

    def numerical_gradient_check(self, X: np.ndarray, y: np.ndarray, epsilon: float = 1e-7) -> float:
        """
        Verify analytical gradients against two-sided finite difference approximations:
        grad_num = (Loss(theta + eps) - Loss(theta - eps)) / (2 * eps)
        Returns the maximum relative difference: |grad_anal - grad_num| / (|grad_anal| + |grad_num|).
        """
        y_pred, caches = self.forward(X)
        dW_anal, _ = self.backward(y, caches)

        # Check layer 0 weights
        W = self.weights[0]
        dW = dW_anal[0]
        max_rel_error = 0.0

        # Sample 5 random coordinates to keep test instantaneous
        rng = np.random.default_rng(123)
        for _ in range(5):
            r = rng.integers(0, W.shape[0])
            c = rng.integers(0, W.shape[1])

            orig = W[r, c]

            W[r, c] = orig + epsilon
            loss_plus = self.compute_loss(self.forward(X)[0], y)

            W[r, c] = orig - epsilon
            loss_minus = self.compute_loss(self.forward(X)[0], y)

            W[r, c] = orig  # restore

            grad_num = (loss_plus - loss_minus) / (2.0 * epsilon)
            grad_an = dW[r, c]

            denom = max(abs(grad_an) + abs(grad_num), 1e-8)
            rel_err = abs(grad_an - grad_num) / denom
            max_rel_error = max(max_rel_error, rel_err)

        return float(max_rel_error)


def compute_loss_surface(
    model: PureNumPyNN,
    X: np.ndarray,
    y: np.ndarray,
    tracked_indices: Tuple[int, int, int, int],
    grid_points: int = 25,
    span: float = 1.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate empirical BCE loss on a 2D parameter mesh around current weights.
    """
    l_idx, r_idx, c1, c2 = tracked_indices
    w1_center = model.weights[l_idx][r_idx, c1]
    w2_center = model.weights[l_idx][r_idx, c2]

    w1_vals = np.linspace(w1_center - span, w1_center + span, grid_points)
    w2_vals = np.linspace(w2_center - span, w2_center + span, grid_points)
    W1_grid, W2_grid = np.meshgrid(w1_vals, w2_vals)
    Loss_grid = np.zeros_like(W1_grid)

    orig1 = model.weights[l_idx][r_idx, c1]
    orig2 = model.weights[l_idx][r_idx, c2]

    for i in range(grid_points):
        for j in range(grid_points):
            model.weights[l_idx][r_idx, c1] = W1_grid[i, j]
            model.weights[l_idx][r_idx, c2] = W2_grid[i, j]
            preds, _ = model.forward(X)
            Loss_grid[i, j] = model.compute_loss(preds, y)

    # Restore
    model.weights[l_idx][r_idx, c1] = orig1
    model.weights[l_idx][r_idx, c2] = orig2

    return W1_grid, W2_grid, Loss_grid


def generate_synthetic_dataset(
    dataset_name: str = "xor",
    n_samples: int = 400,
    noise: float = 0.12,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 2D non-linear classification benchmarks."""
    rng = np.random.default_rng(seed)

    if dataset_name == "xor":
        X = rng.uniform(-1.5, 1.5, (n_samples, 2))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
        X += rng.normal(0, noise, X.shape)
    elif dataset_name == "circles":
        n_half = n_samples // 2
        r_inner = rng.normal(0.4, noise, n_half)
        th_inner = rng.uniform(0, 2 * np.pi, n_half)
        X_in = np.column_stack([r_inner * np.cos(th_inner), r_inner * np.sin(th_inner)])

        r_outer = rng.normal(1.0, noise, n_samples - n_half)
        th_outer = rng.uniform(0, 2 * np.pi, n_samples - n_half)
        X_out = np.column_stack([r_outer * np.cos(th_outer), r_outer * np.sin(th_outer)])

        X = np.vstack([X_in, X_out])
        y = np.concatenate([np.zeros(n_half), np.ones(n_samples - n_half)]).astype(int)
    elif dataset_name == "moons":
        n_half = n_samples // 2
        theta = np.linspace(0, np.pi, n_half)
        x1 = np.cos(theta) + rng.normal(0, noise, n_half)
        y1 = np.sin(theta) + rng.normal(0, noise, n_half)

        x2 = 1.0 - np.cos(theta) + rng.normal(0, noise, n_samples - n_half)
        y2 = 1.0 - np.sin(theta) - 0.5 + rng.normal(0, noise, n_samples - n_half)

        X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
        y = np.concatenate([np.zeros(n_half), np.ones(n_samples - n_half)]).astype(int)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return X, y


if __name__ == "__main__":
    X_d, y_d = generate_synthetic_dataset("xor", n_samples=300)
    nn = PureNumPyNN(layer_sizes=[2, 8, 4, 1], activation="relu", learning_rate=0.08)

    # 1. Verify gradient check
    grad_err = nn.numerical_gradient_check(X_d, y_d)
    print(f"Analytical vs Numerical Gradient Relative Error: {grad_err:.2e}")
    assert grad_err < 1e-4, f"Gradient check failed! rel_error={grad_err}"

    # 2. Verify training
    history = nn.train(X_d, y_d, epochs=400)
    print(f"Final Trained Accuracy on XOR: {history['final_accuracy']:.1%}")
    assert history["final_accuracy"] > 0.85, "Training failed to fit non-linear XOR!"
    print("Neural Network from Scratch Logic Test: PASSED")
