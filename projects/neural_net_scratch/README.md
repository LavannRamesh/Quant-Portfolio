# Neural Network from Scratch (Pure NumPy)

## Mathematical Formulation
This module implements a complete, zero-dependency multi-layer perceptron (MLP) written strictly in pure NumPy vector calculus.

### Forward Propagation
For layer $l \in \{1, \dots, L\}$:
$$Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$$
$$A^{[l]} = g\left(Z^{[l]}\right)$$
Where $g(z) = \max(0, z)$ for ReLU activations, and $g(z) = \frac{1}{1 + e^{-z}}$ for Sigmoid output classification.

### Binary Cross-Entropy Loss
$$\mathcal{L}(y, \hat{y}) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \ln \hat{y}^{(i)} + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right]$$

### Analytical Backpropagation via Matrix Calculus
Output layer error:
$$dZ^{[L]} = A^{[L]} - Y$$
Layer gradients ($l = L, \dots, 1$):
$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{1}{m} (A^{[l-1]})^T dZ^{[l]}, \quad \frac{\partial \mathcal{L}}{\partial b^{[l]}} = \frac{1}{m} \sum_{i=1}^m dZ_{i,:}^{[l]}$$
Hidden activation errors:
$$dZ^{[l-1]} = \left(dZ^{[l]} (W^{[l]})^T\right) \odot g'\left(Z^{[l-1]}\right)$$

### Numerical Gradient Checking
To strictly validate the correctness of the analytical backpropagation implementation, numerical gradients are computed via central finite differences:
$$\left[\nabla_{\text{num}} \mathcal{L}\right]_i = \frac{\mathcal{L}(\theta + \epsilon e_i) - \mathcal{L}(\theta - \epsilon e_i)}{2\epsilon}$$
The relative difference metric:
$$\text{Rel Error} = \frac{\|\nabla_{\text{anal}} - \nabla_{\text{num}}\|_2}{\|\nabla_{\text{anal}}\|_2 + \|\nabla_{\text{num}}\|_2} < 10^{-5}$$
guarantees that no backpropagation bug or sign error exists.

### Loss Landscape Visualization
We evaluate the non-convex loss manifold across a 2D parameter slice $(w_1, w_2)$ around the optimal weights, overlaying the exact 3D descent trajectory taken by momentum gradient descent.

## Stack
- `numpy` (pure linear algebra: matrix multiplications, broadcast addition, chain rule)
- `plotly` (interactive 3D loss surface & 2D decision contours)
- `streamlit` (architecture selection & hyperparameter controls)
