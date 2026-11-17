# 2D Damped Wave Equation Simulator

## Mathematical Formulation
The damped 2D wave equation models a vibrating elastic membrane (such as a drum skin) subject to tension, inertia, and linear viscous friction:
$$\frac{\partial^2 u}{\partial t^2} + \gamma \frac{\partial u}{\partial t} = c^2 \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$
where:
- $u(x, y, t)$ is the transverse membrane displacement.
- $c = \sqrt{T/\rho}$ is the acoustic phase velocity.
- $\gamma \ge 0$ is the viscous damping coefficient.

## Finite Difference Discretization
Using 2nd-order central differences in both time and space:
$$\frac{u_{i,j}^{n+1} - 2u_{i,j}^n + u_{i,j}^{n-1}}{\Delta t^2} + \gamma \frac{u_{i,j}^{n+1} - u_{i,j}^{n-1}}{2\Delta t} = \frac{c^2}{h^2} \left(u_{i+1,j}^n + u_{i-1,j}^n + u_{i,j+1}^n + u_{i,j-1}^n - 4u_{i,j}^n\right)$$

Solving explicitly for the future state $u_{i,j}^{n+1}$:
$$u_{i,j}^{n+1} = \frac{2u_{i,j}^n - \left(1 - \frac{\gamma \Delta t}{2}\right)u_{i,j}^{n-1} + r^2 L_{i,j}^n}{1 + \frac{\gamma \Delta t}{2}}$$
where $r = \frac{c \Delta t}{h}$ is the Courant number and $L_{i,j}^n$ is the discrete 5-point Laplacian.

## Courant-Friedrichs-Lewy (CFL) Stability
By Von Neumann spectral analysis, explicit time integration of 2D hyperbolic wave PDEs is unconditionally stable if and only if:
$$r = \frac{c \Delta t}{h} \le \frac{1}{\sqrt{2}} \approx 0.7071$$
Exceeding this critical limit triggers exponential numerical explosive divergence. The solver automatically guarantees $\Delta t$ satisfies CFL stability.

## Boundary Conditions
Fixed rim Dirichlet boundaries:
$$u(x, y, t) = 0 \quad \forall (x, y) \in \partial\Omega$$

## Stack
- `numpy` (vectorized slice updates for the discrete spatial Laplacian)
- `plotly` (interactive 3D surface rendering with time-step playback)
- `streamlit` (physical parameter tuning)
