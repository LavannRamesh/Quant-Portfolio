# Options Pricer -- Black-Scholes vs Monte Carlo

## Business Question
Can we validate the Black-Scholes closed-form option pricing formula using brute-force Monte Carlo simulation? How many paths are needed for convergence?

## Mathematical Methodology

### Black-Scholes (European Options)
$$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$
$$C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)$$

### Monte Carlo
$$S_T = S \cdot \exp\left((r - \sigma^2/2)T + \sigma\sqrt{T} \cdot Z\right), \quad Z \sim \mathcal{N}(0,1)$$
$$C_{MC} = e^{-rT} \cdot \frac{1}{N} \sum_{i=1}^{N} \max(S_T^{(i)} - K, 0)$$

### Convergence
By the CLT, the standard error of the MC estimate is $\text{SE} = \sigma_{\text{payoff}} / \sqrt{N}$, giving O(1/sqrt(N)) convergence.

## Validation
MC price converges to within 0.5% of BS at N = 1,000,000 paths.

## Stack
- SciPy (normal CDF/PDF for BS)
- NumPy (vectorised MC simulation)
- Plotly (convergence plot, 3D surface)
- Streamlit (interactive UI)
