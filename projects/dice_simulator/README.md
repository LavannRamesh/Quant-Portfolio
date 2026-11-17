# Dice Game Simulator

## Business Question
Given an asymmetric payoff structure on a fair six-sided die, can we verify the theoretical Expected Value through large-scale Monte Carlo simulation?

## Payoff Rules
| Die Roll | Payoff |
|----------|--------|
| 1 or 2   | -$1    |
| 3 or 4   |  $0    |
| 5        | +$1    |
| 6        | +$2    |

## Mathematical Methodology

### Analytical Expected Value
$$E[X] = \sum_{i=1}^{6} P(i) \cdot \text{payoff}(i) = \frac{2}{6}(-1) + \frac{2}{6}(0) + \frac{1}{6}(1) + \frac{1}{6}(2) = \frac{1}{6} \approx 0.1667$$

### Monte Carlo Verification
We simulate N = 100,000 independent rolls using NumPy vectorisation and track the running mean. By the Law of Large Numbers, the sample mean converges to E[X].

## Validation
The simulated E[X] is asserted to be within 2% of the analytical value at N = 100,000.

## Stack
- NumPy (vectorised simulation)
- Plotly (convergence & histogram charts)
- Streamlit (interactive UI)
