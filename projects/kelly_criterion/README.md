# Kelly Criterion Bet Sizing Simulator

## Business Question
How much of your bankroll should you wager on each bet when you have a quantifiable edge? What happens when you over-bet?

## The Kelly Formula
$$f^* = \frac{b \cdot p - q}{b}$$

Where:
- $b$ = net decimal odds (payout per \$1 wagered)
- $p$ = probability of winning
- $q = 1 - p$ = probability of losing
- $f^*$ = optimal fraction of bankroll to bet

## Methodology
We simulate 1,000 independent wealth paths over 500 sequential bets for each fraction (5%, 20%, f*, 60%, 100%). Each path starts with \$100. A path is "ruined" when wealth drops below \$1.

## Key Insight
Over-betting (fraction > Kelly) leads to near-certain ruin over long horizons, even with a genuine statistical edge. The Kelly criterion maximises the geometric growth rate of wealth.

## Stack
- NumPy (vectorised path simulation)
- Plotly (wealth trajectories, box plots)
- Streamlit (interactive UI)
