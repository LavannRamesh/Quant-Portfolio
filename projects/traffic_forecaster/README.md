# Traffic Speed Forecaster (PeMS / METR-LA)

## Business Question
How can transportation operators and route-planning algorithms (like Google Maps ETA engines) anticipate congestion 15 to 60 minutes in advance using real-time highway loop detector telemetry?

## Dataset & Telemetry
- **Source**: California Performance Measurement System (PeMS) / Los Angeles County METR-LA benchmark.
- **Attributes**: 5-minute sampling interval reporting aggregate vehicle velocity (mph), diurnal rush-hour bottlenecks, and incident disruptions.

## Machine Learning Methodology

### Feature Engineering
1. **Autoregressive Lags**: $v_{t-5\text{m}}, v_{t-10\text{m}}, \dots, v_{t-60\text{m}}$ capturing temporal momentum and dissipation waves.
2. **Rolling Statistics**: 30-minute and 60-minute trailing rolling mean, standard deviation, and minimum speeds to capture shock volatility.
3. **Cyclical Temporal Encodings**:
   $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
   $$\text{dow\_sin} = \sin\left(\frac{2\pi \cdot \text{dow}}{7}\right), \quad \text{dow\_cos} = \cos\left(\frac{2\pi \cdot \text{dow}}{7}\right)$$

### Predictive Modeling
Gradient-boosted decision trees via **LightGBM** optimizing Mean Squared Error. Decision trees capture non-linear phase transitions (e.g., the sudden collapse from free-flow laminar traffic to turbulent stop-and-go congestion).

### Out-of-Sample Evaluation
- **MAE** (Mean Absolute Error in mph)
- **RMSE** (Root Mean Squared Error)
- **$R^2$ Score** (Variance explained)
- **MAPE** (Mean Absolute Percentage Error)

## Resiliency & Offline Fallback
The data loader queries live PeMS endpoints. If credentials or network partitions intervene, a realistic, physically-consistent simulation (calibrated to PeMS distributions with Poisson incident injection) is synthesized to guarantee continuous offline execution.

## Stack
- `lightgbm` (high-performance gradient boosting)
- `pandas` & `numpy` (lag vectorization & cyclical transforms)
- `plotly` (interactive forecast tracking)
- `streamlit` (multi-horizon UI)
