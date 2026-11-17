# Airbnb Pricing Engine with SHAP Attribution

## Business Question
How can short-term rental property owners price their listings competitively to maximize revenue, and precisely quantify which amenities, geographic factors, or reviews add the most marginal rental income?

## Dataset
- **Source**: Inside Airbnb (New York City Open Listing Benchmark).
- **Core Features**: Neighborhood district, room category (`Entire home/apt`, `Private room`, `Shared room`), capacity (`accommodates`, `bedrooms`, `bathrooms`), distance to metropolitan center, amenities (WiFi, AC, kitchen), review scores, and Superhost credentials.

## Machine Learning Methodology

### Target Transformation
Rental price $y$ exhibits heavy positive skewness. To preserve homoscedasticity and model percentage-based premiums, the objective trains on log price:
$$\tilde{y} = \ln(y)$$
$$\hat{y} = \exp\left(f_{\text{XGBoost}}(X)\right)$$

### Predictive Architecture: XGBoost Regressor
Extreme Gradient Boosting with regularized tree depth and subsampling to eliminate overfitting on outlier luxury properties.

### Interpretability: SHAP (Shapley Additive exPlanations)
Based on cooperative game theory, SHAP decomposes the prediction into fair additive allocations per feature:
$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i$$
- **Global Beeswarm**: Reveals macroeconomic drivers (e.g., room type and central location dominate variance).
- **Local Waterfall / Breakdown**: Provides hosts actionable ROI metrics for incremental upgrades (e.g., how much adding AC or earning Superhost status will elevate their nightly rate).

## Stack
- `xgboost` (gradient-boosted trees)
- `shap` (TreeExplainer game-theoretic attributions)
- `plotly` & `matplotlib` (beeswarm & prediction charts)
- `streamlit` (pricing simulator)
