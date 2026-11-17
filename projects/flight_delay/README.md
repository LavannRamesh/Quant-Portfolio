# Flight Delay Predictor (US BTS On-Time Performance)

## Business Question
Can we forecast whether a scheduled domestic commercial flight will encounter a delay exceeding 15 minutes based solely on pre-departure features (carrier, route topology, departure time, seasonality)?

## Dataset & Industry Standards
- **Source**: US Bureau of Transportation Statistics (BTS) On-Time Reporting Carrier Flight Performance.
- **FAA/DOT Definition**: A flight is officially classified as "delayed" if its actual arrival or departure exceeds the scheduled gate time by 15 or more minutes ($\text{ARR\_DELAY} \ge 15$).

## Machine Learning Methodology

### Temporal Cyclical Encodings
Linear representations of cyclic features (hour 23 to 0, month 12 to 1) introduce boundary discontinuities. We project time features onto continuous harmonic coordinates:
$$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
$$\text{dow\_sin} = \sin\left(\frac{2\pi \cdot \text{dow}}{7}\right), \quad \text{dow\_cos} = \cos\left(\frac{2\pi \cdot \text{dow}}{7}\right)$$

### Predictive Architecture: LightGBM Binary Classifier
- Binary log-loss objective with class-weight balancing to counteract baseline label skew (historically ~20-22% delay rate).
- Tree splitting on carrier-airport interactions directly discovers hub congestion bottlenecks (such as ORD or EWR slot constraints).

### Evaluation Metrics
- **Receiver Operating Characteristic (ROC-AUC)**: Measures ranking quality across discrimination thresholds.
- **Precision / Recall / F1-Score**: Evaluates performance on positive delay classifications.

## Resiliency & Fallback
The data loader queries live TranStats endpoints. If ASP.NET tokens or federal server blocks trigger, the system seamlessly transitions to a calibrated US BTS generator reflecting genuine carrier delay weights, hub congestion factors, and diurnal propagation curves.

## Stack
- `lightgbm` (gradient-boosted classification)
- `pandas` & `numpy` (harmonic encodings)
- `plotly` (interactive gauges, ROC curve, and risk curves)
- `streamlit` (flight search form)
