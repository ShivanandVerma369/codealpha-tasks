# Sales Prediction & Advertising Analytics Report

## 1. Model Performance Summary (Test Set, N=40)

| Model | MAE | MSE | RMSE | R2 |
| --- | --- | --- | --- | --- |
| Gradient Boosting | 0.6187 | 0.5326 | 0.7298 | 0.9831 |
| Random Forest | 0.6203 | 0.5909 | 0.7687 | 0.9813 |
| Decision Tree | 0.9850 | 2.1750 | 1.4748 | 0.9311 |
| Linear Regression | 1.4608 | 3.1741 | 1.7816 | 0.8994 |

**Best Performing Model**: `Gradient Boosting` (Selected based on highest $R^2$ and lowest RMSE).

## 2. Linear Regression Coefficients

| Feature | Coefficient |
| --- | --- |
| Intercept | 2.9791 |
| TV | 0.0447 |
| Radio | 0.1892 |
| Newspaper | 0.0028 |

### Coefficient Interpretation (Holding other variables constant):
- **Intercept (2.9791)**: Base predicted sales when spend on all three channels is 0.
- **TV (0.0447)**: Within this fitted linear model, each 1-unit increase in TV spend is associated with a 0.0447 unit increase in predicted sales.
- **Radio (0.1892)**: Within this fitted linear model, each 1-unit increase in Radio spend is associated with a 0.1892 unit increase in predicted sales.
- **Newspaper (0.0028)**: Within this fitted linear model, each 1-unit increase in Newspaper spend is associated with a 0.0028 unit change in predicted sales.

## 3. Feature Importance (Gradient Boosting)

| Channel | Importance Score | Percentage (%) |
| --- | --- | --- |
| TV | 0.6126 | 61.26% |
| Radio | 0.3809 | 38.09% |
| Newspaper | 0.0065 | 0.65% |

## 4. Actual vs Predicted Sample Table (First 15 Test Observations)

| TV | Radio | Newspaper | Actual Sales | Predicted Sales | Absolute Error |
| --- | --- | --- | --- | --- | --- |
| 163.3 | 31.6 | 52.9 | 16.9 | 17.42 | 0.52 |
| 195.4 | 47.7 | 52.9 | 22.4 | 21.75 | 0.65 |
| 292.9 | 28.3 | 43.2 | 21.4 | 20.27 | 1.13 |
| 11.7 | 36.9 | 45.2 | 7.3 | 6.52 | 0.78 |
| 220.3 | 49.0 | 3.2 | 24.7 | 23.62 | 1.08 |
| 75.1 | 35.0 | 52.7 | 12.6 | 13.05 | 0.45 |
| 216.8 | 43.9 | 27.2 | 22.3 | 22.85 | 0.55 |
| 50.0 | 11.6 | 18.4 | 8.4 | 9.44 | 1.04 |
| 222.4 | 3.4 | 13.1 | 11.5 | 11.9 | 0.4 |
| 175.1 | 22.5 | 31.5 | 14.9 | 15.87 | 0.97 |
| 31.5 | 24.6 | 2.2 | 9.5 | 8.06 | 1.44 |
| 56.2 | 5.7 | 29.7 | 8.7 | 9.21 | 0.51 |
| 234.5 | 3.4 | 84.8 | 11.9 | 12.36 | 0.46 |
| 5.4 | 29.9 | 9.4 | 5.3 | 4.6 | 0.7 |
| 139.5 | 2.1 | 26.6 | 10.3 | 10.4 | 0.1 |

## 5. Key Marketing Insights
1. **TV and Radio** show the strongest positive relationship and contribution to predicted sales.
2. **Newspaper** shows the weakest relationship with sales and near-zero coefficient / minimal feature importance score.
3. **Non-linear Models** (e.g. Random Forest / Gradient Boosting) capture interaction effects between TV and Radio advertising.
4. Correlation and model coefficients describe observed relationships in data, not guaranteed real-world causal impacts.
