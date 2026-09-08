# Part 10 — Model Evaluation Key Findings

## Holdout Performance

- R²: **0.7141**
- RMSE on logPrice: **0.2926**
- MAE on logPrice: **0.1853**
- Dollar RMSE: **$91,603.50**
- Dollar MAE: **$57,784.38**
- Median absolute dollar error: **$35,237.57**
- Dollar MAPE: **21.55%**
- Median APE: **11.44%**

## Calibration

- Slope: **1.0842**
- Intercept: **$-18,548.73**
- Calibration R²: **0.8320**

## Residual Diagnostics

- Mean residual: **-0.0102**
- Residual standard deviation: **0.2934**
- Pearson residual/predicted p-value: **0.222148**
- Spearman absolute-residual/predicted p-value: **0.0258602**
- Shapiro-Wilk p-value: **1.96177e-10**

## Feature Importance

The strongest built-in model feature was **taxAssessedValue** with importance **0.796**.

Feature importance indicates predictive contribution within this fitted model; it does not establish causal influence.

## Price-Segment Performance

The price-segment output provides mean absolute and percentage errors across four holdout price ranges. Dollar error and percentage error should be considered together because higher-priced properties can naturally have larger dollar errors.

## Largest Errors

The `largest_prediction_errors.csv` file contains the 20 holdout observations with the largest absolute prediction errors for investigation.

## Overall Assessment

The selected model demonstrates meaningful predictive performance on unseen holdout properties. The evaluation also identifies residual behavior, calibration, price ranges, and observations where predictions are less accurate.

The model should be presented as a portfolio predictive-analysis model, not as a guaranteed property valuation system.
