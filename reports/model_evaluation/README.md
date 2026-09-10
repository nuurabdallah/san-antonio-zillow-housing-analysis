# Model Evaluation

This folder contains the final evaluation outputs for the selected housing-price prediction model.

The evaluation focuses on predictive performance, generalization, residual behavior, price-segment errors, calibration, and model interpretation.

## Selected Model

The final selected model is a **Tuned Gradient Boosting** model trained to predict `logPrice`.

The model was selected because it produced the strongest cross-validated performance among the candidate models evaluated during model development.

## Holdout Performance

The final model was evaluated on a holdout test set that was not used during model fitting.

Key holdout results include:

- R²: **0.9068**
- Dollar MAE: **$52,141**
- MAPE: **14.45%**
- Median Absolute Error: **$33,235**
- Mean Dollar Error: **+$14,649**

The positive mean dollar error indicates that, on average, predictions were below observed listing prices.

## Cross-Validation Context

The selected model achieved the following 10-fold cross-validation results during model development:

- Mean CV R²: **0.8286**
- Mean CV RMSE: **0.2558**
- Mean CV MAE: **0.1800**

The difference between training and cross-validation performance indicates some degree of overfitting, although the model retained strong predictive performance on unseen data.

## Error Analysis

Evaluation of prediction errors showed that model performance varies across the housing-price distribution.

The model tends to:

- Overpredict lower-priced properties
- Underpredict higher-priced properties
- Compress predictions toward the center of the observed price distribution

Examples of observed directional bias include:

- Properties below $150K: approximately **89.29%** were overpredicted.
- Properties from $500K–$1M: approximately **65.81%** were underpredicted.
- Properties above $2M: approximately **80%** were underpredicted.

These results indicate that the model performs well overall but has difficulty capturing the extremes of the San Antonio housing market.

## Calibration

The model demonstrated strong overall calibration, with systematic compression at the price extremes.

Calibration results included:

- Log-price calibration slope: **1.0428**
- Log-price calibration R²: **0.9083**
- Dollar calibration slope: **1.1320**
- Dollar calibration R²: **0.9762**

The highest prediction decile underpredicted actual prices by approximately **$95,378 on average**.

## Residual Diagnostics

Residual analysis was used to evaluate whether prediction errors showed systematic patterns.

The diagnostics identified:

- Heteroscedasticity
- Non-normal residual behavior
- Residual skewness
- Excess kurtosis
- Influential observations

The strongest relationship identified between error magnitude and a predictor was with `taxAssessedValue`, with a Spearman correlation of approximately **−0.1761**.

These diagnostics are important because strong overall predictive performance does not eliminate the possibility of systematic error patterns.

## Model Interpretation

The final model's most important predictive variables were:

1. `taxAssessedValue`
2. `area`
3. `lotAreaSqFt`
4. `daysOnZillow`
5. `beds`
6. `baths`

Feature importance and permutation importance provide complementary views of the variables contributing to model predictions


