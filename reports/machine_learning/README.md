# Machine Learning

This folder contains the machine-learning development outputs for the San Antonio Zillow Housing Analysis.

The objective was to predict listing price using property and market characteristics while avoiding target leakage from variables derived directly from listing price or Zestimate information.

## Modeling Dataset

The final analytical dataset contains:

- 810 properties
- 810 unique Zillow property IDs
- 0 duplicate property IDs

After applying complete-case requirements to the approved modeling variables:

- 731 complete modeling cases were available for machine learning.
- Rows with missing required modeling variables were excluded rather than having values fabricated or imputed for this analysis.

## Target

The modeling target is:

```text
logPrice
```

Using the log-transformed listing price helps reduce the influence of the highly skewed housing-price distribution.

## Approved Predictors

The final model uses six predictors:

```text
area
beds
baths
lotAreaSqFt
taxAssessedValue
daysOnZillow
```

The model intentionally excludes variables that could introduce target leakage, including:

```text
price
pricePerSqFt
zestimate
zestimate_gap
zestimate_gap_pct
deal_score
opportunity_status
```

## Models Evaluated

The project compared multiple model families.

### Linear Models

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net

### Tree-Based Models

- Random Forest
- Gradient Boosting
- Tuned Gradient Boosting

## Model Selection

A shuffled 10-fold cross-validation procedure using `random_state = 42` was used to compare candidate models.

| Model | Mean CV R² |
|---|---:|
| Linear Regression | 0.3896 |
| Ridge | 0.3935 |
| Lasso | 0.3953 |
| Elastic Net | 0.3937 |
| Random Forest | 0.8128 |
| Gradient Boosting | 0.8151 |
| **Tuned Gradient Boosting** | **0.8286** |

The Tuned Gradient Boosting model produced the strongest cross-validated performance.

## Selected Model

The final model uses:

```text
n_estimators = 100
learning_rate = 0.05
max_depth = 3
min_samples_split = 5
min_samples_leaf = 1
subsample = 1.0
random_state = 42
```

### Cross-Validated Performance

- Mean CV R²: **0.8286**
- CV R² SD: **0.0783**
- Mean CV RMSE: **0.2558**
- Mean CV MAE: **0.1800**
- Mean training R²: **0.9090**
- Generalization gap: **0.0805**

The training-to-cross-validation difference indicates some overfitting, while the model still substantially outperformed the linear baselines.

## Model Interpretation

The strongest predictive signals were:

1. Tax Assessed Value
2. Living Area
3. Lot Area
4. Days on Zillow
5. Bathrooms
6. Bedrooms

Multiple importance approaches produced the same overall ranking, with tax assessed value and living area substantially more influential than the remaining predictors.

Feature importance represents predictive contribution within the fitted model and does not establish causation.

## Key Outputs

### Model Development

`model_development_comparison.csv`

Contains the cross-validation comparison of the candidate models.

### Train/Test Split

`ml_train_test_split.csv`

Documents the modeling population and train/test allocation.

### Feature Importance

`selected_model_feature_importance.csv`

Contains the selected Gradient Boosting model's built-in feature importance.

### Permutation Importance

`selected_model_permutation_importance.csv`

Contains permutation-based importance estimates for the selected model.

### Holdout Predictions

`selected_model_holdout_predictions.csv`

Contains observed and predicted values for the holdout observations.

## Charts

The `charts/` directory contains visualizations covering:

- Cross-validated R²
- Cross-validated RMSE
- Model generalization gap
- Holdout predictions
- Residual behavior
- Feature importance
- Permutation importance

## Detailed Documentation

For the full modeling methodology, see:

`MACHINE_LEARNING_METHODOLOGY.md`

For the summarized findings and interpretation, see:

`MACHINE_LEARNING_KEY_FINDINGS.md`

## Reproducibility

The complete machine-learning workflow is implemented in:

```text
python/09_machine_learning.py
```

The model-development workflow uses the feature-engineered analytical dataset and explicitly validates the approved predictor set to prevent target leakage.

## Important Interpretation Note

This model is a portfolio predictive-analysis model designed to demonstrate the end-to-end application of data preparation, model comparison, cross-validation, predictive modeling, and model interpretation.

It should not be presented as a guaranteed property valuation system.


