# Part 9 — Machine Learning Methodology

## Purpose

Part 9 develops and compares machine-learning models for predicting
Zillow listing prices in San Antonio.

The modeling target is `logPrice`, the natural-log transformation of
listing price.

## Source Dataset

- Source: `data/processed/San_Antonio_Zillow_Feature_Engineered.csv`
- Full analytical population: 810 properties
- Complete machine-learning population: 731 properties
- Target: `logPrice`

## Approved Predictors

The model-development feature set is:

- `area`
- `beds`
- `baths`
- `lotAreaSqFt`
- `taxAssessedValue`
- `daysOnZillow`

## Target Leakage Controls

The following variables are intentionally excluded from the predictor set:

- `price`
- `pricePerSqFt`
- `zestimate`
- `zestimate_gap`
- `zestimate_gap_pct`
- `deal_score`
- `opportunity_status`

These variables either directly contain the target, are calculated from
the target, or are business metrics derived from listing price and therefore
could create target leakage.

## Missing Data

Machine-learning development uses complete cases for the target and all six
approved predictors. The complete-case subset is used only for model
training and comparison.

The original analytical dataset is not modified, and missing values remain
missing in the source dataset.

## Train/Test Design

- Holdout test proportion: 20%
- Random state: 42
- Cross-validation: 10-fold shuffled K-fold
- Cross-validation random state: 42

Cross-validation is performed on the training portion of the data. The
holdout test set is reserved for model-development checking and subsequent
evaluation.

## Candidate Models

The following models were developed:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Elastic Net
5. Random Forest Regressor
6. Gradient Boosting Regressor
7. Tuned Gradient Boosting Regressor

Linear models use standardization where appropriate. Tree-based models do
not require feature scaling.

## Development Metrics

Models are compared using:

- Cross-validated R²
- Cross-validated RMSE on logPrice
- Cross-validated MAE on logPrice
- Cross-validated training R²
- Generalization gap
- Holdout-test R²
- Holdout-test RMSE
- Holdout-test MAE

## Development Candidate

The development candidate selected for subsequent model evaluation is:

**Tuned Gradient Boosting**

Its cross-validated development results were:

- Mean CV R²: 0.8472
- CV R² standard deviation: 0.0816
- Mean CV RMSE: 0.2434
- Mean CV MAE: 0.1725
- Mean training R²: 0.9295
- Generalization gap: 0.0823

Selection in Part 9 is a development decision. Final model evaluation,
residual diagnostics, error analysis, calibration, and business interpretation
are addressed in Part 10.

## Reproducibility

The random seed is fixed at `42`. The complete model-development
process is implemented in `python/09_machine_learning.py`.

The script does not modify the analytical dataset.
