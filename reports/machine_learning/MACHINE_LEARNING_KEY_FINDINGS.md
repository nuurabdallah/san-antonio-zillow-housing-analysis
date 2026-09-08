# Part 9 — Machine Learning Key Findings

## Modeling Population

- Full analytical population: 810 properties.
- Complete machine-learning cases: 731.
- ML complete-case coverage: 90.25%.
- Target: `logPrice`.
- Predictors: `area`, `beds`, `baths`, `lotAreaSqFt`,
  `taxAssessedValue`, and `daysOnZillow`.

## Model Development

Seven candidate models were developed and compared using shuffled
10-fold cross-validation on the training data.

### Best Cross-Validated R²

**Tuned Gradient Boosting**

- Mean CV R²: 0.8472
- CV R² standard deviation: 0.0816

### Lowest Cross-Validated RMSE

**Tuned Gradient Boosting**

- Mean CV RMSE: 0.2434

### Smallest Generalization Gap

**Tuned Gradient Boosting**

- Generalization gap: 0.0823

## Development Candidate

The model carried forward to Part 10 is:

**Tuned Gradient Boosting**

Development metrics:

- Mean CV R²: 0.8472
- CV R² standard deviation: 0.0816
- Mean CV RMSE: 0.2434
- Mean CV MAE: 0.1725
- Mean training R²: 0.9295
- Generalization gap: 0.0823
- Holdout-test R²: 0.7141
- Holdout-test RMSE: 0.2926
- Holdout-test MAE: 0.1853

## Interpretation

The development results show how well the candidate models capture
variation in log-transformed listing price using the approved structural
and property-level predictors.

The model selected for Part 10 is a development candidate rather than a
final business-decision model. Final evaluation should consider residual
behavior, error magnitude, generalization, calibration, feature importance,
and other diagnostics.

## Target Leakage

No approved predictor is directly derived from listing price or another
target-derived business metric.

Variables such as `pricePerSqFt`, `zestimate_gap`, `zestimate_gap_pct`,
and `deal_score` were intentionally excluded from the predictor set.
