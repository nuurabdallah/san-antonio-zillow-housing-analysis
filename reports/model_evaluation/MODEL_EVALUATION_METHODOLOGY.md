# Part 10 — Model Evaluation & Diagnostics

Part 10 formally evaluates the Tuned Gradient Boosting development candidate from Part 9.

**Source:** `data/processed/San_Antonio_Zillow_Feature_Engineered.csv`

**Population:** 810 analytical properties; 731 complete ML cases; 584 training records; 147 holdout records.

**Target:** `logPrice`

**Predictors:** `area`, `beds`, `baths`, `lotAreaSqFt`, `taxAssessedValue`, `daysOnZillow`

Target-derived/leakage-prone fields were excluded: `price`, `pricePerSqFt`, `zestimate`, `zestimate_gap`, `zestimate_gap_pct`, `deal_score`, and `opportunity_status`.

The holdout split reproduces Part 9 using random state 42. Evaluation includes log-price and dollar-scale metrics, calibration, residual diagnostics, price-segment error analysis, largest prediction errors, feature importance, and permutation importance.

Price segments use training-population quartiles to avoid deriving evaluation categories from the holdout set.

Residual normality is reported descriptively; normal residuals are not required for a tree-based predictive model. Statistical diagnostics do not establish causation.

The feature-engineered dataset was not modified.
