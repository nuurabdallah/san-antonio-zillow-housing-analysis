# Python Analysis Workflow

This directory contains the Python workflow used for the San Antonio Zillow Housing Analysis project.

The scripts are organized in the order in which the analytical workflow was performed, from data collection and preparation through statistical analysis, visualization, machine learning, and model evaluation.

 

## Workflow

```text
01 Data Collection
        ↓
02 Data Cleaning
        ↓
03 Data Quality Analysis
        ↓
04 Outlier Analysis
        ↓
05 Feature Engineering
        ↓
06 Exploratory Data Analysis
        ↓
07 Statistical Analysis
        ↓
08 Visualization
        ↓
09 Machine Learning
        ↓
10 Model Evaluation 
```

## Scripts

### 01 — Data Collection

Collects Zillow housing listing data and establishes the raw dataset used for the analysis.

Key responsibilities include:

- Collecting property-level listing information
- Extracting Zillow property identifiers (`zpid`)
- Validating collected records
- Removing duplicate properties
- Saving the raw collected data

 

### 02 — Data Cleaning

Cleans and prepares the collected Zillow data for analysis.

Key responsibilities include:

- Validating listing prices
- Handling invalid or implausible values
- Standardizing property characteristics
- Preserving legitimate missing values
- Validating duplicate records
- Creating initial analytical fields such as price per square foot and log-transformed price

 

### 03 — Data Quality Analysis

Performs a formal data-quality audit of the analytical dataset.

The audit evaluates:

- Dataset structure
- Missing values
- Key-field completeness
- Zestimate coverage
- Tax assessed value coverage
- Coordinate coverage
- Numeric validity
- Machine-learning completeness
- Duplicate records
- Data-type consistency

Missing values are evaluated and documented rather than automatically imputed.

 

### 04 — Outlier Analysis

Identifies and evaluates statistical outliers within the housing dataset.

Outlier analysis includes:

- Market-wide IQR analysis
- Z-score analysis
- Property-type-specific outlier analysis
- Extreme-value identification
- Multiple-outlier observations
- Comparison of statistical outliers with potential data-quality issues

Outliers are investigated and documented rather than automatically removed.

 

### 05 — Feature Engineering

Creates the analytical features used throughout the project.

Key engineered variables include:

- `pricePerSqFt`
- `logPrice`
- `zestimate_gap`
- `zestimate_gap_pct`
- `opportunity_status`
- `deal_score`
- `price_segment`
- `zip_listing_count`
- `zip_sample_status`
- `hasPriceChange`

Feature engineering also prepares the leakage-safe predictor set used for machine learning.

 

### 06 — Exploratory Data Analysis

Performs exploratory analysis of the San Antonio housing market.

The analysis examines:

- Overall listing prices
- Property characteristics
- Price segments
- Property types
- ZIP-code differences
- Price per square foot
- Living area
- Zestimate relationships
- Potential opportunities
- Correlations between variables

ZIP-code analyses distinguish sufficiently sampled ZIP codes from sparse ZIP codes.

 

### 07 — Statistical Analysis

Applies formal statistical methods to evaluate relationships and differences within the housing market.

Methods include:

- Pearson and Spearman correlation
- Welch's t-test
- Mann–Whitney U tests
- ANOVA
- Kruskal–Wallis tests
- Effect-size measures
- Confidence intervals
- OLS regression
- Variance Inflation Factor (VIF)
- Breusch–Pagan testing
- Normality diagnostics
- Durbin–Watson testing
- Cook's distance
- Nested regression analysis

The statistical models use the leakage-safe predictor framework established for the project.

 

### 08 — Visualization

Creates the publication-ready Python visualizations used to communicate the analysis.

Visualizations include:

- Listing price distributions
- Log-price distributions
- Price relationships with property characteristics
- Correlation analysis
- ZIP-code pricing comparisons
- Property-type comparisons
- Listing Price vs. Zestimate
- Zestimate gap analysis
- Potential opportunities by ZIP code
- Bedroom-group comparisons

 

### 09 — Machine Learning

Develops and compares machine-learning models for predicting listing prices.

The modeling workflow:

- Uses `logPrice` as the target
- Uses six leakage-safe predictors
- Uses an 80/20 train-test split
- Uses random state 42
- Uses 10-fold cross-validation
- Compares multiple regression and tree-based models
- Selects the final model based on cross-validation performance

The final selected model is a tuned Gradient Boosting model.

#### Machine-Learning Predictors

The final model uses:

- `area`
- `beds`
- `baths`
- `lotAreaSqFt`
- `taxAssessedValue`
- `daysOnZillow`

Zestimate-derived variables and other price-derived variables are excluded from the predictive model to prevent target leakage.

 

### 10 — Model Evaluation

Performs formal evaluation and diagnostic analysis of the final Gradient Boosting model.

Evaluation includes:

- Holdout R²
- RMSE
- MAE
- Median absolute error
- MAPE
- Calibration analysis
- Residual diagnostics
- Price-segment error analysis
- Feature importance
- Permutation importance
- Largest prediction errors

The evaluation focuses not only on overall predictive performance but also on where the model performs well and where systematic prediction errors occur.

 

## Reproducibility

The Python workflow is designed to be executed sequentially.

The general workflow is:

1. Collect the source data.
2. Clean and validate the dataset.
3. Perform the data-quality audit.
4. Analyze outliers.
5. Engineer analytical features.
6. Perform exploratory analysis.
7. Conduct statistical analysis.
8. Generate presentation visualizations.
9. Train and compare machine-learning models.
10. Evaluate the final model.

The analytical dataset contains **810 properties with 810 unique Zillow property identifiers (`zpid`)**.

Machine-learning analysis uses the complete cases available for the six approved predictors and the `logPrice` target. Missing analytical values are not artificially replaced simply to increase the modeling population.

 

## Key Methodological Principles

This project follows several important analytical principles:

- Missing Zestimate values are retained as missing rather than treated as zero.
- Legitimate outliers are investigated rather than automatically deleted.
- Sparse ZIP codes are retained in the analytical dataset but excluded from standalone ZIP rankings when the minimum sample-size requirement is not met.
- Price-derived variables are excluded from the machine-learning predictor set when they would introduce target leakage.
- Zestimate variables are excluded from the machine-learning model because the project evaluates whether listing prices can be predicted independently of Zestimate information.
- Statistical analysis and machine learning serve different purposes: statistical models help evaluate relationships and inference, while machine-learning models focus on predictive performance.


