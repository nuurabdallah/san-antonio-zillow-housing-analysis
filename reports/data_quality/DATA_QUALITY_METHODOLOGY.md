# Data Quality Methodology

## Dataset

The analysis-ready San Antonio Zillow dataset is the
authoritative dataset used for downstream SQL analysis,
Tableau dashboards, exploratory analysis, and machine-learning
modeling.

## Missing Values

Missing values are retained when they represent unavailable
source information rather than an obvious data error.

### Zestimate

Missing Zestimate values are retained as missing.

They are not replaced with listing price, tax-assessed value,
or another estimated value.

This preserves the distinction between an unavailable Zillow
valuation and an observed valuation.

### Tax-Assessed Value

Missing tax-assessed values are retained as missing.

Tax-assessed value is used only for analyses and models where
the field is available.

### Coordinates

Missing latitude or longitude values are retained as missing.

Properties are not removed solely because geographic
coordinates are unavailable.

## Machine Learning

Machine-learning models use complete cases for the required
predictors and target.

The primary predictors are:

- area
- beds
- baths
- lotAreaSqFt
- taxAssessedValue
- daysOnZillow

The target variable is:

- logPrice

Rows missing one or more required modeling variables are
excluded from the modeling subset rather than having values
artificially imputed.

## Outliers

Legitimate high-value properties are retained.

Extreme observations are investigated separately during
outlier and residual analysis rather than automatically
removed from the analytical dataset.

## Duplicate Properties

zpid is treated as the primary property identifier.

Duplicate property identifiers are monitored during
validation.

## Derived Variables

pricePerSqFt and logPrice are recalculated from the validated
listing price and living-area fields.

## Reproducibility

This script audits the dataset without modifying the
analysis-ready source file.

The generated reports document completeness, missingness,
numeric validity, and modeling readiness.