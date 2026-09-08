
# Part 5 — Feature Engineering Methodology

## Purpose

This stage creates and validates the final engineered features used
throughout the San Antonio Zillow housing analysis.

The feature-engineering process preserves the complete
810-property analytical population.

## Source Dataset

Input:

`data/processed/San_Antonio_Zillow_Analysis.csv`

The source dataset contains:

- 810 properties
- 67 source columns

Output:

`data/processed/San_Antonio_Zillow_Feature_Engineered.csv`

## Population

- Total properties: 810
- Unique zpid: 810
- Duplicate zpid: 0

No observations were removed during feature engineering.

## ZIP Code Standardization

The source Zillow field:

`addressZipcode`

is standardized into:

`zipcode`

The original `addressZipcode` field is preserved.

ZIP codes with at least 10 listings are classified as:

`Sufficient Sample`

ZIP codes with fewer than 10 listings are classified as:

`Sparse Sample`

Sparse ZIPs remain in the dataset and are not removed.

## Engineered Features

### Price Per Square Foot

Calculated as:

`Listing Price / Living Area`

This is used for market analysis and property comparison.

It is **not used as an ML predictor** because it directly contains
the listing price target.

### Log Price

Calculated as:

`ln(Listing Price)`

This is the dependent variable used for the finalized price
prediction model.

### Zestimate Gap

Calculated as:

`Zestimate - Listing Price`

Positive values indicate that the Zestimate is above the asking price.

### Zestimate Gap Percentage

Calculated as:

`(Zestimate - Listing Price) / Listing Price`

This measures the relative difference between Zillow's Zestimate
and the listing price.

### Opportunity Status

Properties are classified as:

- Potential Opportunity — Zestimate exceeds listing price
- Above Zestimate — Zestimate does not exceed listing price
- Null / missing Zestimate — Zestimate unavailable

Properties without Zestimate data remain missing rather than being
imputed.

### Deal Score

Deal Score is a percentile-based relative ranking of
`zestimate_gap_pct`.

Higher scores indicate properties with larger positive Zestimate gaps
relative to other properties with available Zestimate data.

Deal Score is an analytical ranking metric, not a guaranteed measure
of investment return.

### Price Segments

Properties are divided into:

- Under $150K
- $150K-$300K
- $300K-$500K
- $500K-$1M
- $1M-$2M
- $2M+

## Machine-Learning Feature Separation

The finalized ML predictors are:

1. area
2. beds
3. baths
4. lotAreaSqFt
5. taxAssessedValue
6. daysOnZillow

Target:

`logPrice`

## Target Leakage Prevention

The following variables are intentionally excluded from the ML
predictor set because they contain or are directly derived from
listing price:

- pricePerSqFt
- zestimate_gap
- zestimate_gap_pct
- deal_score
- opportunity_status
- price_segment

This separation allows the project to use these variables for
business analysis without contaminating the independent price
prediction model.

## Coverage

- Zestimate records: 576
- Potential opportunities: 16
- Sufficient-sample ZIP codes: 31
- Sparse ZIP codes: 28

## Data Treatment

Feature engineering does not automatically remove statistical
outliers.

Legitimate high-value, low-value, unusually large, or otherwise
unusual properties remain part of the analytical dataset.

Missing Zestimate and tax-assessed values remain missing.

## Validation

The final feature-engineered dataset is required to maintain:

- 810 properties
- 810 unique zpids
- 0 duplicate zpids
- no missing ZIP codes
- valid positive listing prices
- valid price-per-square-foot calculations
- valid log-price calculations
- Zestimate-gap coverage matching Zestimate availability
- 16 potential Zestimate opportunities
- Deal Score coverage matching Zestimate coverage
- Deal Score values between 0 and 100
- complete price segmentation
- complete ZIP sample classification

All validation checks must pass before this dataset is used
downstream.
