# Outlier Analysis

This folder contains the outlier investigation and statistical screening performed on the San Antonio Zillow housing dataset.

The purpose of this analysis was to identify unusually high or low observations, determine whether they represented potential data-quality issues or legitimate market variation, and document the results before modeling.

## Analysis Approach

The outlier workflow evaluated both market-wide and property-type-specific distributions.

### Statistical Screening

The analysis used:

- Interquartile Range (IQR) analysis
- Z-score screening
- Descriptive statistics
- Record-level outlier flags
- Property-type-specific IQR analysis
- Comparison of market-level and property-type-level outliers
- Intersection analysis across multiple outlier criteria

For IQR analysis, observations outside the following range were flagged:

```text
Lower Bound = Q1 − 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Extreme standardized observations were also screened using:

```text
|z-score| ≥ 3
```

## Variables Investigated

Outlier analysis was performed across key housing and modeling variables, including:

- Listing price
- Living area
- Bedrooms
- Bathrooms
- Days on Zillow
- Lot area
- Price per square foot
- Log-transformed listing price
- Tax assessed value

## Market-Level Analysis

The folder contains separate high- and low-value outputs for several variables, including:

- `price_highest.csv`
- `price_lowest.csv`
- `area_highest.csv`
- `area_lowest.csv`
- `beds_highest.csv`
- `beds_lowest.csv`
- `baths_highest.csv`
- `baths_lowest.csv`
- `daysOnZillow_highest.csv`
- `daysOnZillow_lowest.csv`
- `lotAreaSqFt_highest.csv`
- `lotAreaSqFt_lowest.csv`
- `pricePerSqFt_highest.csv`
- `pricePerSqFt_lowest.csv`
- `logPrice_highest.csv`
- `logPrice_lowest.csv`
- `taxAssessedValue_highest.csv`
- `taxAssessedValue_lowest.csv`

These files provide record-level views of the most extreme observations for each variable.

## IQR Analysis

The following outputs summarize IQR-based screening:

- `iqr_outlier_summary.csv`
- `property_type_iqr_outlier_summary.csv`
- `record_level_iqr_flags.csv`

Property-type-specific IQR analysis was used because distributions can differ substantially across housing types. An observation that is extreme relative to the entire market may be normal within its property-type group.

## Z-Score Analysis

The file:

- `zscore_outlier_summary.csv`

contains the summary of standardized extreme-value screening.

Z-score screening was treated as an additional diagnostic rather than an automatic rule for removing observations.

## Property-Type Analysis

The following files document differences between market-level and property-type-level outlier classifications:

- `property_type_outlier_summary.csv`
- `property_type_iqr_outlier_summary.csv`
- `market_vs_property_type_outliers.csv`

This comparison helps distinguish observations that are unusual across the overall market from observations that are unusual only within a particular property-type distribution.

## Record-Level Classification

The analysis also produced:

- `outlier_classification_summary.csv`
- `record_level_outlier_flags.csv`
- `outlier_intersection_summary.csv`

These outputs allow individual properties to be evaluated across multiple outlier criteria rather than relying on a single statistical test.

## Treatment of Outliers

Outliers were **investigated rather than automatically removed**.

San Antonio's housing market contains substantial variation in property price, size, location, and property type. As a result, extreme observations can represent legitimate market conditions rather than data errors.

The final analytical dataset retained valid observations after the outlier investigation.

This approach preserves the underlying market distribution and prevents legitimate high-value or low-value properties from being excluded solely because they are statistically unusual.

## Relationship to the Modeling Workflow

Outlier analysis was completed before the machine-learning stage.

Rather than removing legitimate extreme properties, model evaluation was used to determine whether the final model performed differently across the housing-price distribution.

This became particularly important during model evaluation, where systematic differences were observed at the lower and upper ends of the market.

## Key Takeaway

The outlier analysis provides a structured screening framework for identifying unusual housing observations while preserving legitimate market variation.

**An observation being classified as an outlier does not automatically mean that it is incorrect or should be removed.**

The results are therefore used as a diagnostic and documentation layer within the broader data-quality and modeling workflow.


