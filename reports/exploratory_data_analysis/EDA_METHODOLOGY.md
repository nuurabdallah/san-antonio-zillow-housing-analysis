# Part 6 — Exploratory Data Analysis Methodology

## Purpose

Part 6 examines the structure, distributions, relationships, and
business characteristics of the finalized feature-engineered San
Antonio Zillow dataset.

## Source

The analysis uses:

`data/processed/San_Antonio_Zillow_Feature_Engineered.csv`

The analytical population contains 810 unique properties.

## Descriptive Analysis

Descriptive statistics include:

- Count
- Mean
- Standard deviation
- Minimum
- 1st percentile
- 5th percentile
- 25th percentile
- Median
- 75th percentile
- 95th percentile
- 99th percentile
- Maximum
- Missing-value counts

## Correlation Analysis

Both Pearson and Spearman correlations are calculated.

Pearson correlation measures linear association.

Spearman correlation measures monotonic association based on ranks.

Correlation does not establish causation.

## ZIP Analysis

ZIP-level comparisons use the established project rule:

- 10 or more listings = Sufficient Sample
- Fewer than 10 listings = Sparse Sample

Sparse ZIP codes remain in the dataset but should not be treated as
standalone rankings without appropriate caution.

## Property-Type Composition

Property-type composition is examined because ZIP-level price
differences can reflect differences in the types of properties
represented within each ZIP.

## Zestimate Analysis

Zestimate analysis is restricted to properties with non-missing
Zestimate values.

Zestimate gap is:

`Zestimate - Listing Price`

Zestimate gap percentage is:

`(Zestimate - Listing Price) / Listing Price`

## Visualization

Twenty exploratory charts are generated with Matplotlib and saved to:

`reports/eda/charts/`

## Machine Learning Separation

EDA includes price-derived business metrics for descriptive purposes,
but these variables are not treated as machine-learning predictors when
they directly contain or derive from the target listing price.

The finalized ML predictor set remains:

- area
- beds
- baths
- lotAreaSqFt
- taxAssessedValue
- daysOnZillow

The ML target remains:

- logPrice
