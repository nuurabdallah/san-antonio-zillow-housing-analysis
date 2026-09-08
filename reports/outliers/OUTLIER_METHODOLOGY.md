# Outlier Analysis Methodology

## Objective

Part 4 investigates unusually high or low observations
within the San Antonio Zillow analysis-ready dataset.

The purpose is to identify observations that may require
investigation and to understand the distribution of the
housing market.

## Variables

The following variables are examined:

- price
- area
- pricePerSqFt
- beds
- baths
- lotAreaSqFt
- taxAssessedValue
- daysOnZillow
- logPrice

## Market-Wide IQR Analysis

The primary statistical screening method is the
1.5 × IQR rule.

Lower boundary:

Q1 - 1.5 × IQR

Upper boundary:

Q3 + 1.5 × IQR

Observations outside these boundaries are flagged as
market-wide statistical outliers.

## Z-Score Analysis

A secondary screening method identifies observations with
an absolute z-score of at least 3.

This provides another perspective on extreme observations.

Because housing variables such as price, lot area, and
tax-assessed value can be highly skewed, z-score results
are treated as screening evidence rather than proof of
invalid data.

## Property-Type-Specific IQR Analysis

Outlier boundaries are also calculated separately within
each property type.

A minimum of 10 valid observations is required
before a property-type/variable combination receives an
IQR-based outlier classification.

Groups below this threshold are labeled:

"Insufficient Sample"

This prevents very small groups from producing unreliable
statistical classifications.

## Why Property Type Matters

A property that is extreme relative to the entire market
may be normal within its property category.

For example, luxury single-family properties can naturally
have substantially higher prices and living areas than the
broader market.

Property-type-specific analysis therefore provides additional
context for interpreting market-wide outliers.

## Legitimate Market Outliers

Extreme properties are not automatically removed.

Real estate markets naturally contain legitimate observations
at both ends of the distribution.

An expensive property, large property, or unusually high
price-per-square-foot property may represent real market
behavior.

## Data-Quality Rules

Part 2 established explicit rules for clearly invalid values.

For example:

- Listing prices <= $1 were treated as invalid or
  placeholder values.
- Living area must be greater than zero.
- Bedroom and bathroom counts cannot be negative.

These rules are separate from statistical outlier detection.

## Modeling

Outlier analysis does not determine the final machine-learning
training set by itself.

Potential model influence is investigated later through
residual diagnostics, influence measures, cross-validation,
and model error analysis.

## Dataset Integrity

This script does not modify the analysis-ready dataset.

All outlier flags and reports are stored separately.

This preserves reproducibility and allows legitimate extreme
properties to remain available for downstream analysis.