# Data Quality Report

This folder contains the data quality checks and validation outputs for the San Antonio Zillow housing analysis project.

The purpose of the data quality workflow was to verify that the analytical dataset was complete, structurally consistent, and suitable for exploratory analysis, statistical analysis, SQL analysis, Tableau, and machine learning.

## Dataset Overview

The final analytical dataset contains:

- **810 housing listings**
- **810 unique properties (`zpid`)**
- **0 duplicate property IDs**
- **0 missing listing prices**
- **0 missing ZIP codes**
- **0 missing living-area values**
- **71.11% Zestimate coverage**
- **90.25% machine-learning feature completeness**

## Data Quality Checks

The quality-control workflow evaluated:

- Dataset structure and data types
- Missing values
- Duplicate property records
- Numeric field validity
- Coordinate coverage
- Tax-assessed-value coverage
- Zestimate coverage
- Missingness by property type
- Machine-learning completeness
- Key analytical field coverage

## Missing Data

Missing values were retained when they represented legitimate unavailable information rather than invalid records.

Important examples include:

- Zestimate values are missing for some properties because a Zestimate was not available.
- Missing Zestimate values were **not** treated as zero.
- Four properties have missing geographic coordinates.
- Some properties have missing tax-assessed values.

These records were retained where appropriate and excluded only from analyses that specifically require the missing field.

## Machine Learning Completeness

The machine-learning workflow uses a leakage-safe set of predictors:

- `area`
- `beds`
- `baths`
- `lotAreaSqFt`
- `taxAssessedValue`
- `daysOnZillow`

The target variable is:

- `logPrice`

Properties missing required modeling fields were excluded from the modeling dataset while remaining part of the broader analytical dataset.

## Zestimate Coverage

Zestimate availability was evaluated separately from general data completeness.

The final dataset contains:

- **576 listings with Zestimate**
- **234 listings without Zestimate**
- **71.11% Zestimate coverage**

Zestimate-based analyses therefore use only properties with available Zestimate values.

## Files

| File | Description |
|---|---|
| `DATA_QUALITY_METHODOLOGY.md` | Detailed methodology for the data-quality assessment |
| `coordinate_coverage.csv` | Geographic coordinate completeness |
| `data_type_summary.csv` | Dataset field and data-type summary |
| `dataset_profile.csv` | Overall dataset profile |
| `key_field_coverage.csv` | Coverage of important analytical fields |
| `missingness_by_property_type.csv` | Missing-value patterns by property type |
| `missingness_summary.csv` | Overall missing-value summary |
| `ml_complete_case_summary.csv` | Machine-learning complete-case summary |
| `ml_missingness_summary.csv` | Missingness among machine-learning variables |
| `numeric_validation.csv` | Numeric-field validation results |
| `quality_summary.csv` | Overall data-quality summary |
| `tax_assessed_value_coverage.csv` | Tax-assessed-value availability |
| `zestimate_coverage.csv` | Zestimate availability and coverage |

## Role in the Project

The data-quality assessment was completed before downstream analysis to establish confidence in the analytical dataset.

The validated dataset was subsequently used for:

1. Exploratory data analysis
2. Feature engineering
3. Statistical analysis
4. PostgreSQL/SQL analysis
5. Tableau dashboards
6. Machine-learning modeling
7. Model evaluation and diagnostics

## Key Principle

Data-quality issues were **investigated and documented rather than automatically removed**. Records were only excluded from an analysis when the missing or invalid field prevented that specific analysis from being performed reliably.


