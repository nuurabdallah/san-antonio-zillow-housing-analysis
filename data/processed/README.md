# Processed Data

This directory contains the cleaned, analysis-ready, and feature-engineered datasets used throughout the San Antonio Zillow housing analysis project.

## Dataset Pipeline

The datasets represent successive stages of the project's data preparation workflow:

```text
San_Antonio_Zillow_Raw.csv
        ↓
San_Antonio_Zillow_Processed.csv
        ↓
San_Antonio_Zillow_Analysis.csv
        ↓
San_Antonio_Zillow_Feature_Engineered.csv
```

## Files

### `San_Antonio_Zillow_Processed.csv`

Cleaned version of the collected Zillow data. Initial data cleaning and standardization are performed before downstream analysis.

### `San_Antonio_Zillow_Analysis.csv`

Analysis-ready dataset used as the foundation for exploratory analysis and subsequent analytical workflows.

### `San_Antonio_Zillow_Feature_Engineered.csv`

Final analytical dataset containing engineered variables used throughout the project.

Key engineered features include:

- `lotAreaSqFt`
- `pricePerSqFt`
- `logPrice`
- `hasPriceChange`
- Zestimate gap measures
- Deal Score
- Opportunity Status
- Price Segment
- ZIP-code sample-size classifications

## Authoritative Dataset

`San_Antonio_Zillow_Feature_Engineered.csv` is the final analytical dataset used for the completed exploratory analysis, statistical analysis, SQL workflow, Tableau dashboards, and machine-learning workflow.

The final dataset contains 810 property records and 810 unique Zillow property IDs (`zpid`).

## Reproducibility

The transformation process is documented in the Python workflow located in:

`python/02_data_cleaning.py`

and

`python/05_feature_engineering.py`

The datasets are preserved at each major processing stage so the progression from cleaned data to the final analytical dataset can be reviewed and reproduced.



