# Raw Data

This directory contains the raw Zillow housing data collected during the data acquisition stage of the project.

## Source Data

The raw dataset is preserved here as the starting point for the project’s data pipeline.

The data collection workflow is documented in:

`python/01_data_collection.py`

## Data Pipeline

```text
Zillow Source Data
        ↓
Raw Data
        ↓
Data Cleaning
        ↓
Analysis-Ready Dataset
        ↓
Feature Engineering
        ↓
Final Analytical Dataset
```

## Reproducibility

The data collection script documents the fields captured, validation checks, duplicate handling, and output process.

Because Zillow listings and website data can change over time, rerunning the collection process may produce a different dataset from the version used in this analysis.

The processed datasets used for the completed analysis are available in:

`data/processed/`


