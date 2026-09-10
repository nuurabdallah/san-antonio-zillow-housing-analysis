# Feature Engineering

This folder documents the feature engineering stage of the San Antonio Zillow Housing Analysis.

Feature engineering transformed the cleaned analytical dataset into a modeling- and analysis-ready dataset by creating derived variables for pricing analysis, Zestimate comparison, market segmentation, ZIP-code analysis, and machine learning.

## Objective

The goal of feature engineering was to create meaningful analytical variables while preserving the integrity of the original listing data.

The resulting feature-engineered dataset contains **810 properties** and **810 unique Zillow property IDs (`zpid`)**.

No duplicate properties were introduced during feature engineering.

## Engineered Features

### Geographic Features

#### `zipcode`

Standardized ZIP-code information used for geographic and market-level analysis.

ZIP code is treated as a **categorical/geographic variable**, not as a continuous numerical measurement.

#### `zip_listing_count`

Number of valid listings represented within each ZIP code.

This variable provides context for ZIP-level comparisons and helps identify areas with limited sample sizes.

#### `zip_sample_status`

Classifies ZIP codes according to the project sample-size rule:

- **Sufficient Sample:** 10 or more valid listings
- **Sparse Sample:** Fewer than 10 valid listings

Sparse ZIP codes remain in the dataset but are excluded from standalone ZIP ranking comparisons.

---

## Pricing Features

### `pricePerSqFt`

Calculated as:

```text
Listing Price / Living Area
```

This provides a standardized measure of listing price relative to property size.

### `logPrice`

Natural-log transformation of listing price.

The transformation was created because listing prices are strongly right-skewed and the log scale provides a more suitable target for statistical and machine-learning analysis.

`logPrice` is the target variable used in the final machine-learning workflow.

### `price_segment`

Categorizes properties into six listing-price segments to support market segmentation and error analysis.

The segments provide a consistent framework for comparing properties across different price ranges.

---

## Zestimate Features

Zestimate-related features were created to evaluate how Zillow's estimated property values compare with observed listing prices.

### `zestimate_gap`

Measures the dollar difference between Zestimate and listing price:

```text
Zestimate - Listing Price
```

Interpretation:

- Positive value = Zestimate is above the listing price
- Negative value = Zestimate is below the listing price

### `zestimate_gap_pct`

Measures the Zestimate gap relative to listing price:

```text
(Zestimate - Listing Price) / Listing Price × 100
```

This allows properties with different listing prices to be compared on a relative basis.

### `opportunity_status`

Classifies properties according to the project's Zestimate-based opportunity methodology.

The classification is intended for **screening and analytical comparison**, not as a guarantee of investment return.

### `deal_score`

Ranks properties based on their relative positive Zestimate gap percentage.

A higher Deal Score indicates a larger relative difference between Zestimate and listing price among properties with available Zestimate information.

Zestimate-derived variables are used for opportunity analysis but are **not used as predictors in the machine-learning model**.

---

## Listing Activity Feature

### `hasPriceChange`

Indicator identifying whether a listing has a recorded price change.

This converts price-change information into a format suitable for descriptive and analytical comparisons.

---

## Machine-Learning Feature Selection

A major purpose of feature engineering was to establish a **leakage-safe modeling dataset**.

The final machine-learning target is:

```text
logPrice
```

The six predictors used in the final modeling workflow are:

```text
area
beds
baths
lotAreaSqFt
taxAssessedValue
daysOnZillow
```

These variables were selected because they represent property characteristics, assessed value, and listing-market conditions without directly incorporating the target price or Zestimate-derived opportunity calculations.

### Excluded from Machine Learning

The following variables were intentionally excluded as predictors:

- `zestimate`
- `pricePerSqFt`
- `zestimate_gap`
- `zestimate_gap_pct`
- `deal_score`
- `opportunity_status`

These variables either contain information derived directly from listing price, contain Zestimate information, or otherwise create potential target leakage.

Keeping these variables out of the predictive feature set ensures that the model estimates listing price without relying on Zestimate-based information.

---

## Missing-Value Treatment

Feature engineering did not automatically replace missing values with zero.

In particular:

- Missing Zestimate values remain missing.
- Missing tax assessed values remain missing.
- Missing coordinates remain missing.
- Coordinate-dependent analyses exclude records without valid coordinates when necessary.

Missingness was treated as an analytical characteristic of the dataset rather than automatically interpreted as a zero value.

---

## Feature Documentation Files

### `feature_dictionary.csv`

Provides definitions and descriptions of the engineered and analytical variables.

It serves as the reference dictionary for understanding how each feature is defined and used.

### `feature_summary.csv`

Provides a summary of the feature-engineered dataset and the resulting variables.

### `FEATURE_ENGINEERING_METHODOLOGY.md`

Contains the detailed methodology used to create and validate the engineered features.

---

## Validation

The feature-engineering workflow validated that:

- The final dataset contains 810 properties.
- Zillow property IDs remain unique.
- No duplicate properties were introduced.
- Required analytical variables are available where expected.
- Zestimate-derived fields preserve missing Zestimate values rather than converting them to zero.
- ZIP sample-size classifications follow the project's minimum threshold.
- Opportunity calculations are based only on properties with available Zestimate information.
- The machine-learning feature set remains leakage-safe.

The completed feature-engineered dataset contains **16 identified potential Zestimate-based opportunities** among listings with available Zestimate information.

---

## Role in the Project Workflow

Feature engineering connects the cleaned dataset to the project's analytical and predictive workflows:

```text
Data Collection
      ↓
Data Cleaning
      ↓
Data Quality Analysis
      ↓
Outlier Analysis
      ↓
Feature Engineering
      ↓
Exploratory Data Analysis
      ↓
Statistical Analysis
      ↓
Machine Learning
      ↓
Model Evaluation
      ↓
Business Insights
```

The feature-engineered dataset serves as the primary input for the subsequent EDA, statistical analysis, and machine-learning stages.

## Reproducibility

The feature-engineering process is implemented in:

```text
python/05_feature_engineering.py
```

The resulting feature-engineered dataset is stored in:

```text
data/processed/San_Antonio_Zillow_Feature_Engineered.csv
```

The methodology and supporting outputs in this folder document how the engineered variables were created and validated.


