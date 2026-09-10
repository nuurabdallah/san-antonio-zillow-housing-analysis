# Exploratory Data Analysis

This folder contains the exploratory data analysis (EDA) performed on the San Antonio Zillow housing dataset.

The objective of the EDA was to understand the structure of the housing market, examine price and property characteristics, identify relationships between variables, evaluate Zestimate behavior, and identify patterns that informed the later statistical analysis and machine-learning workflow.

## EDA Scope

The analysis includes:

- Descriptive statistics
- Data distributions
- Price and living-area analysis
- Bedroom and bathroom analysis
- Tax assessed value analysis
- Price-per-square-foot analysis
- Days-on-Zillow analysis
- Zestimate comparison analysis
- Property-type analysis
- Price-segment analysis
- ZIP-code analysis
- Correlation analysis
- Potential opportunity analysis

## Dataset

The EDA uses the feature-engineered analytical dataset containing:

- 810 housing listings
- 810 unique `zpid` values
- No duplicate properties
- Engineered pricing and market variables
- Zestimate availability indicators
- ZIP-level sample-size classifications
- Property-level opportunity metrics

Missing values were retained where appropriate rather than automatically imputed or treated as zero.

## Supporting Analysis Files

### Descriptive Statistics

- `descriptive_statistics.csv`
- `market_overview.csv`

These files summarize the overall structure and descriptive characteristics of the housing market.

### Correlation Analysis

- `correlation_matrix_pearson.csv`
- `correlation_matrix_spearman.csv`
- `correlation_pairs.csv`

Both Pearson and Spearman correlation methods were used to evaluate relationships between numerical variables.

Correlation results are descriptive and should not be interpreted as evidence of causation.

### Market Segmentation

- `price_segment_analysis.csv`
- `property_type_analysis.csv`

These outputs examine differences across listing-price segments and property types.

### ZIP-Code Analysis

- `zip_analysis.csv`
- `zip_property_type_composition.csv`

ZIP-level analysis was used to evaluate geographic differences in listing prices and market composition.

ZIP codes with fewer than 10 valid listings were treated as sparse samples and were not used for standalone ZIP ranking comparisons.

### Zestimate Analysis

- `zestimate_analysis.csv`
- `opportunity_analysis.csv`

These files examine the relationship between listing price and Zestimate and identify properties where the observed listing price differs substantially from the Zestimate.

The analysis uses the term **potential opportunities** rather than guaranteed investment opportunities.

## Visualizations

The `charts/` directory contains 20 EDA visualizations.

### Price and Distribution Analysis

- `01_listing_price_distribution.png`
- `02_log_price_distribution.png`
- `03_living_area_distribution.png`
- `09_days_on_zillow_distribution.png`
- `17_price_segment_distribution.png`

These charts examine the distributions of listing price, log-transformed price, living area, days on Zillow, and price segments.

### Property Characteristics

- `04_price_vs_living_area.png`
- `05_price_by_bedrooms.png`
- `06_price_by_bathrooms.png`
- `16_property_type_distribution.png`

These visualizations examine how listing prices vary with property characteristics and property type.

### Pricing Relationships

- `07_price_vs_tax_assessed_value.png`
- `08_price_vs_price_per_sqft.png`
- `12_log_price_vs_living_area.png`
- `13_log_price_vs_tax_assessed_value.png`
- `14_log_price_vs_price_per_sqft.png`

These charts investigate relationships between listing prices and important housing-value variables.

### Zestimate Relationships

- `10_zestimate_gap_distribution.png`
- `11_listing_price_vs_zestimate.png`
- `15_log_price_vs_zestimate.png`

These visualizations examine Zestimate coverage, Zestimate gaps, and the relationship between listing prices and Zestimates.

### Geographic Analysis

- `18_top_zip_median_price.png`
- `19_top_zip_median_price_per_sqft.png`

These charts highlight differences in median listing price and median price per square foot across sufficiently sampled ZIP codes.

### Correlation Visualization

- `20_pearson_correlation_heatmap.png`

This heatmap provides a visual overview of Pearson correlations among the primary numerical variables.

## Key EDA Findings

The EDA established several important patterns that were carried forward into later analysis:

1. San Antonio listing prices show substantial right-skew, making log-transformed price useful for statistical and machine-learning analysis.

2. Living area, tax assessed value, and price per square foot show important relationships with listing price.

3. Housing characteristics such as bedrooms, bathrooms, and property type contribute to meaningful differences in observed listing prices.

4. Housing prices vary substantially across ZIP codes, with geographic comparisons requiring minimum sample-size controls.

5. Zestimate values provide useful comparative information for listings where Zestimate data are available, but Zestimate availability is incomplete.

6. Potential Zestimate-based opportunities can be identified by comparing listing prices with available Zestimates, but these should be interpreted as screening results rather than guaranteed investment returns.

## Role in the Project Workflow

EDA served as the bridge between data preparation and formal statistical and predictive analysis.

The workflow progressed from:

```text
Data Cleaning
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

The EDA findings were used to guide:

- Variable selection
- Transformation of listing price
- ZIP-code analysis
- Property-type comparisons
- Statistical testing
- Machine-learning feature selection
- Zestimate opportunity analysis
- Final business interpretation

## Reproducibility

The EDA results were generated by:

```text
python/06_exploratory_data_analysis.py
```

The corresponding interactive notebook is:

```text
notebooks/01_eda.ipynb
```

The notebook presents the major EDA findings in an interactive, narrative format, while the files in this directory preserve the detailed analytical outputs and visualizations.

## Important Interpretation Note

EDA findings describe relationships and patterns in the observed dataset.

They should not automatically be interpreted as causal relationships, investment recommendations, or evidence that one housing characteristic directly causes a change in listing price.
