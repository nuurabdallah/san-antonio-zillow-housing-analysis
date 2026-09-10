# Reports

This folder contains the analysis reports, validation outputs, statistical results, visualizations, machine-learning results, and model evaluation artifacts generated throughout the San Antonio Zillow housing analysis project.

## Report Organization

The reports are organized according to the major stages of the analytical workflow:

```text
reports/
├── data_quality/
├── outliers/
├── exploratory_data_analysis/
├── feature_engineering/
├── statistical_analysis/
├── visualization/
├── machine_learning/
├── model_evaluation/
└── business_insights/
```

## Report Categories

### Data Quality

Contains data-quality assessments and validation outputs covering:

- Dataset structure
- Missing values
- Duplicate records
- Numeric validation
- Coordinate coverage
- Zestimate coverage
- Tax-assessed-value coverage
- Machine-learning completeness

### Outlier Analysis

Contains documentation and results from the investigation of unusual observations and potential outliers.

Outliers were investigated using statistical screening and contextual analysis rather than automatically removed.

### Exploratory Data Analysis

Contains charts and analytical outputs examining:

- Listing-price distributions
- Property characteristics
- Living area
- Bedrooms and bathrooms
- Property types
- ZIP-code differences
- Zestimate relationships
- Housing-market patterns

### Feature Engineering

Contains documentation and outputs for analytical variables created from the source data, including:

- Price per square foot
- Log-transformed listing price
- Zestimate gaps
- Zestimate gap percentage
- Deal Score
- Price segments
- ZIP-code sample-size classifications
- Price-change indicators

### Statistical Analysis

Contains statistical analysis examining relationships between housing characteristics and listing prices, including correlation analysis, group comparisons, hypothesis tests, regression analysis, and diagnostic results.

### Visualization

Contains presentation-ready Python visualizations created during the analytical workflow.

### Machine Learning

Contains machine-learning model results, including:

- Candidate model comparisons
- Cross-validation results
- Final model results
- Feature importance
- Prediction analysis
- Model interpretation

The final model is a tuned Gradient Boosting model predicting `logPrice` without using Zestimate-derived variables as predictors.

### Model Evaluation

Contains detailed evaluation of the final model, including:

- Prediction error
- Calibration
- Residual analysis
- Segment-level performance
- Prediction bias
- Error diagnostics
- Model interpretation

### Business Insights

Contains the final business-facing findings and recommendations derived from the completed analysis.

## Purpose

These reports provide an auditable record of the analytical workflow and support the conclusions presented in the project's GitHub documentation, Tableau dashboards, and notebooks.


