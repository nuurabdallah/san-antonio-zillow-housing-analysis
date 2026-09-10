# Jupyter Notebooks

This folder contains the two polished Jupyter notebooks developed for the San Antonio Zillow Housing Analysis project.

The notebooks provide an interactive presentation of the project's exploratory analysis and machine learning workflow. They are designed to complement the production Python scripts, SQL analysis, reports, and Tableau dashboards rather than duplicate every analytical experiment.

---

## Notebook Workflow

The notebooks follow the project's analytical workflow:

```text
Data
  ↓
Exploratory Data Analysis
  ↓
Statistical Findings
  ↓
Machine Learning
  ↓
Model Evaluation
  ↓
Business Insights
```

Both notebooks use the project's authoritative feature-engineered analytical dataset.

---

## 01 — Exploratory Data Analysis

**File:** `01_eda.ipynb`

This notebook presents the exploratory analysis used to understand the San Antonio housing market before statistical modeling and machine learning.

### Contents

- Project objective and analytical context
- Dataset structure and descriptive statistics
- Data quality and missing-value review
- Listing price distributions
- Log-transformed listing price
- Property characteristics
- Living area, bedrooms, and bathrooms
- Correlation analysis
- ZIP-code market analysis
- Property-type analysis
- Zestimate coverage and comparison
- Zestimate gap analysis
- Potential opportunity analysis
- Key exploratory findings
- Conclusion

### Purpose

The notebook provides an interactive view of the major patterns identified during exploratory data analysis and establishes the foundation for the project's statistical and machine learning work.

---

## 02 — Housing Price Model

**File:** `02_housing_price_model.ipynb`

This notebook presents the project's machine learning workflow for predicting San Antonio listing prices without using Zillow Zestimate as a predictor.

### Contents

- Modeling objective
- Analytical dataset preparation
- Leakage-safe predictor selection
- Train/test methodology
- Candidate model comparison
- Gradient Boosting model selection
- Hyperparameter tuning
- Cross-validation
- Holdout evaluation
- Prediction performance
- Feature importance
- Model interpretation
- Error analysis
- Calibration
- Price-segment performance
- Business interpretation
- Conclusion

### Final Model

The selected model is a **Tuned Gradient Boosting Regressor**.

The model predicts the log-transformed listing price (`logPrice`) using:

- Living area (`area`)
- Bedrooms (`beds`)
- Bathrooms (`baths`)
- Lot area (`lotAreaSqFt`)
- Tax-assessed value (`taxAssessedValue`)
- Days on Zillow (`daysOnZillow`)

Zestimate-derived variables are intentionally excluded from the predictive model to avoid target leakage and allow the model to independently estimate listing prices.

---

## Relationship to the Python Workflow

The notebooks are the interactive presentation layer for analysis that was developed and validated through the project's Python workflow.

The production Python scripts are located in:

```text
python/
├── 01_data_collection.py
├── 02_data_cleaning.py
├── 03_data_quality_analysis.py
├── 04_outlier_analysis.py
├── 05_feature_engineering.py
├── 06_exploratory_data_analysis.py
├── 07_statistical_analysis.py
├── 08_visualization.py
├── 09_machine_learning.py
└── 10_model_evaluation.py
```

The notebooks provide a more readable, interactive narrative for portfolio presentation, while the Python scripts preserve the structured analytical workflow.

---

## Data Source

The notebooks use the project's authoritative feature-engineered dataset:

```text
data/processed/San_Antonio_Zillow_Feature_Engineered.csv
```

The dataset contains **810 housing listings/properties** used throughout the project's final analytical workflow.

---

## Reproducibility

The notebooks are designed to run from the `notebooks/` directory using the relative project path to the processed analytical dataset.

Required Python packages are documented in the repository's:

```text
requirements.txt
```

For the most complete and reproducible workflow, the notebooks should be considered alongside the numbered Python scripts, SQL pipeline, analytical reports, and Tableau workbook.

---

## Portfolio Role

These notebooks are intended to make the project easier to understand for recruiters, hiring managers, and other reviewers.

They provide an interactive narrative of:

1. **What was analyzed**
2. **How the data was explored**
3. **What statistical patterns were identified**
4. **How the prediction model was developed**
5. **How the model was evaluated**
6. **What the results mean from a business perspective**

The notebooks complement the project's production code and reporting artifacts without replacing them.


