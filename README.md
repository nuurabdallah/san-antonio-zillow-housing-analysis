# San Antonio Zillow Housing Analysis

## End-to-End Data Analytics & Machine Learning Project

An end-to-end analysis of San Antonio residential housing listings using Python, PostgreSQL/SQL, Tableau, and machine learning.

The project analyzes housing-market patterns, geographic pricing differences, Zestimate relationships, property-level pricing opportunities, and the factors associated with listing prices.

### Portfolio Focus

`Data Analytics` • `Data Science` • `Python` • `SQL` • `PostgreSQL` • `Tableau` • `Machine Learning` • `Real Estate Analytics`

---

# Project Overview

This project was developed as a complete data analytics and data science workflow rather than as a single machine-learning exercise.

The workflow moves from data collection and validation through data cleaning, feature engineering, exploratory analysis, PostgreSQL/SQL analysis, visualization, machine learning, model interpretation, error analysis, calibration, and business recommendations.

The final analytical dataset contains:

- 810 residential properties
- 810 unique Zillow property IDs
- 0 duplicate properties
- 71.11% Zestimate coverage
- 731 complete machine-learning observations

The project combines technical analysis with business-facing interpretation to answer six primary questions about the San Antonio housing market.

---

# Business Questions

The project was designed to answer:

1. What factors are most strongly associated with San Antonio listing prices?

2. Which ZIP codes have the highest and lowest housing prices?

3. How closely do Zillow Zestimates align with observed listing prices?

4. Can listing prices be predicted without relying on Zestimate?

5. Can properties where listing price differs substantially from Zestimate be identified?

6. Where does the prediction model perform poorly?

---

# Analytical Workflow

```text
Zillow Listing Data
        │
        ▼
Data Collection
        │
        ▼
Data Cleaning & Quality Validation
        │
        ▼
Feature Engineering
        │
        ▼
Exploratory Data Analysis
        │
        ├───────────────┐
        ▼               ▼
   Python EDA      PostgreSQL / SQL
                        │
                        ▼
                Geographic Analysis
                        │
                        ▼
                  Tableau Layer
                        │
                        ▼
                 Tableau Dashboards
                        
Python Modeling
        │
        ▼
Model Comparison
        │
        ▼
Gradient Boosting Selection
        │
        ▼
Model Evaluation
        │
        ▼
SHAP / ICE Interpretation
        │
        ▼
Residual & Error Analysis
        │
        ▼
Calibration
        │
        ▼
Business Insights

