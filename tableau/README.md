# Tableau Dashboards

This directory contains the Tableau dashboards created for the San Antonio Zillow Housing Analysis project.

The dashboards connect directly to the PostgreSQL analytical layer and are designed to communicate the project's major market, Zestimate, and potential-opportunity findings.

---

## Tableau Data Architecture

The Tableau workflow uses PostgreSQL as the analytical data source.

```text
San Antonio Zillow Data
        ↓
PostgreSQL Staging Layer
        ↓
PostgreSQL Analytical Layer
        ↓
Analytical Views
        ↓
Tableau
```

The primary Tableau-ready property-level view is:

```text
analytics.v_tableau_listings
```

This approach keeps core analytical calculations in PostgreSQL while Tableau is used primarily for interactive visualization and business communication.

---

## Dashboard 1 — San Antonio Housing Market

Provides a high-level overview of the San Antonio housing market.

### Key Performance Indicators

- Total Listings
- Median Listing Price
- Average Listing Price
- Median Price per Square Foot
- Median Days on Zillow
- Zestimate Coverage

### Visualizations

- Median Listing Price by ZIP Code
- Median Price per Square Foot by ZIP Code
- Listing Price Distribution
- Average Price by Bedrooms
- Price versus Living Area

The dashboard provides a broad view of market pricing, property characteristics, and geographic differences.

---

## Dashboard 2 — Zestimate Analysis

Examines the relationship between Zillow listing prices and Zestimate values.

### Analysis Includes

- Listing Price versus Zestimate
- Potential Zestimate Gaps
- Potential Opportunities by ZIP Code
- Total Potential Zestimate Gap by ZIP Code
- Top Potential Zestimate Gap by Property
- Opportunity Status

### Interactive Filters

- ZIP Code
- Listing Price
- Bedrooms
- Bathrooms

Potential opportunities are treated as analytical signals rather than guaranteed investment returns.

---

## Dashboard 3 — Investment Opportunities

Focuses on properties identified as potential Zestimate-based opportunities.

### Key Metrics

- Potential Opportunities
- Average Potential Gap
- Median Potential Gap
- Average Potential Gap Percentage
- Deal Score

### Analysis Includes

- Deal Score Ranking
- Zestimate Gap Percentage versus Listing Price
- Price per Square Foot Analysis
- Property-Level Opportunity Analysis

The dashboard identifies properties where Zestimate exceeds listing price and ranks opportunities using the project's relative Deal Score methodology.

Deal Score is a screening and ranking metric, not a guarantee of investment performance.

---

## Tableau Methodology

The Tableau dashboards use the validated PostgreSQL analytical layer rather than independently recreating the project's core calculations.

Important methodological decisions include:

- Missing Zestimate values are retained as missing rather than treated as zero.
- Sparse ZIP codes are retained in the analytical data but are excluded from standalone ZIP rankings when the minimum sample-size requirement is not met.
- ZIP-level comparisons use the project's minimum sample-size threshold of 10 listings.
- Potential Zestimate opportunities are identified when Zestimate exceeds listing price.
- Deal Score is based on the relative Zestimate gap percentage.
- Tableau is used for visualization and interactive exploration rather than predictive modeling.

---

## Dashboard Purpose

Together, the three dashboards provide three levels of business communication:

```text
Dashboard 1
Market Overview
        ↓
Dashboard 2
Zestimate Analysis
        ↓
Dashboard 3
Potential Opportunities
```

This progression allows users to move from understanding the overall San Antonio housing market to examining Zestimate relationships and finally exploring individual potential opportunities.

---

## Connection to the Analytical Workflow

The Tableau dashboards represent the final visualization layer of the project.

```text
Data Collection
        ↓
Data Cleaning
        ↓
Data Quality Analysis
        ↓
Feature Engineering
        ↓
Exploratory Analysis
        ↓
Statistical Analysis
        ↓
Machine Learning
        ↓
Model Evaluation
        ↓
PostgreSQL / SQL Analysis
        ↓
Tableau Dashboards
```

The dashboards complement the Python, SQL, and machine-learning components of the project and provide an interactive business-facing presentation of the analysis.
