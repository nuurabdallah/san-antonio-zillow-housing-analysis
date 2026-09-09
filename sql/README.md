# SQL Analysis

This directory contains the PostgreSQL scripts used to analyze the San Antonio Zillow housing data.

The SQL workflow transforms the finalized analytical dataset into a structured PostgreSQL analytics layer and uses SQL to independently evaluate the project's major business questions.

---

## SQL Workflow

```text
01 Create Database and Schemas
        ↓
02 Create Staging Table
        ↓
03 Create Analytical Table
        ↓
04 Data Quality Checks
        ↓
05 Market Analysis
        ↓
06 ZIP Analysis
        ↓
07 Zestimate Analysis
        ↓
08 Advanced Rankings
        ↓
09 Create Analytical Views
        ↓
10 Final Validation
```

---

## Scripts

### 01 — Create Database and Schemas

Establishes the PostgreSQL database structure used for the project.

The database layer separates the raw staging data from the analytical data used for reporting and visualization.

Key components include:

- Database creation
- Staging schema
- Analytics schema
- Separation of raw and analytical data

---

### 02 — Create Staging Table

Creates the staging table used to receive the finalized analytical dataset before transformation into the analytics layer.

The staging layer preserves the source structure while allowing PostgreSQL to perform controlled data-type conversions and transformations.

Key responsibilities include:

- Defining the staging table structure
- Loading source fields
- Converting source values into appropriate PostgreSQL data types
- Preparing records for the analytical table

---

### 03 — Create Analytical Table

Creates the primary analytical table:

```text
analytics.zillow_listings
```

The analytical table provides a structured SQL-ready representation of the housing data.

It includes standardized analytical fields such as:

- Property identifiers
- Location information
- Listing price
- Bedrooms and bathrooms
- Living area
- Property type
- Days on Zillow
- Zestimate
- Tax assessed value
- Lot area
- Price per square foot
- Price change indicators
- Log-transformed price

The analytical table uses `zpid` as the primary property identifier.

---

### 04 — Data Quality Checks

Performs SQL-based validation of the analytical dataset before analysis.

The checks evaluate:

- Total record count
- Unique property identifiers
- Duplicate records
- Missing prices
- Missing ZIP codes
- Missing living area
- Data completeness
- Key analytical fields
- Zestimate coverage

The final analytical population contains:

- 810 properties
- 810 unique `zpid` values
- 0 duplicate `zpid` values
- 0 missing listing prices
- 0 missing ZIP codes
- 0 missing living-area values

---

### 05 — Market Analysis

Performs market-level housing analysis using PostgreSQL.

The analysis evaluates:

- Listing price distributions
- Average and median prices
- Price per square foot
- Property characteristics
- Price segments
- Property-type differences
- Market-level KPIs

Price segmentation is used to examine how listings are distributed across different price ranges.

---

### 06 — ZIP Analysis

Analyzes housing-market differences across San Antonio ZIP codes.

The analysis includes:

- Listing counts
- Median listing price
- Median price per square foot
- Median living area
- Days on Zillow
- ZIP-level market comparisons

ZIP codes are classified using the project's minimum sample-size rule.

ZIP codes with fewer than 10 listings are classified as:

```text
Sparse Sample
```

ZIP codes with at least 10 listings are classified as:

```text
Sufficient Sample
```

Sparse ZIP codes remain in the analytical dataset but are excluded from standalone ZIP rankings when the minimum sample-size requirement is not met.

---

### 07 — Zestimate Analysis

Evaluates the relationship between Zillow listing prices and Zestimate values.

The analysis includes:

- Listing price versus Zestimate
- Zestimate gaps
- Zestimate gap percentages
- Zestimate coverage
- Potential opportunities
- Properties priced above Zestimate
- Properties priced below Zestimate

A positive Zestimate gap indicates that the Zestimate is greater than the listing price.

Potential opportunities are treated as analytical signals rather than guaranteed investment returns.

Missing Zestimate values remain missing rather than being treated as zero.

---

### 08 — Advanced Rankings

Performs additional ranking and segmentation analyses using SQL.

The analysis includes:

- ZIP-code market rankings
- Median price rankings
- Median price-per-square-foot rankings
- Median living-area rankings
- Days-on-Zillow rankings
- ZIP/property-type composition
- Property-level Zestimate opportunity rankings

Window functions and grouped analytical queries are used to rank and compare properties and ZIP codes.

---

### 09 — Create Analytical Views

Creates reusable PostgreSQL analytical views for downstream analysis and Tableau visualization.

The view layer provides structured outputs for:

- Market overview
- Market KPIs
- Price segments
- Property types
- ZIP analysis
- ZIP/property-type composition
- Zestimate KPIs
- Zestimate opportunities
- Opportunity KPIs
- Tableau-ready property-level analysis

The Tableau-ready analytical view provides a consolidated property-level dataset for visualization.

The database architecture is:

```text
Raw Zillow Data
        ↓
staging.zillow_analysis_raw
        ↓
analytics.zillow_listings
        ↓
Analytical Views
        ↓
Tableau
```

Tableau connects directly to the PostgreSQL analytical layer rather than relying on a CSV workaround.

---

### 10 — Final Validation

Performs the final validation of the PostgreSQL analytical layer.

Validation confirms that:

- The analytical population remains intact
- Property identifiers remain unique
- Major analytical views reconcile to the expected population
- The Tableau-ready view contains the expected records
- Zestimate coverage is consistent
- Opportunity calculations reconcile
- ZIP sample-size classifications are consistent
- Final ZIP-level results can be reproduced from the analytical layer

The final validation confirms the expected 810-property analytical population and the previously established Zestimate opportunity results.

---

## Key SQL Methodological Principles

The SQL analysis follows several important principles:

- The finalized analytical dataset is the authoritative source for the SQL analysis.
- The staging and analytics layers are separated to improve transparency and reproducibility.
- `zpid` is used as the primary property identifier.
- Missing Zestimate values are retained as missing rather than treated as zero.
- Sparse ZIP codes are retained but excluded from standalone ZIP rankings when the minimum sample-size requirement is not met.
- Median statistics are used where appropriate to reduce the influence of extreme housing values.
- Analytical views are created so downstream tools can use consistent SQL-defined calculations.
- SQL analysis is used to independently answer major business questions rather than simply storing data for Tableau.

---

## Reproducibility

The SQL workflow is designed to be executed sequentially.

The general process is:

1. Create the PostgreSQL database and schemas.
2. Create the staging table.
3. Load the finalized analytical dataset into the staging layer.
4. Create the analytical table.
5. Run SQL data-quality checks.
6. Run market-level analysis.
7. Run ZIP-code analysis.
8. Run Zestimate analysis.
9. Run advanced rankings.
10. Create the analytical views.
11. Run final validation.
12. Connect Tableau to the PostgreSQL analytical layer.

The SQL layer is designed to complement the Python workflow rather than replace it.

Python performs the data collection, cleaning, feature engineering, exploratory analysis, statistical analysis, visualization, and machine-learning workflow.

PostgreSQL provides the structured database and analytical SQL layer used for independent validation, business analysis, reusable views, and Tableau connectivity.

---

## Analytical Population

The finalized analytical dataset contains:

```text
810 properties
810 unique zpid values
0 duplicate zpid values
```

The SQL workflow preserves this analytical population while applying appropriate filtering only where the analytical question requires it, such as excluding sparsely sampled ZIP codes from standalone ZIP rankings.

---

## PostgreSQL and Tableau

PostgreSQL serves as the project's analytical database layer.

The SQL views provide Tableau with reusable, validated analytical outputs rather than requiring Tableau to independently recreate core calculations.

This separation allows the project to maintain a clear distinction between:

```text
Data Preparation
        ↓
PostgreSQL Analytics
        ↓
Analytical Views
        ↓
Tableau Visualization
```

This architecture improves consistency, transparency, and reproducibility across the project.


