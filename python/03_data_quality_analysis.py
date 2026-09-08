"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/San_Antonio_Zillow_Analysis.csv"
)

OUTPUT_DIR = Path(
    "reports/data_quality"
)


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset():
    """
    Load the analysis-ready Zillow dataset.
    """

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nAnalysis dataset was not found:\n"
            f"{INPUT_FILE}\n\n"
            "Make sure Part 2 has been completed and "
            "the analysis-ready dataset exists."
        )

    print("=" * 70)
    print("SAN ANTONIO ZILLOW DATA QUALITY ANALYSIS")
    print("=" * 70)

    print(
        f"\nInput file: {INPUT_FILE}"
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns loaded: {len(df.columns):,}"
    )

    return df


# ============================================================
# BASIC DATASET PROFILE
# ============================================================

def create_dataset_profile(df):
    """
    Create a basic structural profile of the dataset.
    """

    profile = pd.DataFrame({
        "Metric": [
            "Total rows",
            "Total columns",
            "Unique properties",
            "Duplicate zpid records",
            "Memory usage (MB)"
        ],
        "Value": [
            len(df),
            len(df.columns),
            df["zpid"].nunique()
            if "zpid" in df.columns
            else np.nan,
            df["zpid"].duplicated().sum()
            if "zpid" in df.columns
            else np.nan,
            round(
                df.memory_usage(
                    deep=True
                ).sum() / (1024 ** 2),
                2
            )
        ]
    })

    return profile


# ============================================================
# MISSING-VALUE ANALYSIS
# ============================================================

def create_missingness_summary(df):
    """
    Calculate missing-value counts and percentages
    for every column.
    """

    total_rows = len(df)

    records = []

    for column in df.columns:

        missing_count = df[column].isna().sum()

        missing_pct = (
            missing_count / total_rows * 100
            if total_rows > 0
            else 0
        )

        non_missing_count = (
            total_rows - missing_count
        )

        completeness_pct = (
            non_missing_count / total_rows * 100
            if total_rows > 0
            else 0
        )

        records.append({
            "column": column,
            "data_type": str(df[column].dtype),
            "missing_count": missing_count,
            "missing_percentage": round(
                missing_pct,
                2
            ),
            "non_missing_count": non_missing_count,
            "completeness_percentage": round(
                completeness_pct,
                2
            )
        })

    summary = pd.DataFrame(
        records
    )

    summary = summary.sort_values(
        by=[
            "missing_count",
            "column"
        ],
        ascending=[
            False,
            True
        ]
    ).reset_index(
        drop=True
    )

    return summary


# ============================================================
# KEY FIELD COVERAGE
# ============================================================

def analyze_key_field_coverage(df):
    """
    Evaluate completeness of important analytical fields.
    """

    key_fields = [
        "zpid",
        "addressZipcode",
        "beds",
        "baths",
        "area",
        "unformattedPrice",
        "price",
        "zestimate",
        "taxAssessedValue",
        "lotAreaValue",
        "lotAreaSqFt",
        "pricePerSqFt",
        "logPrice",
        "daysOnZillow",
        "homeType",
        "priceChange",
        "hasPriceChange"
    ]

    records = []

    total_rows = len(df)

    for field in key_fields:

        if field not in df.columns:

            records.append({
                "field": field,
                "status": "Column not present",
                "missing_count": np.nan,
                "missing_percentage": np.nan,
                "non_missing_count": np.nan,
                "coverage_percentage": np.nan
            })

            continue

        missing_count = df[field].isna().sum()

        non_missing_count = (
            total_rows - missing_count
        )

        coverage_percentage = (
            non_missing_count
            / total_rows
            * 100
            if total_rows > 0
            else 0
        )

        records.append({
            "field": field,
            "status": "Present",
            "missing_count": missing_count,
            "missing_percentage": round(
                missing_count
                / total_rows
                * 100,
                2
            ),
            "non_missing_count": non_missing_count,
            "coverage_percentage": round(
                coverage_percentage,
                2
            )
        })

    return pd.DataFrame(
        records
    )


# ============================================================
# PROPERTY-TYPE MISSINGNESS
# ============================================================

def analyze_missingness_by_property_type(df):
    """
    Analyze missingness in key fields by property type.

    This helps determine whether missing values are
    concentrated within particular property categories.
    """

    fields = [
        "zestimate",
        "taxAssessedValue",
        "lotAreaSqFt",
        "daysOnZillow"
    ]

    if "homeType" not in df.columns:

        return pd.DataFrame()

    records = []

    for property_type, group in df.groupby(
        "homeType",
        dropna=False
    ):

        total = len(group)

        for field in fields:

            if field not in group.columns:

                continue

            missing = group[field].isna().sum()

            missing_pct = (
                missing
                / total
                * 100
                if total > 0
                else 0
            )

            records.append({
                "homeType": property_type,
                "listing_count": total,
                "field": field,
                "missing_count": missing,
                "missing_percentage": round(
                    missing_pct,
                    2
                )
            })

    return pd.DataFrame(
        records
    )


# ============================================================
# MACHINE-LEARNING COMPLETENESS
# ============================================================

def analyze_ml_completeness(df):
    """
    Evaluate completeness of the variables used by the
    machine-learning models.

    Predictors:
        area
        beds
        baths
        lotAreaSqFt
        taxAssessedValue
        daysOnZillow

    Target:
        logPrice
    """

    ml_columns = [
        "area",
        "beds",
        "baths",
        "lotAreaSqFt",
        "taxAssessedValue",
        "daysOnZillow",
        "logPrice"
    ]

    available_columns = [
        column
        for column in ml_columns
        if column in df.columns
    ]

    missing_table = df[
        available_columns
    ].isna().sum().to_frame(
        name="missing_count"
    )

    missing_table[
        "missing_percentage"
    ] = (
        missing_table["missing_count"]
        / len(df)
        * 100
    ).round(2)

    missing_table[
        "complete_count"
    ] = (
        len(df)
        - missing_table["missing_count"]
    )

    missing_table[
        "complete_percentage"
    ] = (
        missing_table["complete_count"]
        / len(df)
        * 100
    ).round(2)

    missing_table = (
        missing_table
        .reset_index()
        .rename(
            columns={
                "index": "field"
            }
        )
    )

    # Complete cases across the entire ML dataset
    complete_cases = (
        df[available_columns]
        .notna()
        .all(axis=1)
        .sum()
    )

    complete_case_percentage = (
        complete_cases
        / len(df)
        * 100
        if len(df) > 0
        else 0
    )

    complete_case_summary = pd.DataFrame({
        "metric": [
            "Total analysis-ready records",
            "Complete ML cases",
            "Incomplete ML cases",
            "Complete ML case percentage"
        ],
        "value": [
            len(df),
            complete_cases,
            len(df) - complete_cases,
            round(
                complete_case_percentage,
                2
            )
        ]
    })

    return (
        missing_table,
        complete_case_summary
    )


# ============================================================
# ZESTIMATE COVERAGE
# ============================================================

def analyze_zestimate_coverage(df):
    """
    Analyze Zestimate availability and coverage.
    """

    if "zestimate" not in df.columns:

        return pd.DataFrame({
            "metric": [
                "Zestimate column"
            ],
            "value": [
                "Not available"
            ]
        })

    total = len(df)

    missing = df["zestimate"].isna().sum()

    available = (
        total - missing
    )

    coverage = (
        available
        / total
        * 100
        if total > 0
        else 0
    )

    summary = pd.DataFrame({
        "metric": [
            "Total listings",
            "Listings with Zestimate",
            "Listings missing Zestimate",
            "Zestimate coverage percentage"
        ],
        "value": [
            total,
            available,
            missing,
            round(
                coverage,
                2
            )
        ]
    })

    return summary


# ============================================================
# TAX ASSESSED VALUE COVERAGE
# ============================================================

def analyze_tax_assessed_coverage(df):
    """
    Analyze tax-assessed-value availability and coverage.
    """

    if "taxAssessedValue" not in df.columns:

        return pd.DataFrame({
            "metric": [
                "Tax assessed value column"
            ],
            "value": [
                "Not available"
            ]
        })

    total = len(df)

    missing = (
        df["taxAssessedValue"]
        .isna()
        .sum()
    )

    available = (
        total - missing
    )

    coverage = (
        available
        / total
        * 100
        if total > 0
        else 0
    )

    summary = pd.DataFrame({
        "metric": [
            "Total listings",
            "Listings with tax assessed value",
            "Listings missing tax assessed value",
            "Tax assessed value coverage percentage"
        ],
        "value": [
            total,
            available,
            missing,
            round(
                coverage,
                2
            )
        ]
    })

    return summary


# ============================================================
# COORDINATE COVERAGE
# ============================================================

def analyze_coordinate_coverage(df):
    """
    Analyze latitude and longitude completeness.

    The final analytical dataset may contain either separate
    latitude/longitude fields or a latLong field depending
    on the processing stage.
    """

    coordinate_fields = []

    if "latitude" in df.columns:
        coordinate_fields.append(
            "latitude"
        )

    if "longitude" in df.columns:
        coordinate_fields.append(
            "longitude"
        )

    if coordinate_fields:

        records = []

        for field in coordinate_fields:

            missing = df[field].isna().sum()

            available = (
                len(df) - missing
            )

            coverage = (
                available
                / len(df)
                * 100
                if len(df) > 0
                else 0
            )

            records.append({
                "field": field,
                "missing_count": missing,
                "missing_percentage": round(
                    missing
                    / len(df)
                    * 100,
                    2
                ),
                "available_count": available,
                "coverage_percentage": round(
                    coverage,
                    2
                )
            })

        coordinate_summary = pd.DataFrame(
            records
        )

    elif "latLong" in df.columns:

        missing = df["latLong"].isna().sum()

        available = (
            len(df) - missing
        )

        coordinate_summary = pd.DataFrame({
            "field": [
                "latLong"
            ],
            "missing_count": [
                missing
            ],
            "missing_percentage": [
                round(
                    missing
                    / len(df)
                    * 100,
                    2
                )
            ],
            "available_count": [
                available
            ],
            "coverage_percentage": [
                round(
                    available
                    / len(df)
                    * 100,
                    2
                )
            ]
        })

    else:

        coordinate_summary = pd.DataFrame({
            "field": [
                "Coordinates"
            ],
            "missing_count": [
                np.nan
            ],
            "missing_percentage": [
                np.nan
            ],
            "available_count": [
                np.nan
            ],
            "coverage_percentage": [
                np.nan
            ]
        })

    return coordinate_summary


# ============================================================
# NUMERIC VALIDATION
# ============================================================

def validate_numeric_fields(df):
    """
    Check key numeric fields for invalid values.
    """

    checks = []

    numeric_rules = {
        "unformattedPrice": lambda x: x <= 1,
        "area": lambda x: x <= 0,
        "beds": lambda x: x < 0,
        "baths": lambda x: x < 0,
        "daysOnZillow": lambda x: x < 0,
        "pricePerSqFt": lambda x: x <= 0,
        "logPrice": lambda x: ~np.isfinite(x)
    }

    for field, rule in numeric_rules.items():

        if field not in df.columns:

            continue

        series = df[field]

        try:

            invalid_mask = (
                series.notna()
                &
                rule(series)
            )

            invalid_count = (
                invalid_mask.sum()
            )

        except Exception:

            invalid_count = np.nan

        checks.append({
            "field": field,
            "invalid_count": invalid_count
        })

    return pd.DataFrame(
        checks
    )


# ============================================================
# DATA TYPE SUMMARY
# ============================================================

def create_data_type_summary(df):
    """
    Summarize data types used in the dataset.
    """

    summary = pd.DataFrame({
        "column": df.columns,
        "data_type": [
            str(dtype)
            for dtype in df.dtypes
        ],
        "unique_values": [
            df[column].nunique(
                dropna=True
            )
            for column in df.columns
        ]
    })

    return summary


# ============================================================
# QUALITY STATUS
# ============================================================

def create_quality_status(
    profile,
    missingness,
    ml_summary,
    numeric_validation
):
    """
    Create a concise overall quality summary.
    """

    total_rows = int(
        profile.loc[
            profile["Metric"] == "Total rows",
            "Value"
        ].iloc[0]
    )

    total_columns = int(
        profile.loc[
            profile["Metric"] == "Total columns",
            "Value"
        ].iloc[0]
    )

    duplicate_count = int(
        profile.loc[
            profile["Metric"] == "Duplicate zpid records",
            "Value"
        ].iloc[0]
    )

    invalid_numeric_count = int(
        numeric_validation[
            "invalid_count"
        ]
        .fillna(0)
        .sum()
    )

    total_missing_cells = int(
        missingness[
            "missing_count"
        ].sum()
    )

    quality_summary = pd.DataFrame({
        "metric": [
            "Total rows",
            "Total columns",
            "Duplicate zpid records",
            "Total missing cells",
            "Invalid numeric values",
            "ML complete cases"
        ],
        "value": [
            total_rows,
            total_columns,
            duplicate_count,
            total_missing_cells,
            invalid_numeric_count,
            ml_summary.loc[
                ml_summary["metric"]
                == "Complete ML cases",
                "value"
            ].iloc[0]
        ]
    })

    return quality_summary


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(
    profile,
    missingness,
    key_coverage,
    property_type_missingness,
    ml_missingness,
    ml_summary,
    zestimate_coverage,
    tax_coverage,
    coordinate_coverage,
    numeric_validation,
    data_types,
    quality_summary
):
    """
    Save all data-quality reports.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    profile.to_csv(
        OUTPUT_DIR /
        "dataset_profile.csv",
        index=False
    )

    missingness.to_csv(
        OUTPUT_DIR /
        "missingness_summary.csv",
        index=False
    )

    key_coverage.to_csv(
        OUTPUT_DIR /
        "key_field_coverage.csv",
        index=False
    )

    property_type_missingness.to_csv(
        OUTPUT_DIR /
        "missingness_by_property_type.csv",
        index=False
    )

    ml_missingness.to_csv(
        OUTPUT_DIR /
        "ml_missingness_summary.csv",
        index=False
    )

    ml_summary.to_csv(
        OUTPUT_DIR /
        "ml_complete_case_summary.csv",
        index=False
    )

    zestimate_coverage.to_csv(
        OUTPUT_DIR /
        "zestimate_coverage.csv",
        index=False
    )

    tax_coverage.to_csv(
        OUTPUT_DIR /
        "tax_assessed_value_coverage.csv",
        index=False
    )

    coordinate_coverage.to_csv(
        OUTPUT_DIR /
        "coordinate_coverage.csv",
        index=False
    )

    numeric_validation.to_csv(
        OUTPUT_DIR /
        "numeric_validation.csv",
        index=False
    )

    data_types.to_csv(
        OUTPUT_DIR /
        "data_type_summary.csv",
        index=False
    )

    quality_summary.to_csv(
        OUTPUT_DIR /
        "quality_summary.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("DATA QUALITY REPORTS SAVED")
    print("=" * 70)

    print(
        f"\nOutput directory: {OUTPUT_DIR}"
    )

    print(
        "\nReports created:"
    )

    report_files = [
        "dataset_profile.csv",
        "missingness_summary.csv",
        "key_field_coverage.csv",
        "missingness_by_property_type.csv",
        "ml_missingness_summary.csv",
        "ml_complete_case_summary.csv",
        "zestimate_coverage.csv",
        "tax_assessed_value_coverage.csv",
        "coordinate_coverage.csv",
        "numeric_validation.csv",
        "data_type_summary.csv",
        "quality_summary.csv"
    ]

    for filename in report_files:

        print(
            f"  - {filename}"
        )


# ============================================================
# PRINT KEY FINDINGS
# ============================================================

def print_key_findings(
    df,
    missingness,
    ml_summary,
    zestimate_coverage,
    tax_coverage,
    coordinate_coverage,
    numeric_validation
):
    """
    Print the most important data-quality findings.
    """

    print("\n" + "=" * 70)
    print("KEY DATA-QUALITY FINDINGS")
    print("=" * 70)

    print(
        f"\nTotal properties: {len(df):,}"
    )

    print(
        f"Total columns: {len(df.columns):,}"
    )

    if "zpid" in df.columns:

        print(
            f"Unique properties: "
            f"{df['zpid'].nunique():,}"
        )

        print(
            f"Duplicate zpids: "
            f"{df['zpid'].duplicated().sum():,}"
        )

    print(
        "\nFields with missing values:"
    )

    nonzero_missing = missingness[
        missingness["missing_count"] > 0
    ]

    for _, row in nonzero_missing.iterrows():

        print(
            f"  {row['column']}: "
            f"{int(row['missing_count']):,} missing "
            f"({row['missing_percentage']:.2f}%)"
        )

    print(
        "\nZestimate coverage:"
    )

    print(
        zestimate_coverage.to_string(
            index=False
        )
    )

    print(
        "\nTax-assessed-value coverage:"
    )

    print(
        tax_coverage.to_string(
            index=False
        )
    )

    print(
        "\nCoordinate coverage:"
    )

    print(
        coordinate_coverage.to_string(
            index=False
        )
    )

    print(
        "\nMachine-learning completeness:"
    )

    print(
        ml_summary.to_string(
            index=False
        )
    )

    print(
        "\nInvalid numeric values:"
    )

    invalid_numeric = numeric_validation[
        numeric_validation["invalid_count"] > 0
    ]

    if invalid_numeric.empty:

        print(
            "  No invalid numeric values detected."
        )

    else:

        print(
            invalid_numeric.to_string(
                index=False
            )
        )


# ============================================================
# DOCUMENT DATA-QUALITY DECISIONS
# ============================================================

def create_methodology_notes():
    """
    Create a Markdown document describing the data-quality
    decisions used in the project.
    """

    notes = """
# Data Quality Methodology

## Dataset

The analysis-ready San Antonio Zillow dataset is the
authoritative dataset used for downstream analysis,
SQL analysis, Tableau dashboards, and machine-learning
modeling.

## Missing Values

Missing values are retained when they represent unavailable
source information rather than an obvious data error.

### Zestimate

Missing Zestimate values are retained as missing.

They are not replaced with listing price, tax-assessed value,
or an estimated substitute.

This preserves the distinction between an unavailable Zillow
valuation and an observed valuation.

### Tax-Assessed Value

Missing tax-assessed values are retained as missing.

Tax-assessed value is used only for analyses and models where
the field is available.

### Coordinates

Missing latitude or longitude values are retained as missing.

Properties are not removed solely because geographic
coordinates are unavailable.

## Machine Learning

Machine-learning models use complete cases for the required
predictors and target.

The primary predictors are:

- area
- beds
- baths
- lotAreaSqFt
- taxAssessedValue
- daysOnZillow

The target variable is:

- logPrice

Rows missing one or more required modeling variables are
excluded from the modeling subset rather than having values
artificially imputed.

## Outliers

Legitimate high-value properties are retained.

Extreme observations are investigated separately during
outlier and residual analysis rather than automatically
removed from the analytical dataset.

## Duplicate Properties

`zpid` is treated as the primary property identifier.

Duplicate property identifiers are monitored during
validation.

## Derived Variables

`pricePerSqFt` and `logPrice` are recalculated from the
validated listing price and living-area fields.

## Reproducibility

This script audits the dataset without modifying the
analysis-ready source file.

The generated reports document completeness, missingness,
numeric validity, and modeling readiness.
"""

    output_file = (
        OUTPUT_DIR /
        "DATA_QUALITY_METHODOLOGY.md"
    )

    output_file.write_text(
        notes.strip(),
        encoding="utf-8"
    )

    print(
        f"\nMethodology notes saved to: "
        f"{output_file}"
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)

    print(
        "PART 3 — DATA QUALITY & "
        "MISSING-VALUE ANALYSIS"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Dataset profile
    # --------------------------------------------------------

    profile = create_dataset_profile(
        df
    )

    # --------------------------------------------------------
    # Missingness
    # --------------------------------------------------------

    missingness = (
        create_missingness_summary(
            df
        )
    )

    # --------------------------------------------------------
    # Key field coverage
    # --------------------------------------------------------

    key_coverage = (
        analyze_key_field_coverage(
            df
        )
    )

    # --------------------------------------------------------
    # Missingness by property type
    # --------------------------------------------------------

    property_type_missingness = (
        analyze_missingness_by_property_type(
            df
        )
    )

    # --------------------------------------------------------
    # Machine-learning completeness
    # --------------------------------------------------------

    (
        ml_missingness,
        ml_summary
    ) = analyze_ml_completeness(
        df
    )

    # --------------------------------------------------------
    # Zestimate coverage
    # --------------------------------------------------------

    zestimate_coverage = (
        analyze_zestimate_coverage(
            df
        )
    )

    # --------------------------------------------------------
    # Tax assessed value coverage
    # --------------------------------------------------------

    tax_coverage = (
        analyze_tax_assessed_coverage(
            df
        )
    )

    # --------------------------------------------------------
    # Coordinate coverage
    # --------------------------------------------------------

    coordinate_coverage = (
        analyze_coordinate_coverage(
            df
        )
    )

    # --------------------------------------------------------
    # Numeric validation
    # --------------------------------------------------------

    numeric_validation = (
        validate_numeric_fields(
            df
        )
    )

    # --------------------------------------------------------
    # Data types
    # --------------------------------------------------------

    data_types = (
        create_data_type_summary(
            df
        )
    )

    # --------------------------------------------------------
    # Overall quality summary
    # --------------------------------------------------------

    quality_summary = (
        create_quality_status(
            profile,
            missingness,
            ml_summary,
            numeric_validation
        )
    )

    # --------------------------------------------------------
    # Print findings
    # --------------------------------------------------------

    print_key_findings(
        df,
        missingness,
        ml_summary,
        zestimate_coverage,
        tax_coverage,
        coordinate_coverage,
        numeric_validation
    )

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    save_reports(
        profile,
        missingness,
        key_coverage,
        property_type_missingness,
        ml_missingness,
        ml_summary,
        zestimate_coverage,
        tax_coverage,
        coordinate_coverage,
        numeric_validation,
        data_types,
        quality_summary
    )

    # --------------------------------------------------------
    # Save methodology notes
    # --------------------------------------------------------

    create_methodology_notes()

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "PART 3 DATA QUALITY ANALYSIS COMPLETE"
    )
    print("=" * 70)

    print(
        "\nThe analysis-ready dataset was audited "
        "but not modified."
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
