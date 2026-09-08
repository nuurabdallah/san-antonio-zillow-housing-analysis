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
# LOAD DATASET
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

    print(f"\nInput file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns):,}")

    return df


# ============================================================
# DATASET PROFILE
# ============================================================

def create_dataset_profile(df):
    """
    Create a basic structural profile of the dataset.
    """

    if "zpid" in df.columns:
        unique_properties = df["zpid"].nunique()
        duplicate_properties = df["zpid"].duplicated().sum()
    else:
        unique_properties = np.nan
        duplicate_properties = np.nan

    memory_mb = (
        df.memory_usage(deep=True).sum()
        / (1024 ** 2)
    )

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
            unique_properties,
            duplicate_properties,
            round(memory_mb, 2)
        ]
    })

    return profile


# ============================================================
# MISSING-VALUE SUMMARY
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

        non_missing_count = (
            total_rows - missing_count
        )

        if total_rows > 0:

            missing_percentage = (
                missing_count
                / total_rows
                * 100
            )

            completeness_percentage = (
                non_missing_count
                / total_rows
                * 100
            )

        else:

            missing_percentage = 0
            completeness_percentage = 0

        records.append({
            "column": column,
            "data_type": str(
                df[column].dtype
            ),
            "missing_count": missing_count,
            "missing_percentage": round(
                missing_percentage,
                2
            ),
            "non_missing_count": non_missing_count,
            "completeness_percentage": round(
                completeness_percentage,
                2
            )
        })

    summary = pd.DataFrame(records)

    summary = summary.sort_values(
        by=[
            "missing_count",
            "column"
        ],
        ascending=[
            False,
            True
        ]
    )

    summary = summary.reset_index(
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
                "coverage_percentage": np.nan
            })

            continue

        missing_count = df[field].isna().sum()

        non_missing_count = (
            total_rows - missing_count
        )

        if total_rows > 0:

            missing_percentage = (
                missing_count
                / total_rows
                * 100
            )

            coverage_percentage = (
                non_missing_count
                / total_rows
                * 100
            )

        else:

            missing_percentage = 0
            coverage_percentage = 0

        records.append({
            "field": field,
            "status": "Present",
            "missing_count": missing_count,
            "missing_percentage": round(
                missing_percentage,
                2
            ),
            "coverage_percentage": round(
                coverage_percentage,
                2
            )
        })

    return pd.DataFrame(records)


# ============================================================
# MISSINGNESS BY PROPERTY TYPE
# ============================================================

def analyze_missingness_by_property_type(df):
    """
    Analyze missingness in important fields by property type.
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

            missing_count = (
                group[field]
                .isna()
                .sum()
            )

            if total > 0:

                missing_percentage = (
                    missing_count
                    / total
                    * 100
                )

            else:

                missing_percentage = 0

            records.append({
                "homeType": property_type,
                "listing_count": total,
                "field": field,
                "missing_count": missing_count,
                "missing_percentage": round(
                    missing_percentage,
                    2
                )
            })

    return pd.DataFrame(records)


# ============================================================
# MACHINE-LEARNING COMPLETENESS
# ============================================================

def analyze_ml_completeness(df):
    """
    Evaluate completeness of variables used by the
    machine-learning workflow.
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

    records = []

    for field in available_columns:

        missing_count = (
            df[field]
            .isna()
            .sum()
        )

        complete_count = (
            len(df)
            - missing_count
        )

        if len(df) > 0:

            missing_percentage = (
                missing_count
                / len(df)
                * 100
            )

            complete_percentage = (
                complete_count
                / len(df)
                * 100
            )

        else:

            missing_percentage = 0
            complete_percentage = 0

        records.append({
            "field": field,
            "missing_count": missing_count,
            "missing_percentage": round(
                missing_percentage,
                2
            ),
            "complete_count": complete_count,
            "complete_percentage": round(
                complete_percentage,
                2
            )
        })

    missingness = pd.DataFrame(
        records
    )

    if available_columns:

        complete_cases = (
            df[available_columns]
            .notna()
            .all(axis=1)
            .sum()
        )

    else:

        complete_cases = 0

    incomplete_cases = (
        len(df)
        - complete_cases
    )

    if len(df) > 0:

        complete_case_percentage = (
            complete_cases
            / len(df)
            * 100
        )

    else:

        complete_case_percentage = 0

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
            incomplete_cases,
            round(
                complete_case_percentage,
                2
            )
        ]
    })

    return (
        missingness,
        complete_case_summary
    )


# ============================================================
# ZESTIMATE COVERAGE
# ============================================================

def analyze_zestimate_coverage(df):
    """
    Calculate Zestimate coverage.
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

    missing = (
        df["zestimate"]
        .isna()
        .sum()
    )

    available = (
        total - missing
    )

    if total > 0:

        coverage = (
            available
            / total
            * 100
        )

    else:

        coverage = 0

    return pd.DataFrame({
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


# ============================================================
# TAX ASSESSED VALUE COVERAGE
# ============================================================

def analyze_tax_assessed_coverage(df):
    """
    Calculate tax-assessed-value coverage.
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

    if total > 0:

        coverage = (
            available
            / total
            * 100
        )

    else:

        coverage = 0

    return pd.DataFrame({
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


# ============================================================
# COORDINATE COVERAGE
# ============================================================

def analyze_coordinate_coverage(df):
    """
    Analyze latitude and longitude completeness.
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

    records = []

    if coordinate_fields:

        for field in coordinate_fields:

            missing = (
                df[field]
                .isna()
                .sum()
            )

            available = (
                len(df) - missing
            )

            if len(df) > 0:

                missing_percentage = (
                    missing
                    / len(df)
                    * 100
                )

                coverage_percentage = (
                    available
                    / len(df)
                    * 100
                )

            else:

                missing_percentage = 0
                coverage_percentage = 0

            records.append({
                "field": field,
                "missing_count": missing,
                "missing_percentage": round(
                    missing_percentage,
                    2
                ),
                "available_count": available,
                "coverage_percentage": round(
                    coverage_percentage,
                    2
                )
            })

    elif "latLong" in df.columns:

        missing = (
            df["latLong"]
            .isna()
            .sum()
        )

        available = (
            len(df) - missing
        )

        if len(df) > 0:

            missing_percentage = (
                missing
                / len(df)
                * 100
            )

            coverage_percentage = (
                available
                / len(df)
                * 100
            )

        else:

            missing_percentage = 0
            coverage_percentage = 0

        records.append({
            "field": "latLong",
            "missing_count": missing,
            "missing_percentage": round(
                missing_percentage,
                2
            ),
            "available_count": available,
            "coverage_percentage": round(
                coverage_percentage,
                2
            )
        })

    else:

        records.append({
            "field": "Coordinates",
            "missing_count": np.nan,
            "missing_percentage": np.nan,
            "available_count": np.nan,
            "coverage_percentage": np.nan
        })

    return pd.DataFrame(records)


# ============================================================
# NUMERIC VALIDATION
# ============================================================

def validate_numeric_fields(df):
    """
    Check important numeric fields for invalid values.
    """

    records = []

    if "unformattedPrice" in df.columns:

        invalid = (
            df["unformattedPrice"].notna()
            &
            (df["unformattedPrice"] <= 1)
        )

        records.append({
            "field": "unformattedPrice",
            "invalid_count": invalid.sum(),
            "rule": "Value must be greater than $1"
        })

    if "area" in df.columns:

        invalid = (
            df["area"].notna()
            &
            (df["area"] <= 0)
        )

        records.append({
            "field": "area",
            "invalid_count": invalid.sum(),
            "rule": "Value must be greater than 0"
        })

    if "beds" in df.columns:

        invalid = (
            df["beds"].notna()
            &
            (df["beds"] < 0)
        )

        records.append({
            "field": "beds",
            "invalid_count": invalid.sum(),
            "rule": "Value cannot be negative"
        })

    if "baths" in df.columns:

        invalid = (
            df["baths"].notna()
            &
            (df["baths"] < 0)
        )

        records.append({
            "field": "baths",
            "invalid_count": invalid.sum(),
            "rule": "Value cannot be negative"
        })

    if "daysOnZillow" in df.columns:

        invalid = (
            df["daysOnZillow"].notna()
            &
            (df["daysOnZillow"] < 0)
        )

        records.append({
            "field": "daysOnZillow",
            "invalid_count": invalid.sum(),
            "rule": "Value cannot be negative"
        })

    if "pricePerSqFt" in df.columns:

        invalid = (
            df["pricePerSqFt"].notna()
            &
            (df["pricePerSqFt"] <= 0)
        )

        records.append({
            "field": "pricePerSqFt",
            "invalid_count": invalid.sum(),
            "rule": "Value must be greater than 0"
        })

    if "logPrice" in df.columns:

        invalid = (
            df["logPrice"].notna()
            &
            ~np.isfinite(
                df["logPrice"]
            )
        )

        records.append({
            "field": "logPrice",
            "invalid_count": invalid.sum(),
            "rule": "Value must be finite"
        })

    return pd.DataFrame(records)


# ============================================================
# DATA TYPE SUMMARY
# ============================================================

def create_data_type_summary(df):
    """
    Summarize data types and unique values.
    """

    records = []

    for column in df.columns:

        records.append({
            "column": column,
            "data_type": str(
                df[column].dtype
            ),
            "unique_values": df[column].nunique(
                dropna=True
            )
        })

    return pd.DataFrame(records)


# ============================================================
# OVERALL QUALITY SUMMARY
# ============================================================

def create_quality_summary(
    df,
    missingness,
    numeric_validation,
    ml_summary
):
    """
    Create a concise overall quality summary.
    """

    if "zpid" in df.columns:

        duplicate_count = (
            df["zpid"]
            .duplicated()
            .sum()
        )

        unique_properties = (
            df["zpid"]
            .nunique()
        )

    else:

        duplicate_count = np.nan
        unique_properties = np.nan

    total_missing_cells = int(
        missingness[
            "missing_count"
        ].sum()
    )

    invalid_numeric_count = int(
        numeric_validation[
            "invalid_count"
        ]
        .fillna(0)
        .sum()
    )

    complete_ml_cases = (
        ml_summary.loc[
            ml_summary["metric"]
            == "Complete ML cases",
            "value"
        ].iloc[0]
    )

    return pd.DataFrame({
        "metric": [
            "Total records",
            "Total columns",
            "Unique properties",
            "Duplicate zpid records",
            "Total missing cells",
            "Invalid numeric values",
            "Complete ML cases"
        ],
        "value": [
            len(df),
            len(df.columns),
            unique_properties,
            duplicate_count,
            total_missing_cells,
            invalid_numeric_count,
            complete_ml_cases
        ]
    })


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
    Print important data-quality findings.
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

    if nonzero_missing.empty:

        print(
            "  No missing values detected."
        )

    else:

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
# METHODOLOGY NOTES
# ============================================================

def create_methodology_notes():
    """
    Save the project's data-quality methodology.
    """

    notes = """# Data Quality Methodology

## Dataset

The analysis-ready San Antonio Zillow dataset is the
authoritative dataset used for downstream SQL analysis,
Tableau dashboards, exploratory analysis, and machine-learning
modeling.

## Missing Values

Missing values are retained when they represent unavailable
source information rather than an obvious data error.

### Zestimate

Missing Zestimate values are retained as missing.

They are not replaced with listing price, tax-assessed value,
or another estimated value.

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

zpid is treated as the primary property identifier.

Duplicate property identifiers are monitored during
validation.

## Derived Variables

pricePerSqFt and logPrice are recalculated from the validated
listing price and living-area fields.

## Reproducibility

This script audits the dataset without modifying the
analysis-ready source file.

The generated reports document completeness, missingness,
numeric validity, and modeling readiness.
"""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "DATA_QUALITY_METHODOLOGY.md"
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
        OUTPUT_DIR / "dataset_profile.csv",
        index=False
    )

    missingness.to_csv(
        OUTPUT_DIR / "missingness_summary.csv",
        index=False
    )

    key_coverage.to_csv(
        OUTPUT_DIR / "key_field_coverage.csv",
        index=False
    )

    property_type_missingness.to_csv(
        OUTPUT_DIR / "missingness_by_property_type.csv",
        index=False
    )

    ml_missingness.to_csv(
        OUTPUT_DIR / "ml_missingness_summary.csv",
        index=False
    )

    ml_summary.to_csv(
        OUTPUT_DIR / "ml_complete_case_summary.csv",
        index=False
    )

    zestimate_coverage.to_csv(
        OUTPUT_DIR / "zestimate_coverage.csv",
        index=False
    )

    tax_coverage.to_csv(
        OUTPUT_DIR / "tax_assessed_value_coverage.csv",
        index=False
    )

    coordinate_coverage.to_csv(
        OUTPUT_DIR / "coordinate_coverage.csv",
        index=False
    )

    numeric_validation.to_csv(
        OUTPUT_DIR / "numeric_validation.csv",
        index=False
    )

    data_types.to_csv(
        OUTPUT_DIR / "data_type_summary.csv",
        index=False
    )

    quality_summary.to_csv(
        OUTPUT_DIR / "quality_summary.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("DATA QUALITY REPORTS SAVED")
    print("=" * 70)

    print(
        f"\nOutput directory: {OUTPUT_DIR}"
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
        "quality_summary.csv",
        "DATA_QUALITY_METHODOLOGY.md"
    ]

    print("\nReports created:")

    for filename in report_files:

        print(
            f"  - {filename}"
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

    # Load data
    df = load_dataset()

    # Dataset profile
    profile = create_dataset_profile(
        df
    )

    # Missingness
    missingness = create_missingness_summary(
        df
    )

    # Key field coverage
    key_coverage = analyze_key_field_coverage(
        df
    )

    # Missingness by property type
    property_type_missingness = (
        analyze_missingness_by_property_type(
            df
        )
    )

    # Machine-learning completeness
    (
        ml_missingness,
        ml_summary
    ) = analyze_ml_completeness(
        df
    )

    # Zestimate coverage
    zestimate_coverage = analyze_zestimate_coverage(
        df
    )

    # Tax-assessed-value coverage
    tax_coverage = analyze_tax_assessed_coverage(
        df
    )

    # Coordinate coverage
    coordinate_coverage = analyze_coordinate_coverage(
        df
    )

    # Numeric validation
    numeric_validation = validate_numeric_fields(
        df
    )

    # Data types
    data_types = create_data_type_summary(
        df
    )

    # Overall quality summary
    quality_summary = create_quality_summary(
        df,
        missingness,
        numeric_validation,
        ml_summary
    )

    # Print findings
    print_key_findings(
        df,
        missingness,
        ml_summary,
        zestimate_coverage,
        tax_coverage,
        coordinate_coverage,
        numeric_validation
    )

    # Save reports
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

    # Save methodology
    create_methodology_notes()

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
