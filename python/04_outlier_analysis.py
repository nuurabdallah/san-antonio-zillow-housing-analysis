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
    "reports/outliers"
)

# Variables selected for outlier analysis.
#
# These correspond to important housing-market and modeling
# variables available in the analysis-ready dataset.

OUTLIER_VARIABLES = [
    "price",
    "area",
    "pricePerSqFt",
    "beds",
    "baths",
    "lotAreaSqFt",
    "taxAssessedValue",
    "daysOnZillow",
    "logPrice"
]

# Number of observations to display/save at each extreme.

EXTREME_N = 10


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():
    """
    Load the analysis-ready dataset.
    """

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nAnalysis dataset was not found:\n"
            f"{INPUT_FILE}\n\n"
            "Make sure Part 2 has been completed."
        )

    print("=" * 70)
    print("SAN ANTONIO ZILLOW OUTLIER ANALYSIS")
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
# PREPARE NUMERIC DATA
# ============================================================

def prepare_numeric_data(df):
    """
    Convert selected variables to numeric values where
    necessary without modifying the source dataset.
    """

    data = df.copy()

    for column in OUTLIER_VARIABLES:

        if column in data.columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    return data


# ============================================================
# IQR OUTLIER ANALYSIS
# ============================================================

def calculate_iqr_outliers(df):
    """
    Identify observations outside the standard 1.5 × IQR
    boundaries.

    Lower boundary:
        Q1 - 1.5 × IQR

    Upper boundary:
        Q3 + 1.5 × IQR
    """

    records = []

    for column in OUTLIER_VARIABLES:

        if column not in df.columns:
            continue

        series = df[column].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        median = series.quantile(0.50)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = (
            q1 - 1.5 * iqr
        )

        upper_bound = (
            q3 + 1.5 * iqr
        )

        lower_count = (
            series < lower_bound
        ).sum()

        upper_count = (
            series > upper_bound
        ).sum()

        total_outliers = (
            lower_count
            + upper_count
        )

        if len(series) > 0:

            outlier_percentage = (
                total_outliers
                / len(series)
                * 100
            )

        else:

            outlier_percentage = 0

        records.append({
            "variable": column,
            "n_valid": len(series),
            "q1": q1,
            "median": median,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "lower_outlier_count": lower_count,
            "upper_outlier_count": upper_count,
            "total_iqr_outliers": total_outliers,
            "outlier_percentage": round(
                outlier_percentage,
                2
            )
        })

    return pd.DataFrame(records)


# ============================================================
# Z-SCORE OUTLIER ANALYSIS
# ============================================================

def calculate_zscore_outliers(df):
    """
    Identify observations with absolute z-score >= 3.

    The z-score is calculated as:

        z = (x - mean) / standard deviation

    This is used as a screening method rather than an
    automatic removal rule.
    """

    records = []

    for column in OUTLIER_VARIABLES:

        if column not in df.columns:
            continue

        series = df[column].dropna()

        if len(series) < 2:
            continue

        mean_value = series.mean()
        std_value = series.std()

        if std_value == 0:

            outlier_count = 0
            positive_count = 0
            negative_count = 0

        else:

            z_scores = (
                (series - mean_value)
                / std_value
            )

            positive_count = (
                z_scores > 3
            ).sum()

            negative_count = (
                z_scores < -3
            ).sum()

            outlier_count = (
                z_scores.abs() >= 3
            ).sum()

        if len(series) > 0:

            outlier_percentage = (
                outlier_count
                / len(series)
                * 100
            )

        else:

            outlier_percentage = 0

        records.append({
            "variable": column,
            "n_valid": len(series),
            "mean": mean_value,
            "std": std_value,
            "negative_zscore_outliers": negative_count,
            "positive_zscore_outliers": positive_count,
            "total_zscore_outliers": outlier_count,
            "outlier_percentage": round(
                outlier_percentage,
                2
            )
        })

    return pd.DataFrame(records)


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

def create_descriptive_statistics(df):
    """
    Create detailed descriptive statistics for the variables
    under investigation.
    """

    available_columns = [
        column
        for column in OUTLIER_VARIABLES
        if column in df.columns
    ]

    if not available_columns:

        return pd.DataFrame()

    statistics = (
        df[available_columns]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
        .transpose()
    )

    statistics = statistics.reset_index()

    statistics = statistics.rename(
        columns={
            "index": "variable"
        }
    )

    return statistics


# ============================================================
# EXTREME OBSERVATIONS
# ============================================================

def create_extreme_observation_tables(df):
    """
    Create the highest and lowest observations for each
    variable.

    Each observation retains identifying and contextual
    information when those fields are available.
    """

    tables = {}

    context_columns = [
        "zpid",
        "address",
        "addressCity",
        "addressState",
        "addressZipcode",
        "zipcode",
        "homeType",
        "beds",
        "baths",
        "area",
        "price",
        "pricePerSqFt",
        "lotAreaSqFt",
        "taxAssessedValue",
        "daysOnZillow",
        "zestimate",
        "logPrice"
    ]

    context_columns = [
        column
        for column in context_columns
        if column in df.columns
    ]

    for variable in OUTLIER_VARIABLES:

        if variable not in df.columns:
            continue

        valid = df[
            df[variable].notna()
        ].copy()

        if valid.empty:
            continue

        highest = (
            valid.sort_values(
                by=variable,
                ascending=False
            )
            .head(EXTREME_N)
        )

        lowest = (
            valid.sort_values(
                by=variable,
                ascending=True
            )
            .head(EXTREME_N)
        )

        highest = highest[
            [
                column
                for column in context_columns
                if column in highest.columns
            ]
        ]

        lowest = lowest[
            [
                column
                for column in context_columns
                if column in lowest.columns
            ]
        ]

        tables[
            f"{variable}_highest"
        ] = highest

        tables[
            f"{variable}_lowest"
        ] = lowest

    return tables


# ============================================================
# FLAG IQR OUTLIERS AT RECORD LEVEL
# ============================================================

def create_record_level_iqr_flags(df):
    """
    Create an audit table showing which records fall outside
    the IQR boundaries.

    The original dataset is not modified.

    A separate audit dataframe is returned.
    """

    audit = df.copy()

    flag_columns = []

    for variable in OUTLIER_VARIABLES:

        if variable not in audit.columns:
            continue

        series = pd.to_numeric(
            audit[variable],
            errors="coerce"
        )

        valid = series.dropna()

        if valid.empty:
            continue

        q1 = valid.quantile(
            0.25
        )

        q3 = valid.quantile(
            0.75
        )

        iqr = q3 - q1

        lower_bound = (
            q1 - 1.5 * iqr
        )

        upper_bound = (
            q3 + 1.5 * iqr
        )

        flag_name = (
            f"{variable}_iqr_outlier"
        )

        audit[flag_name] = (
            series.lt(lower_bound)
            | series.gt(upper_bound)
        )

        audit[flag_name] = (
            audit[flag_name]
            .fillna(False)
        )

        flag_columns.append(
            flag_name
        )

    if flag_columns:

        audit[
            "total_iqr_outlier_flags"
        ] = audit[
            flag_columns
        ].sum(axis=1)

        audit[
            "has_iqr_outlier"
        ] = (
            audit[
                "total_iqr_outlier_flags"
            ] > 0
        )

    else:

        audit[
            "total_iqr_outlier_flags"
        ] = 0

        audit[
            "has_iqr_outlier"
        ] = False

    return audit


# ============================================================
# PROPERTY-TYPE OUTLIER SUMMARY
# ============================================================

def create_property_type_summary(df):
    """
    Compare descriptive statistics across property types.

    This is useful because an observation may appear extreme
    in the full market but be normal within its property type.
    """

    if "homeType" not in df.columns:

        return pd.DataFrame()

    variables = [
        "price",
        "area",
        "pricePerSqFt",
        "lotAreaSqFt",
        "taxAssessedValue"
    ]

    variables = [
        variable
        for variable in variables
        if variable in df.columns
    ]

    records = []

    for property_type, group in df.groupby(
        "homeType",
        dropna=False
    ):

        for variable in variables:

            series = group[
                variable
            ].dropna()

            if series.empty:
                continue

            q1 = series.quantile(
                0.25
            )

            q3 = series.quantile(
                0.75
            )

            iqr = q3 - q1

            lower_bound = (
                q1 - 1.5 * iqr
            )

            upper_bound = (
                q3 + 1.5 * iqr
            )

            outlier_count = (
                (series < lower_bound)
                |
                (series > upper_bound)
            ).sum()

            if len(series) > 0:

                outlier_percentage = (
                    outlier_count
                    / len(series)
                    * 100
                )

            else:

                outlier_percentage = 0

            records.append({
                "homeType": property_type,
                "variable": variable,
                "n": len(series),
                "mean": series.mean(),
                "median": series.median(),
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "iqr_outlier_count": outlier_count,
                "iqr_outlier_percentage": round(
                    outlier_percentage,
                    2
                )
            })

    return pd.DataFrame(records)


# ============================================================
# OUTLIER INTERSECTION SUMMARY
# ============================================================

def create_outlier_intersection_summary(
    record_level_audit
):
    """
    Summarize how many properties are flagged by multiple
    outlier variables.

    Multiple flags can indicate observations that deserve
    closer investigation.
    """

    if (
        "total_iqr_outlier_flags"
        not in record_level_audit.columns
    ):

        return pd.DataFrame()

    counts = (
        record_level_audit[
            "total_iqr_outlier_flags"
        ]
        .value_counts()
        .sort_index()
    )

    records = []

    total = len(
        record_level_audit
    )

    for flag_count, observation_count in (
        counts.items()
    ):

        if total > 0:

            percentage = (
                observation_count
                / total
                * 100
            )

        else:

            percentage = 0

        records.append({
            "iqr_flag_count": int(
                flag_count
            ),
            "property_count": int(
                observation_count
            ),
            "percentage_of_dataset": round(
                percentage,
                2
            )
        })

    return pd.DataFrame(records)


# ============================================================
# DATA-QUALITY VS LEGITIMATE-OUTLIER SCREEN
# ============================================================

def create_outlier_classification_summary(df):
    """
    Create a screening summary that distinguishes clearly
    invalid values from statistically unusual observations.

    This does NOT classify individual properties as erroneous.
    It documents rules that were already established during
    data cleaning.
    """

    records = []

    if "price" in df.columns:

        invalid_price = (
            df["price"].notna()
            &
            (df["price"] <= 1)
        ).sum()

        records.append({
            "variable": "price",
            "screening_rule": (
                "Price <= $1 is treated as an invalid/"
                "placeholder value during data cleaning."
            ),
            "invalid_data_count": int(
                invalid_price
            ),
            "treatment": (
                "Removed during Part 2 cleaning."
            )
        })

    if "area" in df.columns:

        invalid_area = (
            df["area"].notna()
            &
            (df["area"] <= 0)
        ).sum()

        records.append({
            "variable": "area",
            "screening_rule": (
                "Living area must be greater than zero."
            ),
            "invalid_data_count": int(
                invalid_area
            ),
            "treatment": (
                "No invalid observations remain."
            )
        })

    if "beds" in df.columns:

        invalid_beds = (
            df["beds"].notna()
            &
            (df["beds"] < 0)
        ).sum()

        records.append({
            "variable": "beds",
            "screening_rule": (
                "Bedroom count cannot be negative."
            ),
            "invalid_data_count": int(
                invalid_beds
            ),
            "treatment": (
                "No invalid observations remain."
            )
        })

    if "baths" in df.columns:

        invalid_baths = (
            df["baths"].notna()
            &
            (df["baths"] < 0)
        ).sum()

        records.append({
            "variable": "baths",
            "screening_rule": (
                "Bathroom count cannot be negative."
            ),
            "invalid_data_count": int(
                invalid_baths
            ),
            "treatment": (
                "No invalid observations remain."
            )
        })

    records.append({
        "variable": "statistical_outliers",
        "screening_rule": (
            "IQR and z-score methods identify unusual "
            "observations but do not establish that a "
            "property is erroneous."
        ),
        "invalid_data_count": 0,
        "treatment": (
            "Retained for investigation and modeling."
        )
    })

    return pd.DataFrame(
        records
    )


# ============================================================
# PRINT KEY FINDINGS
# ============================================================

def print_key_findings(
    df,
    iqr_summary,
    zscore_summary,
    property_type_summary,
    intersection_summary
):
    """
    Print the most important outlier findings.
    """

    print("\n" + "=" * 70)
    print("KEY OUTLIER FINDINGS")
    print("=" * 70)

    print(
        "\nNo observations were removed by this analysis."
    )

    print(
        "Statistical outliers are treated as observations "
        "requiring investigation."
    )

    print(
        "\nIQR outlier summary:"
    )

    if not iqr_summary.empty:

        display_columns = [
            "variable",
            "n_valid",
            "lower_bound",
            "upper_bound",
            "total_iqr_outliers",
            "outlier_percentage"
        ]

        print(
            iqr_summary[
                display_columns
            ].to_string(
                index=False
            )
        )

    print(
        "\nZ-score outlier summary "
        "(absolute z-score >= 3):"
    )

    if not zscore_summary.empty:

        display_columns = [
            "variable",
            "n_valid",
            "negative_zscore_outliers",
            "positive_zscore_outliers",
            "total_zscore_outliers",
            "outlier_percentage"
        ]

        print(
            zscore_summary[
                display_columns
            ].to_string(
                index=False
            )
        )

    print(
        "\nProperties flagged by multiple IQR rules:"
    )

    if not intersection_summary.empty:

        print(
            intersection_summary.to_string(
                index=False
            )
        )

    if not property_type_summary.empty:

        print(
            "\nProperty types included in "
            "outlier analysis:"
        )

        property_types = (
            df["homeType"]
            .value_counts(
                dropna=False
            )
        )

        for property_type, count in (
            property_types.items()
        ):

            print(
                f"  {property_type}: "
                f"{count:,}"
            )


# ============================================================
# METHODOLOGY NOTES
# ============================================================

def create_methodology_notes():
    """
    Save the methodology used for Part 4.
    """

    notes = """# Outlier Analysis Methodology

## Objective

Part 4 investigates unusually high or low observations
within the San Antonio Zillow analysis-ready dataset.

The purpose is to identify observations that may require
investigation and to understand the shape of the housing
market.

## Variables

The following variables are examined:

- price
- area
- pricePerSqFt
- beds
- baths
- lotAreaSqFt
- taxAssessedValue
- daysOnZillow
- logPrice

## IQR Method

The primary statistical screening method is the
1.5 × IQR rule.

Lower boundary:

Q1 - 1.5 × IQR

Upper boundary:

Q3 + 1.5 × IQR

Observations outside these boundaries are flagged as
statistical outliers.

## Z-Score Method

A secondary screening method identifies observations with
an absolute z-score of at least 3.

This method provides another perspective on extreme values,
particularly for approximately symmetric variables.

Because housing variables such as price and tax-assessed
value are often highly skewed, z-score results are not
interpreted as proof of erroneous data.

## Property-Type Context

Outlier behavior is also examined by property type.

This is important because a value that is extreme across the
entire market may be reasonable within a particular property
category.

## Legitimate Market Outliers

High-priced properties, large properties, unusual price-per-
square-foot values, and other extreme observations are not
automatically removed.

Real estate markets naturally contain legitimate extreme
properties.

## Data-Quality Rules

Part 2 established explicit data-cleaning rules for clearly
invalid values.

For example, listing prices of $1 or less were treated as
invalid or placeholder values.

Those rules are separate from statistical outlier detection.

## Modeling

Outlier analysis does not automatically determine which
observations should be excluded from machine learning.

Potential influence on regression and machine-learning models
is investigated later through residual diagnostics, influence
analysis, and model error analysis.

## Dataset Integrity

This script does not modify the analysis-ready dataset.

All outlier flags and reports are stored separately so that
the original analytical dataset remains reproducible.
"""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "OUTLIER_METHODOLOGY.md"
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
    descriptive_statistics,
    iqr_summary,
    zscore_summary,
    extreme_tables,
    record_level_audit,
    property_type_summary,
    intersection_summary,
    classification_summary
):
    """
    Save all Part 4 reports.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    descriptive_statistics.to_csv(
        OUTPUT_DIR
        / "descriptive_statistics.csv",
        index=False
    )

    iqr_summary.to_csv(
        OUTPUT_DIR
        / "iqr_outlier_summary.csv",
        index=False
    )

    zscore_summary.to_csv(
        OUTPUT_DIR
        / "zscore_outlier_summary.csv",
        index=False
    )

    property_type_summary.to_csv(
        OUTPUT_DIR
        / "property_type_outlier_summary.csv",
        index=False
    )

    intersection_summary.to_csv(
        OUTPUT_DIR
        / "outlier_intersection_summary.csv",
        index=False
    )

    classification_summary.to_csv(
        OUTPUT_DIR
        / "outlier_classification_summary.csv",
        index=False
    )

    record_level_audit.to_csv(
        OUTPUT_DIR
        / "record_level_iqr_flags.csv",
        index=False
    )

    for table_name, table in (
        extreme_tables.items()
    ):

        table.to_csv(
            OUTPUT_DIR
            / f"{table_name}.csv",
            index=False
        )

    print("\n" + "=" * 70)
    print("OUTLIER REPORTS SAVED")
    print("=" * 70)

    print(
        f"\nOutput directory: {OUTPUT_DIR}"
    )

    print(
        "\nCore reports created:"
    )

    core_reports = [
        "descriptive_statistics.csv",
        "iqr_outlier_summary.csv",
        "zscore_outlier_summary.csv",
        "property_type_outlier_summary.csv",
        "outlier_intersection_summary.csv",
        "outlier_classification_summary.csv",
        "record_level_iqr_flags.csv",
        "OUTLIER_METHODOLOGY.md"
    ]

    for filename in core_reports:

        print(
            f"  - {filename}"
        )

    print(
        "\nExtreme-observation tables created:"
    )

    for table_name in extreme_tables:

        print(
            f"  - {table_name}.csv"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "PART 4 — OUTLIER ANALYSIS"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Prepare numeric analysis copy
    # --------------------------------------------------------

    analysis_df = prepare_numeric_data(
        df
    )

    # --------------------------------------------------------
    # Descriptive statistics
    # --------------------------------------------------------

    descriptive_statistics = (
        create_descriptive_statistics(
            analysis_df
        )
    )

    # --------------------------------------------------------
    # IQR analysis
    # --------------------------------------------------------

    iqr_summary = (
        calculate_iqr_outliers(
            analysis_df
        )
    )

    # --------------------------------------------------------
    # Z-score analysis
    # --------------------------------------------------------

    zscore_summary = (
        calculate_zscore_outliers(
            analysis_df
        )
    )

    # --------------------------------------------------------
    # Extreme observations
    # --------------------------------------------------------

    extreme_tables = (
        create_extreme_observation_tables(
            analysis_df
        )
    )

    # --------------------------------------------------------
    # Record-level IQR audit
    # --------------------------------------------------------

    record_level_audit = (
        create_record_level_iqr_flags(
            analysis_df
        )
    )

    # --------------------------------------------------------
    # Property-type analysis
    # --------------------------------------------------------

    property_type_summary = (
        create_property_type_summary(
            analysis_df
        )
    )

    # --------------------------------------------------------
    # Multiple-outlier intersection
    # --------------------------------------------------------

    intersection_summary = (
        create_outlier_intersection_summary(
            record_level_audit
        )
    )

    # --------------------------------------------------------
    # Data-quality vs statistical outlier distinction
    # --------------------------------------------------------

    classification_summary = (
        create_outlier_classification_summary(
            analysis_df
        )
    )

    # --------------------------------------------------------
    # Print findings
    # --------------------------------------------------------

    print_key_findings(
        analysis_df,
        iqr_summary,
        zscore_summary,
        property_type_summary,
        intersection_summary
    )

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    save_reports(
        descriptive_statistics,
        iqr_summary,
        zscore_summary,
        extreme_tables,
        record_level_audit,
        property_type_summary,
        intersection_summary,
        classification_summary
    )

    # --------------------------------------------------------
    # Save methodology
    # --------------------------------------------------------

    create_methodology_notes()

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "PART 4 OUTLIER ANALYSIS COMPLETE"
    )
    print("=" * 70)

    print(
        "\nThe analysis-ready dataset was "
        "not modified."
    )

    print(
        "Outliers were identified for investigation, "
        "not automatically removed."
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
