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

# Variables included in outlier analysis.

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

# Number of extreme observations saved per variable.

EXTREME_N = 10

# Minimum number of valid observations required before
# calculating property-type-specific IQR boundaries.

MIN_GROUP_SIZE = 10


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():
    """
    Load the analysis-ready Zillow dataset.
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
    Create an analysis copy and convert selected variables
    to numeric values.

    The original dataframe is not modified.
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
# DESCRIPTIVE STATISTICS
# ============================================================

def create_descriptive_statistics(df):
    """
    Create detailed descriptive statistics for variables
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
# MARKET-WIDE IQR OUTLIER ANALYSIS
# ============================================================

def calculate_iqr_outliers(df):
    """
    Identify observations outside the standard 1.5 × IQR
    boundaries across the entire dataset.

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

        outlier_percentage = (
            total_outliers
            / len(series)
            * 100
        )

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
# MARKET-WIDE Z-SCORE ANALYSIS
# ============================================================

def calculate_zscore_outliers(df):
    """
    Identify observations with absolute z-score >= 3.

    z = (x - mean) / standard deviation

    Z-score results are used only as a screening method.
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

            negative_count = 0
            positive_count = 0
            outlier_count = 0

        else:

            z_scores = (
                (series - mean_value)
                / std_value
            )

            negative_count = (
                z_scores < -3
            ).sum()

            positive_count = (
                z_scores > 3
            ).sum()

            outlier_count = (
                z_scores.abs() >= 3
            ).sum()

        outlier_percentage = (
            outlier_count
            / len(series)
            * 100
        )

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
# PROPERTY-TYPE IQR ANALYSIS
# ============================================================

def calculate_property_type_iqr(df):
    """
    Calculate IQR boundaries separately within each
    property type.

    A property type must have at least MIN_GROUP_SIZE valid
    observations for a variable before an IQR rule is applied.

    Smaller groups are reported as insufficient sample.
    """

    records = []

    if "homeType" not in df.columns:

        return pd.DataFrame()

    for property_type, group in df.groupby(
        "homeType",
        dropna=False
    ):

        for variable in OUTLIER_VARIABLES:

            if variable not in group.columns:
                continue

            series = group[
                variable
            ].dropna()

            n_valid = len(series)

            if n_valid < MIN_GROUP_SIZE:

                records.append({
                    "homeType": property_type,
                    "variable": variable,
                    "n_valid": n_valid,
                    "sample_status": (
                        "Insufficient Sample"
                    ),
                    "q1": np.nan,
                    "median": (
                        series.median()
                        if n_valid > 0
                        else np.nan
                    ),
                    "q3": np.nan,
                    "iqr": np.nan,
                    "lower_bound": np.nan,
                    "upper_bound": np.nan,
                    "iqr_outlier_count": np.nan,
                    "outlier_percentage": np.nan
                })

                continue

            q1 = series.quantile(
                0.25
            )

            median = series.quantile(
                0.50
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

            outlier_percentage = (
                outlier_count
                / n_valid
                * 100
            )

            records.append({
                "homeType": property_type,
                "variable": variable,
                "n_valid": n_valid,
                "sample_status": (
                    "Sufficient Sample"
                ),
                "q1": q1,
                "median": median,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "iqr_outlier_count": (
                    outlier_count
                ),
                "outlier_percentage": round(
                    outlier_percentage,
                    2
                )
            })

    return pd.DataFrame(records)


# ============================================================
# PROPERTY-TYPE OUTLIER SUMMARY
# ============================================================

def create_property_type_summary(df):
    """
    Create a property-type descriptive summary.

    This provides context for interpreting extreme values.
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

            records.append({
                "homeType": property_type,
                "variable": variable,
                "n": len(series),
                "mean": series.mean(),
                "median": series.median(),
                "min": series.min(),
                "max": series.max(),
                "std": series.std()
            })

    return pd.DataFrame(records)


# ============================================================
# RECORD-LEVEL MARKET-WIDE FLAGS
# ============================================================

def create_record_level_market_flags(df):
    """
    Create record-level market-wide IQR flags.

    The original dataset is not modified.
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
            f"{variable}_market_iqr_outlier"
        )

        audit[flag_name] = (
            series.lt(lower_bound)
            |
            series.gt(upper_bound)
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
            "market_iqr_flag_count"
        ] = audit[
            flag_columns
        ].sum(axis=1)

        audit[
            "has_market_iqr_outlier"
        ] = (
            audit[
                "market_iqr_flag_count"
            ] > 0
        )

    else:

        audit[
            "market_iqr_flag_count"
        ] = 0

        audit[
            "has_market_iqr_outlier"
        ] = False

    return audit


# ============================================================
# RECORD-LEVEL PROPERTY-TYPE FLAGS
# ============================================================

def create_record_level_property_type_flags(
    df,
    property_type_iqr
):
    """
    Apply property-type-specific IQR boundaries to each
    individual property.

    Groups with fewer than MIN_GROUP_SIZE observations
    are not flagged.

    Returns a separate audit dataframe.
    """

    audit = df.copy()

    if "homeType" not in audit.columns:

        audit[
            "property_type_iqr_flag_count"
        ] = 0

        audit[
            "has_property_type_iqr_outlier"
        ] = False

        return audit

    flag_columns = []

    sufficient_rules = property_type_iqr[
        property_type_iqr[
            "sample_status"
        ] == "Sufficient Sample"
    ].copy()

    for _, rule in sufficient_rules.iterrows():

        property_type = rule[
            "homeType"
        ]

        variable = rule[
            "variable"
        ]

        if variable not in audit.columns:
            continue

        column_name = (
            f"{variable}_property_type_iqr_outlier"
        )

        if column_name not in audit.columns:

            audit[column_name] = False

        mask = (
            audit["homeType"]
            == property_type
        )

        values = pd.to_numeric(
            audit[variable],
            errors="coerce"
        )

        outlier_mask = (
            mask
            &
            (
                values.lt(
                    rule["lower_bound"]
                )
                |
                values.gt(
                    rule["upper_bound"]
                )
            )
        )

        audit.loc[
            outlier_mask,
            column_name
        ] = True

        if column_name not in flag_columns:

            flag_columns.append(
                column_name
            )

    if flag_columns:

        audit[
            "property_type_iqr_flag_count"
        ] = audit[
            flag_columns
        ].sum(axis=1)

        audit[
            "has_property_type_iqr_outlier"
        ] = (
            audit[
                "property_type_iqr_flag_count"
            ] > 0
        )

    else:

        audit[
            "property_type_iqr_flag_count"
        ] = 0

        audit[
            "has_property_type_iqr_outlier"
        ] = False

    return audit


# ============================================================
# COMPARISON OF MARKET VS PROPERTY-TYPE FLAGS
# ============================================================

def create_outlier_comparison(audit):
    """
    Compare market-wide and property-type-specific
    outlier classifications.
    """

    market_flag = (
        audit[
            "has_market_iqr_outlier"
        ]
    )

    property_flag = (
        audit[
            "has_property_type_iqr_outlier"
        ]
    )

    conditions = [
        (~market_flag) & (~property_flag),
        market_flag & (~property_flag),
        (~market_flag) & property_flag,
        market_flag & property_flag
    ]

    choices = [
        "No IQR Flag",
        "Market-Wide Only",
        "Property-Type Only",
        "Both Market-Wide and Property-Type"
    ]

    audit[
        "outlier_comparison"
    ] = np.select(
        conditions,
        choices,
        default="Unknown"
    )

    summary = (
        audit[
            "outlier_comparison"
        ]
        .value_counts()
        .rename_axis(
            "classification"
        )
        .reset_index(
            name="property_count"
        )
    )

    summary[
        "percentage_of_dataset"
    ] = (
        summary["property_count"]
        / len(audit)
        * 100
    ).round(2)

    return audit, summary


# ============================================================
# MULTI-OUTLIER SUMMARY
# ============================================================

def create_outlier_intersection_summary(
    audit
):
    """
    Summarize the number of market-wide IQR flags per
    property.
    """

    if (
        "market_iqr_flag_count"
        not in audit.columns
    ):

        return pd.DataFrame()

    counts = (
        audit[
            "market_iqr_flag_count"
        ]
        .value_counts()
        .sort_index()
    )

    records = []

    total = len(audit)

    for flag_count, property_count in (
        counts.items()
    ):

        percentage = (
            property_count
            / total
            * 100
        )

        records.append({
            "iqr_flag_count": int(
                flag_count
            ),
            "property_count": int(
                property_count
            ),
            "percentage_of_dataset": round(
                percentage,
                2
            )
        })

    return pd.DataFrame(records)


# ============================================================
# EXTREME OBSERVATIONS
# ============================================================

def create_extreme_observation_tables(df):
    """
    Save the highest and lowest observations for every
    outlier-analysis variable.
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
            context_columns
        ]

        lowest = lowest[
            context_columns
        ]

        tables[
            f"{variable}_highest"
        ] = highest

        tables[
            f"{variable}_lowest"
        ] = lowest

    return tables


# ============================================================
# DATA-QUALITY VS STATISTICAL OUTLIERS
# ============================================================

def create_outlier_classification_summary(df):
    """
    Document the difference between invalid values and
    statistically unusual but potentially legitimate values.
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
                "Price <= $1 is treated as an "
                "invalid or placeholder value."
            ),
            "invalid_data_count": int(
                invalid_price
            ),
            "treatment": (
                "No such observations remain "
                "after Part 2 cleaning."
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
            "IQR and z-score methods identify "
            "unusual observations."
        ),
        "invalid_data_count": 0,
        "treatment": (
            "Statistical outliers are retained unless "
            "a separate data-quality investigation "
            "establishes that the observation is invalid."
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
    property_type_iqr,
    comparison_summary,
    intersection_summary
):
    """
    Print the most important findings from Part 4.
    """

    print("\n" + "=" * 70)
    print("KEY OUTLIER FINDINGS")
    print("=" * 70)

    print(
        "\nNo observations were removed."
    )

    print(
        "Statistical outliers are treated as observations "
        "requiring investigation."
    )

    print(
        "\nMarket-wide IQR outlier summary:"
    )

    if not iqr_summary.empty:

        columns = [
            "variable",
            "n_valid",
            "lower_bound",
            "upper_bound",
            "total_iqr_outliers",
            "outlier_percentage"
        ]

        print(
            iqr_summary[
                columns
            ].to_string(
                index=False
            )
        )

    print(
        "\nMarket-wide z-score outlier summary "
        "(absolute z-score >= 3):"
    )

    if not zscore_summary.empty:

        columns = [
            "variable",
            "n_valid",
            "negative_zscore_outliers",
            "positive_zscore_outliers",
            "total_zscore_outliers",
            "outlier_percentage"
        ]

        print(
            zscore_summary[
                columns
            ].to_string(
                index=False
            )
        )

    print(
        "\nProperty-type-specific IQR analysis:"
    )

    if not property_type_iqr.empty:

        display_columns = [
            "homeType",
            "variable",
            "n_valid",
            "sample_status",
            "iqr_outlier_count",
            "outlier_percentage"
        ]

        print(
            property_type_iqr[
                display_columns
            ].to_string(
                index=False
            )
        )

    print(
        "\nMarket-wide vs property-type IQR comparison:"
    )

    if not comparison_summary.empty:

        print(
            comparison_summary.to_string(
                index=False
            )
        )

    print(
        "\nNumber of market-wide IQR flags per property:"
    )

    if not intersection_summary.empty:

        print(
            intersection_summary.to_string(
                index=False
            )
        )

    if "homeType" in df.columns:

        print(
            "\nProperty-type sample sizes:"
        )

        counts = (
            df["homeType"]
            .value_counts(
                dropna=False
            )
        )

        for property_type, count in (
            counts.items()
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

    notes = f"""# Outlier Analysis Methodology

## Objective

Part 4 investigates unusually high or low observations
within the San Antonio Zillow analysis-ready dataset.

The purpose is to identify observations that may require
investigation and to understand the distribution of the
housing market.

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

## Market-Wide IQR Analysis

The primary statistical screening method is the
1.5 × IQR rule.

Lower boundary:

Q1 - 1.5 × IQR

Upper boundary:

Q3 + 1.5 × IQR

Observations outside these boundaries are flagged as
market-wide statistical outliers.

## Z-Score Analysis

A secondary screening method identifies observations with
an absolute z-score of at least 3.

This provides another perspective on extreme observations.

Because housing variables such as price, lot area, and
tax-assessed value can be highly skewed, z-score results
are treated as screening evidence rather than proof of
invalid data.

## Property-Type-Specific IQR Analysis

Outlier boundaries are also calculated separately within
each property type.

A minimum of {MIN_GROUP_SIZE} valid observations is required
before a property-type/variable combination receives an
IQR-based outlier classification.

Groups below this threshold are labeled:

"Insufficient Sample"

This prevents very small groups from producing unreliable
statistical classifications.

## Why Property Type Matters

A property that is extreme relative to the entire market
may be normal within its property category.

For example, luxury single-family properties can naturally
have substantially higher prices and living areas than the
broader market.

Property-type-specific analysis therefore provides additional
context for interpreting market-wide outliers.

## Legitimate Market Outliers

Extreme properties are not automatically removed.

Real estate markets naturally contain legitimate observations
at both ends of the distribution.

An expensive property, large property, or unusually high
price-per-square-foot property may represent real market
behavior.

## Data-Quality Rules

Part 2 established explicit rules for clearly invalid values.

For example:

- Listing prices <= $1 were treated as invalid or
  placeholder values.
- Living area must be greater than zero.
- Bedroom and bathroom counts cannot be negative.

These rules are separate from statistical outlier detection.

## Modeling

Outlier analysis does not determine the final machine-learning
training set by itself.

Potential model influence is investigated later through
residual diagnostics, influence measures, cross-validation,
and model error analysis.

## Dataset Integrity

This script does not modify the analysis-ready dataset.

All outlier flags and reports are stored separately.

This preserves reproducibility and allows legitimate extreme
properties to remain available for downstream analysis.
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
    property_type_iqr,
    property_type_summary,
    comparison_summary,
    intersection_summary,
    classification_summary,
    record_level_audit,
    extreme_tables
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

    property_type_iqr.to_csv(
        OUTPUT_DIR
        / "property_type_iqr_outlier_summary.csv",
        index=False
    )

    property_type_summary.to_csv(
        OUTPUT_DIR
        / "property_type_outlier_summary.csv",
        index=False
    )

    comparison_summary.to_csv(
        OUTPUT_DIR
        / "market_vs_property_type_outliers.csv",
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
        / "record_level_outlier_flags.csv",
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

    core_reports = [
        "descriptive_statistics.csv",
        "iqr_outlier_summary.csv",
        "zscore_outlier_summary.csv",
        "property_type_iqr_outlier_summary.csv",
        "property_type_outlier_summary.csv",
        "market_vs_property_type_outliers.csv",
        "outlier_intersection_summary.csv",
        "outlier_classification_summary.csv",
        "record_level_outlier_flags.csv",
        "OUTLIER_METHODOLOGY.md"
    ]

    print(
        "\nCore reports created:"
    )

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
    # Load dataset
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Create analysis copy
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
    # Market-wide IQR analysis
    # --------------------------------------------------------

    iqr_summary = (
        calculate_iqr_outliers(
            analysis_df
        )
    )

    # --------------------------------------------------------
    # Market-wide z-score analysis
    # --------------------------------------------------------

    zscore_summary = (
        calculate_zscore_outliers(
            analysis_df
        )
    )

    # --------------------------------------------------------
    # Property-type-specific IQR analysis
    # --------------------------------------------------------

    property_type_iqr = (
        calculate_property_type_iqr(
            analysis_df
        )
    )

    # --------------------------------------------------------
    # Property-type descriptive summary
    # --------------------------------------------------------

    property_type_summary = (
        create_property_type_summary(
            analysis_df
        )
    )

    # --------------------------------------------------------
    # Record-level market-wide flags
    # --------------------------------------------------------

    record_level_audit = (
        create_record_level_market_flags(
            analysis_df
        )
    )

    # --------------------------------------------------------
    # Record-level property-type flags
    # --------------------------------------------------------

    record_level_audit = (
        create_record_level_property_type_flags(
            record_level_audit,
            property_type_iqr
        )
    )

    # --------------------------------------------------------
    # Compare market-wide vs property-type classifications
    # --------------------------------------------------------

    (
        record_level_audit,
        comparison_summary
    ) = create_outlier_comparison(
        record_level_audit
    )

    # --------------------------------------------------------
    # Multiple market-wide IQR flags
    # --------------------------------------------------------

    intersection_summary = (
        create_outlier_intersection_summary(
            record_level_audit
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
        property_type_iqr,
        comparison_summary,
        intersection_summary
    )

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    save_reports(
        descriptive_statistics,
        iqr_summary,
        zscore_summary,
        property_type_iqr,
        property_type_summary,
        comparison_summary,
        intersection_summary,
        classification_summary,
        record_level_audit,
        extreme_tables
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

    print(
        "\nProperty-type-specific analysis used a "
        f"minimum sample size of {MIN_GROUP_SIZE}."
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()


