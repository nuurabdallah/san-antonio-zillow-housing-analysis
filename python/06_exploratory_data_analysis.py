# ============================================================
# PART 6 — EXPLORATORY DATA ANALYSIS
# San Antonio Zillow Housing Analysis
# ============================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "San_Antonio_Zillow_Feature_Engineered.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "eda"
)

CHART_DIR = (
    OUTPUT_DIR
    / "charts"
)


# ============================================================
# ANALYSIS VARIABLES
# ============================================================

NUMERIC_VARIABLES = [
    "price",
    "logPrice",
    "area",
    "beds",
    "baths",
    "lotAreaSqFt",
    "taxAssessedValue",
    "daysOnZillow",
    "pricePerSqFt",
    "zestimate",
    "zestimate_gap",
    "zestimate_gap_pct",
]


CORRELATION_VARIABLES = [
    "price",
    "logPrice",
    "area",
    "beds",
    "baths",
    "lotAreaSqFt",
    "taxAssessedValue",
    "daysOnZillow",
    "pricePerSqFt",
    "zestimate",
]


REQUIRED_COLUMNS = [
    "zpid",
    "zipcode",
    "price",
    "logPrice",
    "area",
    "beds",
    "baths",
    "homeType",
    "daysOnZillow",
    "lotAreaSqFt",
    "taxAssessedValue",
    "pricePerSqFt",
    "zestimate",
    "zestimate_gap",
    "zestimate_gap_pct",
    "deal_score",
    "opportunity_status",
    "price_segment",
    "zip_listing_count",
    "zip_sample_status",
]


PRICE_SEGMENT_ORDER = [
    "Under $150K",
    "$150K-$300K",
    "$300K-$500K",
    "$500K-$1M",
    "$1M-$2M",
    "$2M+",
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("=" * 70)
    print("PART 6 — EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    print("\nSAN ANTONIO ZILLOW EXPLORATORY DATA ANALYSIS")

    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Input file: {INPUT_FILE}")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "Feature-engineered dataset not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"\nRows loaded: {len(df)}")
    print(f"Columns loaded: {len(df.columns)}")

    if len(df) != 810:

        raise ValueError(
            f"Expected 810 rows, found {len(df)}."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Required analytical columns are missing:\n"
            + "\n".join(missing_columns)
        )

    print("\nRequired analytical columns validated.")

    return df


# ============================================================
# DATASET INTEGRITY
# ============================================================

def validate_dataset_integrity(df):

    print("\n" + "=" * 70)
    print("DATASET INTEGRITY VALIDATION")
    print("=" * 70)

    unique_zpid = df["zpid"].nunique()
    duplicate_zpid = df["zpid"].duplicated().sum()

    print(f"Rows: {len(df)}")
    print(f"Unique zpid: {unique_zpid}")
    print(f"Duplicate zpid: {duplicate_zpid}")

    if len(df) != 810:
        raise ValueError("Dataset row count changed.")

    if unique_zpid != 810:
        raise ValueError(
            "Expected 810 unique properties."
        )

    if duplicate_zpid != 0:
        raise ValueError(
            "Duplicate zpid values detected."
        )

    print("\nDataset integrity validation passed.")


# ============================================================
# MARKET OVERVIEW
# ============================================================

def create_market_overview(df):

    overview = pd.DataFrame({
        "metric": [
            "Total Listings",
            "Unique Properties",
            "Mean Listing Price",
            "Median Listing Price",
            "Minimum Listing Price",
            "Maximum Listing Price",
            "Mean Living Area",
            "Median Living Area",
            "Mean Price Per Sq Ft",
            "Median Price Per Sq Ft",
            "Mean Days on Zillow",
            "Median Days on Zillow",
            "Mean Bedrooms",
            "Median Bedrooms",
            "Mean Bathrooms",
            "Median Bathrooms",
            "Zestimate Coverage",
            "Tax Assessed Value Coverage",
        ],

        "value": [
            len(df),
            df["zpid"].nunique(),
            df["price"].mean(),
            df["price"].median(),
            df["price"].min(),
            df["price"].max(),
            df["area"].mean(),
            df["area"].median(),
            df["pricePerSqFt"].mean(),
            df["pricePerSqFt"].median(),
            df["daysOnZillow"].mean(),
            df["daysOnZillow"].median(),
            df["beds"].mean(),
            df["beds"].median(),
            df["baths"].mean(),
            df["baths"].median(),
            df["zestimate"].notna().mean(),
            df["taxAssessedValue"].notna().mean(),
        ],
    })

    return overview


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

def create_descriptive_statistics(df):

    available_variables = [
        column
        for column in NUMERIC_VARIABLES
        if column in df.columns
    ]

    stats = (
        df[available_variables]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        )
        .T
    )

    stats["missing"] = (
        df[available_variables]
        .isna()
        .sum()
    )

    stats["missing_percentage"] = (
        stats["missing"]
        / len(df)
        * 100
    )

    stats = stats.reset_index()

    stats = stats.rename(
        columns={
            "index": "variable"
        }
    )

    return stats


# ============================================================
# PRICE SEGMENT ANALYSIS
# ============================================================

def create_price_segment_analysis(df):

    result = (
        df.groupby(
            "price_segment",
            dropna=False
        )
        .agg(
            listing_count=("zpid", "count"),
            mean_price=("price", "mean"),
            median_price=("price", "median"),
            mean_area=("area", "mean"),
            median_area=("area", "median"),
            mean_price_per_sqft=(
                "pricePerSqFt",
                "mean"
            ),
            median_price_per_sqft=(
                "pricePerSqFt",
                "median"
            ),
            mean_days_on_zillow=(
                "daysOnZillow",
                "mean"
            ),
            median_days_on_zillow=(
                "daysOnZillow",
                "median"
            ),
        )
        .reset_index()
    )

    result["sort_order"] = (
        result["price_segment"]
        .map({
            "Under $150K": 1,
            "$150K-$300K": 2,
            "$300K-$500K": 3,
            "$500K-$1M": 4,
            "$1M-$2M": 5,
            "$2M+": 6,
        })
    )

    result = (
        result
        .sort_values("sort_order")
        .drop(columns="sort_order")
    )

    result["percentage_of_listings"] = (
        result["listing_count"]
        / len(df)
        * 100
    )

    return result


# ============================================================
# PROPERTY TYPE ANALYSIS
# ============================================================

def create_property_type_analysis(df):

    result = (
        df.groupby(
            "homeType",
            dropna=False
        )
        .agg(
            listing_count=("zpid", "count"),
            mean_price=("price", "mean"),
            median_price=("price", "median"),
            mean_area=("area", "mean"),
            median_area=("area", "median"),
            mean_price_per_sqft=(
                "pricePerSqFt",
                "mean"
            ),
            median_price_per_sqft=(
                "pricePerSqFt",
                "median"
            ),
        )
        .reset_index()
    )

    result["percentage_of_listings"] = (
        result["listing_count"]
        / len(df)
        * 100
    )

    return result.sort_values(
        "listing_count",
        ascending=False
    )


# ============================================================
# ZIP ANALYSIS
# ============================================================

def create_zip_analysis(df):

    result = (
        df.groupby("zipcode")
        .agg(
            listing_count=("zpid", "count"),
            mean_price=("price", "mean"),
            median_price=("price", "median"),
            std_price=("price", "std"),
            mean_log_price=("logPrice", "mean"),
            median_log_price=("logPrice", "median"),
            median_area=("area", "median"),
            median_price_per_sqft=(
                "pricePerSqFt",
                "median"
            ),
            mean_price_per_sqft=(
                "pricePerSqFt",
                "mean"
            ),
        )
        .reset_index()
    )

    result["sample_status"] = np.where(
        result["listing_count"] >= 10,
        "Sufficient Sample",
        "Sparse Sample"
    )

    return result.sort_values(
        "median_price",
        ascending=False
    )


# ============================================================
# ZIP × PROPERTY TYPE COMPOSITION
# ============================================================

def create_zip_property_type_analysis(df):

    composition = (
        df.groupby(
            ["zipcode", "homeType"],
            dropna=False
        )
        .size()
        .reset_index(
            name="listing_count"
        )
    )

    zip_totals = (
        df.groupby("zipcode")
        .size()
        .reset_index(
            name="zip_total_listings"
        )
    )

    composition = composition.merge(
        zip_totals,
        on="zipcode",
        how="left"
    )

    composition["percentage_of_zip"] = (
        composition["listing_count"]
        / composition["zip_total_listings"]
        * 100
    )

    return composition.sort_values(
        [
            "zipcode",
            "listing_count"
        ],
        ascending=[
            True,
            False
        ]
    )


# ============================================================
# ZESTIMATE ANALYSIS
# ============================================================

def create_zestimate_analysis(df):

    z_df = df[
        df["zestimate"].notna()
    ].copy()

    if z_df.empty:

        return pd.DataFrame()

    result = pd.DataFrame({
        "metric": [
            "Listings with Zestimate",
            "Listings without Zestimate",
            "Zestimate Coverage",
            "Mean Zestimate",
            "Median Zestimate",
            "Mean Zestimate Gap",
            "Median Zestimate Gap",
            "Mean Zestimate Gap %",
            "Median Zestimate Gap %",
            "Listings Above Zestimate",
            "Potential Opportunities",
        ],

        "value": [
            len(z_df),
            df["zestimate"].isna().sum(),
            len(z_df) / len(df),
            z_df["zestimate"].mean(),
            z_df["zestimate"].median(),
            z_df["zestimate_gap"].mean(),
            z_df["zestimate_gap"].median(),
            z_df["zestimate_gap_pct"].mean(),
            z_df["zestimate_gap_pct"].median(),
            (
                z_df["opportunity_status"]
                == "Above Zestimate"
            ).sum(),
            (
                z_df["opportunity_status"]
                == "Potential Opportunity"
            ).sum(),
        ],
    })

    return result


# ============================================================
# OPPORTUNITY ANALYSIS
# ============================================================

def create_opportunity_analysis(df):

    opportunities = df[
        df["opportunity_status"]
        == "Potential Opportunity"
    ].copy()

    if opportunities.empty:

        return pd.DataFrame()

    result = (
        opportunities
        .groupby("zipcode")
        .agg(
            opportunity_count=("zpid", "count"),
            median_gap=("zestimate_gap", "median"),
            mean_gap=("zestimate_gap", "mean"),
            median_gap_pct=(
                "zestimate_gap_pct",
                "median"
            ),
            mean_gap_pct=(
                "zestimate_gap_pct",
                "mean"
            ),
            total_gap=(
                "zestimate_gap",
                "sum"
            ),
        )
        .reset_index()
    )

    return result.sort_values(
        "opportunity_count",
        ascending=False
    )


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def create_correlation_analysis(df):

    available_variables = [
        column
        for column in CORRELATION_VARIABLES
        if column in df.columns
    ]

    pearson = (
        df[available_variables]
        .corr(method="pearson")
    )

    spearman = (
        df[available_variables]
        .corr(method="spearman")
    )

    pearson.to_csv(
        OUTPUT_DIR
        / "correlation_matrix_pearson.csv"
    )

    spearman.to_csv(
        OUTPUT_DIR
        / "correlation_matrix_spearman.csv"
    )

    return pearson, spearman


# ============================================================
# CORRELATION PAIR TABLE
# ============================================================

def create_correlation_pairs(df):

    available_variables = [
        column
        for column in CORRELATION_VARIABLES
        if column in df.columns
    ]

    correlation = (
        df[available_variables]
        .corr(method="pearson")
    )

    rows = []

    for i, variable_1 in enumerate(
        available_variables
    ):

        for j, variable_2 in enumerate(
            available_variables
        ):

            if j <= i:
                continue

            rows.append({
                "variable_1": variable_1,
                "variable_2": variable_2,
                "pearson_correlation": correlation.loc[
                    variable_1,
                    variable_2
                ],
            })

    result = pd.DataFrame(rows)

    result["absolute_correlation"] = (
        result["pearson_correlation"]
        .abs()
    )

    return result.sort_values(
        "absolute_correlation",
        ascending=False
    )


# ============================================================
# VISUALIZATION OUTPUT CLEANUP
# ============================================================

def clear_existing_charts():
    """Remove prior PNG charts so each EDA run produces one clean chart set."""

    CHART_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    removed = 0

    for chart_file in CHART_DIR.glob("*.png"):
        chart_file.unlink()
        removed += 1

    if removed:
        print(
            f"Cleared {removed} existing PNG chart(s) from:"
            f"\n  {CHART_DIR}"
        )
    else:
        print("No existing PNG charts to clear.")


# ============================================================
# VISUALIZATION HELPER
# ============================================================

def save_figure(filename):

    CHART_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.tight_layout()

    plt.savefig(
        CHART_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# CHART 1 — PRICE DISTRIBUTION
# ============================================================

def plot_price_distribution(df):

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["price"].dropna(),
        bins=40
    )

    plt.title(
        "San Antonio Zillow Listing Price Distribution"
    )

    plt.xlabel(
        "Listing Price ($)"
    )

    plt.ylabel(
        "Number of Listings"
    )

    plt.ticklabel_format(
        style="plain",
        axis="x"
    )

    save_figure(
        "01_listing_price_distribution.png"
    )


# ============================================================
# CHART 2 — LOG PRICE DISTRIBUTION
# ============================================================

def plot_log_price_distribution(df):

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["logPrice"].dropna(),
        bins=40
    )

    plt.title(
        "Distribution of Log Listing Price"
    )

    plt.xlabel(
        "Log Listing Price"
    )

    plt.ylabel(
        "Number of Listings"
    )

    save_figure(
        "02_log_price_distribution.png"
    )


# ============================================================
# CHART 3 — LIVING AREA DISTRIBUTION
# ============================================================

def plot_area_distribution(df):

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["area"].dropna(),
        bins=40
    )

    plt.title(
        "Distribution of Living Area"
    )

    plt.xlabel(
        "Living Area (Sq Ft)"
    )

    plt.ylabel(
        "Number of Listings"
    )

    save_figure(
        "03_living_area_distribution.png"
    )


# ============================================================
# CHART 4 — PRICE VS LIVING AREA
# ============================================================

def plot_price_vs_area(df):

    subset = df[
        df["area"].notna()
        & df["price"].notna()
    ]

    plt.figure(figsize=(10, 6))

    plt.scatter(
        subset["area"],
        subset["price"],
        alpha=0.5
    )

    plt.title(
        "Listing Price vs. Living Area"
    )

    plt.xlabel(
        "Living Area (Sq Ft)"
    )

    plt.ylabel(
        "Listing Price ($)"
    )

    plt.ticklabel_format(
        style="plain",
        axis="y"
    )

    save_figure(
        "04_price_vs_living_area.png"
    )


# ============================================================
# CHART 5 — PRICE BY BEDROOM COUNT
# ============================================================

def plot_price_vs_bedrooms(df):

    grouped = (
        df.groupby("beds")
        .agg(
            mean_price=("price", "mean"),
            listing_count=("zpid", "count")
        )
        .reset_index()
        .sort_values("beds")
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        grouped["beds"],
        grouped["mean_price"],
        marker="o"
    )

    plt.title(
        "Average Listing Price by Bedroom Count"
    )

    plt.xlabel(
        "Bedrooms"
    )

    plt.ylabel(
        "Average Listing Price ($)"
    )

    plt.xticks(
        grouped["beds"]
    )

    plt.ticklabel_format(
        style="plain",
        axis="y"
    )

    save_figure(
        "05_price_by_bedrooms.png"
    )


# ============================================================
# CHART 6 — PRICE BY BATHROOM COUNT
# ============================================================

def plot_price_vs_bathrooms(df):

    grouped = (
        df.groupby("baths")
        .agg(
            mean_price=("price", "mean"),
            listing_count=("zpid", "count")
        )
        .reset_index()
        .sort_values("baths")
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        grouped["baths"],
        grouped["mean_price"],
        marker="o"
    )

    plt.title(
        "Average Listing Price by Bathroom Count"
    )

    plt.xlabel(
        "Bathrooms"
    )

    plt.ylabel(
        "Average Listing Price ($)"
    )

    plt.xticks(
        grouped["baths"]
    )

    plt.ticklabel_format(
        style="plain",
        axis="y"
    )

    save_figure(
        "06_price_by_bathrooms.png"
    )


# ============================================================
# CHART 7 — PRICE VS TAX ASSESSED VALUE
# ============================================================

def plot_price_vs_tax_value(df):

    subset = df[
        df["taxAssessedValue"].notna()
        & df["price"].notna()
    ]

    plt.figure(figsize=(10, 6))

    plt.scatter(
        subset["taxAssessedValue"],
        subset["price"],
        alpha=0.5
    )

    plt.title(
        "Listing Price vs. Tax Assessed Value"
    )

    plt.xlabel(
        "Tax Assessed Value ($)"
    )

    plt.ylabel(
        "Listing Price ($)"
    )

    plt.ticklabel_format(
        style="plain",
        axis="both"
    )

    save_figure(
        "07_price_vs_tax_assessed_value.png"
    )


# ============================================================
# CHART 8 — PRICE VS PRICE PER SQ FT
# ============================================================

def plot_price_vs_ppsf(df):

    subset = df[
        df["pricePerSqFt"].notna()
        & df["price"].notna()
    ]

    plt.figure(figsize=(10, 6))

    plt.scatter(
        subset["pricePerSqFt"],
        subset["price"],
        alpha=0.5
    )

    plt.title(
        "Listing Price vs. Price per Square Foot"
    )

    plt.xlabel(
        "Price per Square Foot ($)"
    )

    plt.ylabel(
        "Listing Price ($)"
    )

    plt.ticklabel_format(
        style="plain",
        axis="both"
    )

    save_figure(
        "08_price_vs_price_per_sqft.png"
    )


# ============================================================
# CHART 9 — DAYS ON ZILLOW DISTRIBUTION
# ============================================================

def plot_days_on_zillow(df):

    subset = df[
        df["daysOnZillow"].notna()
    ]

    plt.figure(figsize=(10, 6))

    plt.hist(
        subset["daysOnZillow"],
        bins=30
    )

    plt.title(
        "Distribution of Days on Zillow"
    )

    plt.xlabel(
        "Days on Zillow"
    )

    plt.ylabel(
        "Number of Listings"
    )

    save_figure(
        "09_days_on_zillow_distribution.png"
    )


# ============================================================
# CHART 10 — ZESTIMATE GAP DISTRIBUTION
# ============================================================

def plot_zestimate_gap(df):

    subset = df[
        df["zestimate_gap"].notna()
    ]

    plt.figure(figsize=(10, 6))

    plt.hist(
        subset["zestimate_gap"],
        bins=40
    )

    plt.title(
        "Zestimate Gap Distribution"
    )

    plt.xlabel(
        "Zestimate - Listing Price ($)"
    )

    plt.ylabel(
        "Number of Listings"
    )

    plt.ticklabel_format(
        style="plain",
        axis="x"
    )

    save_figure(
        "10_zestimate_gap_distribution.png"
    )


# ============================================================
# CHART 11 — LISTING PRICE VS ZESTIMATE
# ============================================================

def plot_zestimate_vs_price(df):

    subset = df[
        df["zestimate"].notna()
        & df["price"].notna()
    ]

    plt.figure(figsize=(10, 6))

    plt.scatter(
        subset["price"],
        subset["zestimate"],
        alpha=0.5
    )

    min_value = min(
        subset["price"].min(),
        subset["zestimate"].min()
    )

    max_value = max(
        subset["price"].max(),
        subset["zestimate"].max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.title(
        "Listing Price vs. Zestimate"
    )

    plt.xlabel(
        "Listing Price ($)"
    )

    plt.ylabel(
        "Zestimate ($)"
    )

    plt.ticklabel_format(
        style="plain",
        axis="both"
    )

    save_figure(
        "11_listing_price_vs_zestimate.png"
    )


# ============================================================
# LOG-PRICE SCATTER PLOTS
# ============================================================

def plot_log_price_vs_area(df):

    subset = df[
        df["area"].notna()
        & df["logPrice"].notna()
    ]

    plt.figure(figsize=(10, 6))

    plt.scatter(
        subset["area"],
        subset["logPrice"],
        alpha=0.5
    )

    plt.title(
        "Log Listing Price vs. Living Area"
    )

    plt.xlabel(
        "Living Area (Sq Ft)"
    )

    plt.ylabel(
        "Log Listing Price"
    )

    save_figure(
        "12_log_price_vs_living_area.png"
    )


def plot_log_price_vs_tax_value(df):

    subset = df[
        df["taxAssessedValue"].notna()
        & df["logPrice"].notna()
    ]

    plt.figure(figsize=(10, 6))

    plt.scatter(
        subset["taxAssessedValue"],
        subset["logPrice"],
        alpha=0.5
    )

    plt.title(
        "Log Listing Price vs. Tax Assessed Value"
    )

    plt.xlabel(
        "Tax Assessed Value ($)"
    )

    plt.ylabel(
        "Log Listing Price"
    )

    plt.ticklabel_format(
        style="plain",
        axis="x"
    )

    save_figure(
        "13_log_price_vs_tax_assessed_value.png"
    )


def plot_log_price_vs_ppsf(df):

    subset = df[
        df["pricePerSqFt"].notna()
        & df["logPrice"].notna()
    ]

    plt.figure(figsize=(10, 6))

    plt.scatter(
        subset["pricePerSqFt"],
        subset["logPrice"],
        alpha=0.5
    )

    plt.title(
        "Log Listing Price vs. Price per Square Foot"
    )

    plt.xlabel(
        "Price per Square Foot ($)"
    )

    plt.ylabel(
        "Log Listing Price"
    )

    plt.ticklabel_format(
        style="plain",
        axis="x"
    )

    save_figure(
        "14_log_price_vs_price_per_sqft.png"
    )


def plot_log_price_vs_zestimate(df):

    subset = df[
        df["zestimate"].notna()
        & df["logPrice"].notna()
    ]

    plt.figure(figsize=(10, 6))

    plt.scatter(
        subset["zestimate"],
        subset["logPrice"],
        alpha=0.5
    )

    plt.title(
        "Log Listing Price vs. Zestimate"
    )

    plt.xlabel(
        "Zestimate ($)"
    )

    plt.ylabel(
        "Log Listing Price"
    )

    plt.ticklabel_format(
        style="plain",
        axis="x"
    )

    save_figure(
        "15_log_price_vs_zestimate.png"
    )


# ============================================================
# CHART 20 — PROPERTY TYPE DISTRIBUTION
# ============================================================

def plot_property_type_distribution(df):

    counts = (
        df["homeType"]
        .fillna("Unknown")
        .value_counts()
        .sort_values()
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        counts.index.astype(str),
        counts.values
    )

    plt.title(
        "Listings by Property Type"
    )

    plt.xlabel(
        "Number of Listings"
    )

    plt.ylabel(
        "Property Type"
    )

    save_figure(
        "16_property_type_distribution.png"
    )


# ============================================================
# CHART 17 — PRICE SEGMENT DISTRIBUTION
# ============================================================

def plot_price_segment_distribution(df):

    counts = (
        df["price_segment"]
        .value_counts()
        .reindex(
            PRICE_SEGMENT_ORDER,
            fill_value=0
        )
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        counts.index,
        counts.values
    )

    plt.title(
        "Listings by Price Segment"
    )

    plt.xlabel(
        "Price Segment"
    )

    plt.ylabel(
        "Number of Listings"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    save_figure(
        "17_price_segment_distribution.png"
    )


# ============================================================
# CHART 18 — TOP ZIP CODES BY MEDIAN PRICE
# ============================================================

def plot_top_zip_prices(df):

    zip_df = (
        df[
            df["zip_sample_status"]
            == "Sufficient Sample"
        ]
        .groupby("zipcode")
        .agg(
            median_price=("price", "median")
        )
        .sort_values(
            "median_price",
            ascending=False
        )
        .head(10)
        .sort_values(
            "median_price"
        )
    )

    plt.figure(figsize=(10, 6))

    y_positions = np.arange(len(zip_df))

    plt.scatter(
        zip_df["median_price"],
        y_positions,
        s=70
    )

    plt.yticks(
        y_positions,
        zip_df.index.astype(str)
    )

    for y, value in zip(y_positions, zip_df["median_price"]):
        plt.plot(
            [0, value],
            [y, y],
            linewidth=1
        )

    plt.title(
        "Top 10 ZIP Codes by Median Listing Price"
    )

    plt.xlabel(
        "Median Listing Price ($)"
    )

    plt.ylabel(
        "ZIP Code"
    )

    plt.ticklabel_format(
        style="plain",
        axis="x"
    )

    save_figure(
        "18_top_zip_median_price.png"
    )


# ============================================================
# CHART 19 — TOP ZIP CODES BY MEDIAN PRICE/SQ FT
# ============================================================

def plot_top_zip_ppsf(df):

    zip_df = (
        df[
            df["zip_sample_status"]
            == "Sufficient Sample"
        ]
        .groupby("zipcode")
        .agg(
            median_ppsf=(
                "pricePerSqFt",
                "median"
            )
        )
        .sort_values(
            "median_ppsf",
            ascending=False
        )
        .head(10)
        .sort_values(
            "median_ppsf"
        )
    )

    plt.figure(figsize=(10, 6))

    y_positions = np.arange(len(zip_df))

    plt.scatter(
        zip_df["median_ppsf"],
        y_positions,
        s=70
    )

    plt.yticks(
        y_positions,
        zip_df.index.astype(str)
    )

    for y, value in zip(y_positions, zip_df["median_ppsf"]):
        plt.plot(
            [0, value],
            [y, y],
            linewidth=1
        )

    plt.title(
        "Top 10 ZIP Codes by Median Price per Square Foot"
    )

    plt.xlabel(
        "Median Price per Square Foot ($)"
    )

    plt.ylabel(
        "ZIP Code"
    )

    save_figure(
        "19_top_zip_median_price_per_sqft.png"
    )


# ============================================================
# CHART 16 — CORRELATION HEATMAP
# ============================================================

def plot_correlation_heatmap(df):

    available_variables = [
        column
        for column in CORRELATION_VARIABLES
        if column in df.columns
    ]

    correlation = (
        df[available_variables]
        .corr(method="pearson")
    )

    plt.figure(figsize=(12, 10))

    plt.imshow(
        correlation,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1
    )

    for i in range(len(correlation.index)):
        for j in range(len(correlation.columns)):
            plt.text(
                j,
                i,
                f"{correlation.iloc[i, j]:.2f}",
                ha="center",
                va="center"
            )

    plt.colorbar(
        label="Pearson Correlation"
    )

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    plt.title(
        "Pearson Correlation Matrix"
    )

    save_figure(
        "20_pearson_correlation_heatmap.png"
    )


# ============================================================
# KEY FINDINGS
# ============================================================

def create_key_findings(
    df,
    price_segments,
    property_types,
    zip_analysis,
    correlation_pairs
):

    sufficient_zip = zip_analysis[
        zip_analysis["sample_status"]
        == "Sufficient Sample"
    ]

    if sufficient_zip.empty:

        top_zip = None
        lowest_zip = None

    else:

        top_zip = (
            sufficient_zip
            .sort_values(
                "median_price",
                ascending=False
            )
            .iloc[0]
        )

        lowest_zip = (
            sufficient_zip
            .sort_values(
                "median_price",
                ascending=True
            )
            .iloc[0]
        )

    largest_property_type = (
        property_types
        .sort_values(
            "listing_count",
            ascending=False
        )
        .iloc[0]
    )

    largest_price_segment = (
        price_segments
        .sort_values(
            "listing_count",
            ascending=False
        )
        .iloc[0]
    )

    strongest_pair = (
        correlation_pairs.iloc[0]
    )

    opportunity_count = (
        df["opportunity_status"]
        == "Potential Opportunity"
    ).sum()

    if top_zip is not None:

        zip_finding = (
            f'The highest-median-price sufficient-sample '
            f'ZIP is {top_zip["zipcode"]}, with a median '
            f'listing price of ${top_zip["median_price"]:,.0f}.'
        )

        lowest_zip_finding = (
            f'The lowest-median-price sufficient-sample '
            f'ZIP is {lowest_zip["zipcode"]}, with a median '
            f'listing price of ${lowest_zip["median_price"]:,.0f}.'
        )

    else:

        zip_finding = (
            "No sufficient-sample ZIP codes were available."
        )

        lowest_zip_finding = (
            "No sufficient-sample ZIP codes were available."
        )

    findings = f"""# Part 6 — Exploratory Data Analysis Findings

## Dataset

The feature-engineered analytical dataset contains **810 unique
San Antonio Zillow property listings**.

No observations were removed during exploratory analysis.

## Market Overview

- Median listing price: ${df["price"].median():,.0f}
- Mean listing price: ${df["price"].mean():,.0f}
- Median living area: {df["area"].median():,.0f} sq ft
- Mean living area: {df["area"].mean():,.0f} sq ft
- Median price per square foot: ${df["pricePerSqFt"].median():,.2f}
- Median days on Zillow: {df["daysOnZillow"].median():.0f} days

## Price Segmentation

The largest price segment is **{largest_price_segment["price_segment"]}**,
containing {int(largest_price_segment["listing_count"]):,} listings
({largest_price_segment["percentage_of_listings"]:.2f}% of the dataset).

## Property Types

The most represented property type is **{largest_property_type["homeType"]}**,
with {int(largest_property_type["listing_count"]):,} listings
({largest_property_type["percentage_of_listings"]:.2f}% of the dataset).

## ZIP Code Analysis

ZIP-level comparisons use the established minimum sample size of
10 listings.

{zip_finding}

{lowest_zip_finding}

## Zestimate Analysis

Zestimate information is available for
{df["zestimate"].notna().sum():,} listings
({df["zestimate"].notna().mean() * 100:.2f}% coverage).

There are {int(opportunity_count):,} listings where Zestimate exceeds
listing price and therefore meet the project's definition of a
Potential Opportunity.

## Correlation Analysis

The strongest absolute Pearson correlation among the analyzed variables
is between **{strongest_pair["variable_1"]}** and
**{strongest_pair["variable_2"]}**, with a correlation of
{strongest_pair["pearson_correlation"]:.4f}.

Correlation measures describe association and do not establish
causation.

## Analytical Considerations

Several variables are strongly related to listing price, but these
relationships should not automatically be interpreted as causal.

ZIP-level comparisons should account for sample size and
property-type composition.

Zestimate-based opportunity analysis is limited to listings with
available Zestimate information.

The exploratory analysis provides the foundation for subsequent
statistical modeling, machine learning, visualization, and business
analysis.
"""

    return findings


# ============================================================
# METHODOLOGY
# ============================================================

def create_methodology_notes():

    return """# Part 6 — Exploratory Data Analysis Methodology

## Purpose

Part 6 examines the structure, distributions, relationships, and
business characteristics of the finalized feature-engineered San
Antonio Zillow dataset.

## Source

The analysis uses:

`data/processed/San_Antonio_Zillow_Feature_Engineered.csv`

The analytical population contains 810 unique properties.

## Descriptive Analysis

Descriptive statistics include:

- Count
- Mean
- Standard deviation
- Minimum
- 1st percentile
- 5th percentile
- 25th percentile
- Median
- 75th percentile
- 95th percentile
- 99th percentile
- Maximum
- Missing-value counts

## Correlation Analysis

Both Pearson and Spearman correlations are calculated.

Pearson correlation measures linear association.

Spearman correlation measures monotonic association based on ranks.

Correlation does not establish causation.

## ZIP Analysis

ZIP-level comparisons use the established project rule:

- 10 or more listings = Sufficient Sample
- Fewer than 10 listings = Sparse Sample

Sparse ZIP codes remain in the dataset but should not be treated as
standalone rankings without appropriate caution.

## Property-Type Composition

Property-type composition is examined because ZIP-level price
differences can reflect differences in the types of properties
represented within each ZIP.

## Zestimate Analysis

Zestimate analysis is restricted to properties with non-missing
Zestimate values.

Zestimate gap is:

`Zestimate - Listing Price`

Zestimate gap percentage is:

`(Zestimate - Listing Price) / Listing Price`

## Visualization

Twenty exploratory charts are generated with Matplotlib and saved to:

`reports/eda/charts/`

## Machine Learning Separation

EDA includes price-derived business metrics for descriptive purposes,
but these variables are not treated as machine-learning predictors when
they directly contain or derive from the target listing price.

The finalized ML predictor set remains:

- area
- beds
- baths
- lotAreaSqFt
- taxAssessedValue
- daysOnZillow

The ML target remains:

- logPrice
"""


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    market_overview,
    descriptive_stats,
    price_segments,
    property_types,
    zip_analysis,
    zip_property_type,
    zestimate_summary,
    opportunity_analysis,
    correlation_pairs,
    findings
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CHART_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    market_overview.to_csv(
        OUTPUT_DIR
        / "market_overview.csv",
        index=False
    )

    descriptive_stats.to_csv(
        OUTPUT_DIR
        / "descriptive_statistics.csv",
        index=False
    )

    price_segments.to_csv(
        OUTPUT_DIR
        / "price_segment_analysis.csv",
        index=False
    )

    property_types.to_csv(
        OUTPUT_DIR
        / "property_type_analysis.csv",
        index=False
    )

    zip_analysis.to_csv(
        OUTPUT_DIR
        / "zip_analysis.csv",
        index=False
    )

    zip_property_type.to_csv(
        OUTPUT_DIR
        / "zip_property_type_composition.csv",
        index=False
    )

    if not zestimate_summary.empty:

        zestimate_summary.to_csv(
            OUTPUT_DIR
            / "zestimate_analysis.csv",
            index=False
        )

    if not opportunity_analysis.empty:

        opportunity_analysis.to_csv(
            OUTPUT_DIR
            / "opportunity_analysis.csv",
            index=False
        )

    correlation_pairs.to_csv(
        OUTPUT_DIR
        / "correlation_pairs.csv",
        index=False
    )

    methodology_file = (
        OUTPUT_DIR
        / "EDA_METHODOLOGY.md"
    )

    methodology_file.write_text(
        create_methodology_notes(),
        encoding="utf-8"
    )

    findings_file = (
        OUTPUT_DIR
        / "EDA_KEY_FINDINGS.md"
    )

    findings_file.write_text(
        findings,
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("EDA OUTPUTS SAVED")
    print("=" * 70)

    print("\nOutput directory:")
    print(f"  {OUTPUT_DIR}")

    print("\nAnalysis reports:")
    print("  - market_overview.csv")
    print("  - descriptive_statistics.csv")
    print("  - price_segment_analysis.csv")
    print("  - property_type_analysis.csv")
    print("  - zip_analysis.csv")
    print("  - zip_property_type_composition.csv")
    print("  - zestimate_analysis.csv")
    print("  - opportunity_analysis.csv")
    print("  - correlation_matrix_pearson.csv")
    print("  - correlation_matrix_spearman.csv")
    print("  - correlation_pairs.csv")
    print("  - EDA_METHODOLOGY.md")
    print("  - EDA_KEY_FINDINGS.md")

    print("\nCharts:")
    print("  - 20 PNG visualization files")


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    df,
    price_segments,
    property_types,
    zip_analysis
):

    sufficient_zip_count = (
        zip_analysis[
            zip_analysis["sample_status"]
            == "Sufficient Sample"
        ]
        .shape[0]
    )

    sparse_zip_count = (
        zip_analysis[
            zip_analysis["sample_status"]
            == "Sparse Sample"
        ]
        .shape[0]
    )

    opportunity_count = int(
        (
            df["opportunity_status"]
            == "Potential Opportunity"
        ).sum()
    )

    print("\n" + "=" * 70)
    print("PART 6 EXPLORATORY DATA ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"\nFinal analytical records: "
        f"{len(df)}"
    )

    print(
        f"Unique properties: "
        f"{df['zpid'].nunique()}"
    )

    print(
        f"Median listing price: "
        f"${df['price'].median():,.0f}"
    )

    print(
        f"Mean listing price: "
        f"${df['price'].mean():,.0f}"
    )

    print(
        f"Median price per sq ft: "
        f"${df['pricePerSqFt'].median():,.2f}"
    )

    print(
        f"Median days on Zillow: "
        f"{df['daysOnZillow'].median():.0f}"
    )

    print(
        f"Zestimate coverage: "
        f"{df['zestimate'].notna().mean() * 100:.2f}%"
    )

    print(
        f"Potential opportunities: "
        f"{opportunity_count}"
    )

    print(
        f"\nSufficient-sample ZIP codes: "
        f"{sufficient_zip_count}"
    )

    print(
        f"Sparse ZIP codes: "
        f"{sparse_zip_count}"
    )

    print(
        f"\nProperty types analyzed: "
        f"{len(property_types)}"
    )

    print(
        f"Price segments analyzed: "
        f"{len(price_segments)}"
    )

    print(
        "\nThe feature-engineered dataset was not modified."
    )

    print(
        "EDA outputs were saved for subsequent analysis."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    validate_dataset_integrity(df)

    print("\nCreating market overview...")
    market_overview = (
        create_market_overview(df)
    )

    print("Creating descriptive statistics...")
    descriptive_stats = (
        create_descriptive_statistics(df)
    )

    print("Creating price segment analysis...")
    price_segments = (
        create_price_segment_analysis(df)
    )

    print("Creating property type analysis...")
    property_types = (
        create_property_type_analysis(df)
    )

    print("Creating ZIP analysis...")
    zip_analysis = (
        create_zip_analysis(df)
    )

    print(
        "Creating ZIP/property-type composition..."
    )

    zip_property_type = (
        create_zip_property_type_analysis(df)
    )

    print("Creating Zestimate analysis...")
    zestimate_summary = (
        create_zestimate_analysis(df)
    )

    print("Creating opportunity analysis...")
    opportunity_analysis = (
        create_opportunity_analysis(df)
    )

    print("Creating correlation analysis...")

    pearson, spearman = (
        create_correlation_analysis(df)
    )

    print(
        "Creating correlation pair analysis..."
    )

    correlation_pairs = (
        create_correlation_pairs(df)
    )

    print("\nPreparing visualization output directory...")
    clear_existing_charts()

    print("\nGenerating visualizations...")

    plot_price_distribution(df)

    plot_log_price_distribution(df)

    plot_area_distribution(df)

    plot_price_vs_area(df)

    plot_price_vs_bedrooms(df)

    plot_price_vs_bathrooms(df)

    plot_price_vs_tax_value(df)

    plot_price_vs_ppsf(df)

    plot_days_on_zillow(df)

    plot_zestimate_gap(df)

    plot_zestimate_vs_price(df)

    plot_log_price_vs_area(df)

    plot_log_price_vs_tax_value(df)

    plot_log_price_vs_ppsf(df)

    plot_log_price_vs_zestimate(df)

    plot_property_type_distribution(df)

    plot_price_segment_distribution(df)

    plot_top_zip_prices(df)

    plot_top_zip_ppsf(df)

    plot_correlation_heatmap(df)

    chart_count = len(list(CHART_DIR.glob("*.png")))

    if chart_count != 20:
        raise ValueError(
            f"Expected 20 EDA charts, found {chart_count}."
        )

    print(
        f"\nGenerated {chart_count} EDA charts."
    )

    findings = create_key_findings(
        df,
        price_segments,
        property_types,
        zip_analysis,
        correlation_pairs
    )

    save_outputs(
        market_overview,
        descriptive_stats,
        price_segments,
        property_types,
        zip_analysis,
        zip_property_type,
        zestimate_summary,
        opportunity_analysis,
        correlation_pairs,
        findings
    )

    print_final_summary(
        df,
        price_segments,
        property_types,
        zip_analysis
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()