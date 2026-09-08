"""
Part 8 — Visualization
San Antonio Zillow Housing Analysis

Purpose:
    Create a reproducible Python visualization layer that translates the
    EDA and statistical findings into publication-ready analytical charts.

Input:
    data/processed/San_Antonio_Zillow_Feature_Engineered.csv

Outputs:
    reports/visualization/charts/
    reports/visualization/VISUALIZATION_METHODOLOGY.md
    reports/visualization/VISUALIZATION_KEY_FINDINGS.md

Important:
    - The analytical dataset is never modified.
    - The full 810-property analytical population is preserved.
    - Missing Zestimate and tax-assessed values remain missing.
    - Charts are based on the finalized feature-engineered dataset.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "San_Antonio_Zillow_Feature_Engineered.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "visualization"
CHART_DIR = OUTPUT_DIR / "charts"

ALPHA = 0.05
EXPECTED_ROWS = 810
EXPECTED_UNIQUE_ZPID = 810


# ============================================================================
# REQUIRED COLUMNS
# ============================================================================

REQUIRED_COLUMNS = [
    "zpid",
    "zipcode",
    "price",
    "logPrice",
    "area",
    "beds",
    "baths",
    "homeType",
    "lotAreaSqFt",
    "taxAssessedValue",
    "daysOnZillow",
    "zestimate",
    "pricePerSqFt",
    "zestimate_gap",
    "zestimate_gap_pct",
    "opportunity_status",
]


# ============================================================================
# HELPERS
# ============================================================================

def validate_dataset(df):
    """Validate the expected analytical population and key identifiers."""

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    duplicate_zpid = df["zpid"].duplicated().sum()

    print("\n" + "=" * 70)
    print("DATASET INTEGRITY VALIDATION")
    print("=" * 70)
    print(f"Rows: {len(df):,}")
    print(f"Unique zpid: {df['zpid'].nunique():,}")
    print(f"Duplicate zpid: {duplicate_zpid:,}")

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} analytical records, "
            f"but found {len(df):,}."
        )

    if df["zpid"].nunique() != EXPECTED_UNIQUE_ZPID:
        raise ValueError(
            f"Expected {EXPECTED_UNIQUE_ZPID:,} unique zpids, "
            f"but found {df['zpid'].nunique():,}."
        )

    if duplicate_zpid != 0:
        raise ValueError("Duplicate zpid values detected.")

    print("\nDataset integrity validation passed.")


def clear_old_charts():
    """Remove previously generated PNG charts to prevent stale outputs."""

    CHART_DIR.mkdir(parents=True, exist_ok=True)

    old_files = list(CHART_DIR.glob("*.png"))

    for file_path in old_files:
        file_path.unlink()

    print(f"\nCleared {len(old_files):,} existing PNG chart(s).")


def save_current_figure(filename):
    """Save the current matplotlib figure and close it."""

    output_path = CHART_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return output_path


def clean_numeric(df, columns):
    """Convert selected columns to numeric without modifying the source data."""

    result = df.copy()

    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    return result


def format_currency(value):
    """Format a numeric value as whole-dollar currency."""

    if pd.isna(value):
        return "N/A"

    return f"${value:,.0f}"


def format_percent(value):
    """Format a decimal as a percentage."""

    if pd.isna(value):
        return "N/A"

    return f"{value * 100:.2f}%"


def add_regression_line(x, y):
    """Add a simple least-squares regression line when enough data exists."""

    valid = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(valid) < 3:
        return

    coefficients = np.polyfit(valid["x"], valid["y"], 1)
    x_values = np.linspace(valid["x"].min(), valid["x"].max(), 100)
    y_values = coefficients[0] * x_values + coefficients[1]

    plt.plot(x_values, y_values, linewidth=2)


def correlation_matrix_plot(df):
    """Create a correlation matrix with numeric annotations."""

    columns = [
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
    ]

    corr = df[columns].corr()

    fig_width = max(10, len(columns) * 0.8)
    fig_height = max(8, len(columns) * 0.7)

    plt.figure(figsize=(fig_width, fig_height))

    plt.imshow(
        corr,
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    plt.colorbar(label="Correlation")

    plt.xticks(
        range(len(columns)),
        columns,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(columns)),
        columns,
    )

    for row in range(len(columns)):
        for col in range(len(columns)):
            value = corr.iloc[row, col]

            if pd.notna(value):
                plt.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    plt.title("Correlation Matrix of Key Housing Variables")
    plt.xlabel("Variables")
    plt.ylabel("Variables")

    return save_current_figure("08_correlation_matrix.png")


# ============================================================================
# CHART 1 — LISTING PRICE DISTRIBUTION
# ============================================================================

def chart_01_listing_price_distribution(df):
    data = df["price"].dropna()

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=40)
    plt.axvline(
        data.median(),
        linestyle="--",
        linewidth=2,
        label=f"Median: {format_currency(data.median())}",
    )
    plt.axvline(
        data.mean(),
        linestyle=":",
        linewidth=2,
        label=f"Mean: {format_currency(data.mean())}",
    )

    plt.title("San Antonio Listing Price Distribution")
    plt.xlabel("Listing Price ($)")
    plt.ylabel("Number of Properties")
    plt.legend()

    return save_current_figure("01_listing_price_distribution.png")


# ============================================================================
# CHART 2 — LOG PRICE DISTRIBUTION
# ============================================================================

def chart_02_log_price_distribution(df):
    data = df["logPrice"].dropna()

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=40)

    plt.title("Distribution of Log-Transformed Listing Price")
    plt.xlabel("logPrice")
    plt.ylabel("Number of Properties")

    return save_current_figure("02_log_price_distribution.png")


# ============================================================================
# CHART 3 — AREA VS LOG PRICE
# ============================================================================

def chart_03_area_vs_log_price(df):
    data = df[["area", "logPrice"]].dropna()

    plt.figure(figsize=(10, 6))
    plt.scatter(data["area"], data["logPrice"], alpha=0.55, s=22)
    add_regression_line(data["area"], data["logPrice"])

    plt.title("Living Area vs. Log Listing Price")
    plt.xlabel("Living Area (sq ft)")
    plt.ylabel("Log Listing Price")

    return save_current_figure("03_area_vs_log_price.png")


# ============================================================================
# CHART 4 — BEDROOMS VS LOG PRICE
# ============================================================================

def chart_04_bedrooms_vs_log_price(df):
    data = df[["beds", "logPrice"]].dropna()

    grouped = (
        data.groupby("beds", as_index=False)["logPrice"]
        .agg(["mean", "count"])
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    plt.scatter(
        data["beds"],
        data["logPrice"],
        alpha=0.30,
        s=18,
    )

    plt.plot(
        grouped["beds"],
        grouped["mean"],
        marker="o",
        linewidth=2,
        label="Mean logPrice",
    )

    plt.title("Bedrooms vs. Log Listing Price")
    plt.xlabel("Bedrooms")
    plt.ylabel("Log Listing Price")
    plt.legend()

    return save_current_figure("04_bedrooms_vs_log_price.png")


# ============================================================================
# CHART 5 — BATHROOMS VS LOG PRICE
# ============================================================================

def chart_05_bathrooms_vs_log_price(df):
    data = df[["baths", "logPrice"]].dropna()

    grouped = (
        data.groupby("baths", as_index=False)["logPrice"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("baths")
    )

    plt.figure(figsize=(10, 6))
    plt.scatter(
        data["baths"],
        data["logPrice"],
        alpha=0.30,
        s=18,
    )

    plt.plot(
        grouped["baths"],
        grouped["mean"],
        marker="o",
        linewidth=2,
        label="Mean logPrice",
    )

    plt.title("Bathrooms vs. Log Listing Price")
    plt.xlabel("Bathrooms")
    plt.ylabel("Log Listing Price")
    plt.legend()

    return save_current_figure("05_bathrooms_vs_log_price.png")


# ============================================================================
# CHART 6 — LOT AREA VS LOG PRICE
# ============================================================================

def chart_06_lot_area_vs_log_price(df):
    data = df[["lotAreaSqFt", "logPrice"]].dropna()

    plt.figure(figsize=(10, 6))
    plt.scatter(
        data["lotAreaSqFt"],
        data["logPrice"],
        alpha=0.50,
        s=20,
    )
    add_regression_line(data["lotAreaSqFt"], data["logPrice"])

    plt.title("Lot Area vs. Log Listing Price")
    plt.xlabel("Lot Area (sq ft)")
    plt.ylabel("Log Listing Price")

    return save_current_figure("06_lot_area_vs_log_price.png")


# ============================================================================
# CHART 7 — TAX ASSESSED VALUE VS LOG PRICE
# ============================================================================

def chart_07_tax_assessed_vs_log_price(df):
    data = df[["taxAssessedValue", "logPrice"]].dropna()

    plt.figure(figsize=(10, 6))
    plt.scatter(
        data["taxAssessedValue"],
        data["logPrice"],
        alpha=0.55,
        s=22,
    )
    add_regression_line(
        data["taxAssessedValue"],
        data["logPrice"],
    )

    plt.title("Tax-Assessed Value vs. Log Listing Price")
    plt.xlabel("Tax-Assessed Value ($)")
    plt.ylabel("Log Listing Price")

    return save_current_figure(
        "07_tax_assessed_value_vs_log_price.png"
    )


# ============================================================================
# CHART 8 — CORRELATION MATRIX
# ============================================================================

def chart_08_correlation_matrix(df):
    return correlation_matrix_plot(df)


# ============================================================================
# CHART 9 — MEDIAN PRICE BY SUFFICIENT-SAMPLE ZIP
# ============================================================================

def chart_09_zip_median_price(df):
    zip_counts = df["zipcode"].value_counts()

    sufficient_zips = zip_counts[zip_counts >= 10].index

    data = df[df["zipcode"].isin(sufficient_zips)].copy()

    summary = (
        data.groupby("zipcode")["price"]
        .median()
        .sort_values()
    )

    plt.figure(figsize=(11, 10))

    y_positions = np.arange(len(summary))

    plt.barh(
        y_positions,
        summary.values,
    )

    plt.yticks(y_positions, summary.index.astype(str))
    plt.xlabel("Median Listing Price ($)")
    plt.ylabel("ZIP Code")
    plt.title("Median Listing Price by Sufficient-Sample ZIP Code")

    for position, value in zip(y_positions, summary.values):
        plt.text(
            value,
            position,
            f" ${value:,.0f}",
            va="center",
            fontsize=8,
        )

    return save_current_figure("09_zip_median_listing_price.png")


# ============================================================================
# CHART 10 — MEDIAN PRICE PER SQ FT BY ZIP
# ============================================================================

def chart_10_zip_median_price_per_sqft(df):
    zip_counts = df["zipcode"].value_counts()
    sufficient_zips = zip_counts[zip_counts >= 10].index

    data = df[df["zipcode"].isin(sufficient_zips)].copy()

    summary = (
        data.groupby("zipcode")["pricePerSqFt"]
        .median()
        .sort_values()
    )

    plt.figure(figsize=(11, 10))

    y_positions = np.arange(len(summary))

    plt.barh(
        y_positions,
        summary.values,
    )

    plt.yticks(y_positions, summary.index.astype(str))
    plt.xlabel("Median Price per Sq Ft ($)")
    plt.ylabel("ZIP Code")
    plt.title("Median Price per Sq Ft by Sufficient-Sample ZIP Code")

    for position, value in zip(y_positions, summary.values):
        plt.text(
            value,
            position,
            f" ${value:,.0f}",
            va="center",
            fontsize=8,
        )

    return save_current_figure("10_zip_median_price_per_sqft.png")


# ============================================================================
# CHART 11 — PROPERTY TYPE PRICE DISTRIBUTION
# ============================================================================

def chart_11_property_type_distribution(df):
    data = df[["homeType", "price"]].dropna()

    groups = []
    labels = []

    for home_type in sorted(data["homeType"].unique()):
        values = data.loc[
            data["homeType"] == home_type,
            "price",
        ].dropna()

        if len(values) > 0:
            groups.append(values)
            labels.append(home_type)

    plt.figure(figsize=(11, 7))
    plt.boxplot(
        groups,
        labels=labels,
        showfliers=False,
    )

    plt.title("Listing Price Distribution by Property Type")
    plt.xlabel("Property Type")
    plt.ylabel("Listing Price ($)")
    plt.xticks(rotation=25, ha="right")

    return save_current_figure("11_property_type_price_distribution.png")


# ============================================================================
# CHART 12 — LISTING PRICE VS ZESTIMATE
# ============================================================================

def chart_12_listing_vs_zestimate(df):
    data = df[["price", "zestimate"]].dropna()

    plt.figure(figsize=(10, 7))
    plt.scatter(
        data["price"],
        data["zestimate"],
        alpha=0.50,
        s=24,
    )

    minimum = min(data["price"].min(), data["zestimate"].min())
    maximum = max(data["price"].max(), data["zestimate"].max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=2,
        label="Equal value line",
    )

    plt.title("Listing Price vs. Zillow Zestimate")
    plt.xlabel("Listing Price ($)")
    plt.ylabel("Zestimate ($)")
    plt.legend()

    return save_current_figure("12_listing_price_vs_zestimate.png")


# ============================================================================
# CHART 13 — ZESTIMATE GAP DISTRIBUTION
# ============================================================================

def chart_13_zestimate_gap_distribution(df):
    data = df["zestimate_gap"].dropna()

    if data.empty:
        raise ValueError("No valid Zestimate gap values available.")

    # Keep the full dataset represented while using the middle 98% for the
    # main visual range so extreme gaps do not flatten the central distribution.
    lower_bound = data.quantile(0.01)
    upper_bound = data.quantile(0.99)
    display_data = data[
        (data >= lower_bound) &
        (data <= upper_bound)
    ]

    plt.figure(figsize=(10, 6))
    plt.hist(display_data, bins=40)

    plt.axvline(
        0,
        linestyle="--",
        linewidth=2,
        label="No gap",
    )

    plt.axvline(
        data.median(),
        linestyle=":",
        linewidth=2,
        label=f"Median gap: {format_currency(data.median())}",
    )

    plt.title("Zestimate Gap Distribution — Middle 98% of Observations")
    plt.xlabel("Zestimate Gap ($)")
    plt.ylabel("Number of Properties")
    plt.legend()

    plt.text(
        0.98,
        0.95,
        f"Full sample: N = {len(data):,}\n"
        f"Median: {format_currency(data.median())}\n"
        f"1st–99th percentile: "
        f"{format_currency(lower_bound)} to {format_currency(upper_bound)}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    return save_current_figure("13_zestimate_gap_distribution.png")


# ============================================================================
# CHART 14 — POTENTIAL OPPORTUNITIES BY ZIP
# ============================================================================

def chart_14_opportunities_by_zip(df):
    status = df["opportunity_status"].astype(str).str.lower()

    opportunity_mask = status.isin(
        [
            "potential opportunity",
            "opportunity",
            "potential_opportunity",
        ]
    )

    opportunity_data = df.loc[opportunity_mask].copy()

    if opportunity_data.empty:
        summary = pd.Series(dtype=float)
    else:
        summary = (
            opportunity_data.groupby("zipcode")
            .size()
            .sort_values(ascending=True)
        )

    plt.figure(figsize=(11, 7))

    if len(summary) > 0:
        y_positions = np.arange(len(summary))

        plt.barh(
            y_positions,
            summary.values,
        )

        plt.yticks(
            y_positions,
            summary.index.astype(str),
        )

        for position, value in zip(y_positions, summary.values):
            plt.text(
                value,
                position,
                f" {int(value)}",
                va="center",
                fontsize=8,
            )

    plt.xlabel("Number of Potential Opportunities")
    plt.ylabel("ZIP Code")
    plt.title("Potential Zestimate Opportunities by ZIP Code")

    plt.text(
        0.98,
        0.03,
        "Opportunity criteria are defined by the feature-engineered dataset.",
        transform=plt.gca().transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )

    return save_current_figure("14_potential_opportunities_by_zip.png")


# ============================================================================
# CHART 15 — 3 BEDROOM VS 4+ BEDROOM COMPARISON
# ============================================================================

def chart_15_bedroom_group_comparison(df):
    data = df[["beds", "price"]].dropna().copy()

    data["bedroom_group"] = np.where(
        data["beds"] == 3,
        "3 Bedrooms",
        np.where(data["beds"] >= 4, "4+ Bedrooms", np.nan),
    )

    data = data.dropna(subset=["bedroom_group"])

    grouped = (
        data.groupby("bedroom_group")["price"]
        .agg(["mean", "std", "count"])
    )

    order = ["3 Bedrooms", "4+ Bedrooms"]
    grouped = grouped.reindex(order)

    standard_error = grouped["std"] / np.sqrt(grouped["count"])

    # 95% normal-approximation confidence intervals.
    margin = 1.96 * standard_error

    plt.figure(figsize=(9, 6))

    x_positions = np.arange(len(grouped))

    plt.bar(
        x_positions,
        grouped["mean"],
        yerr=margin,
        capsize=6,
    )

    plt.xticks(x_positions, grouped.index)
    plt.ylabel("Mean Listing Price ($)")
    plt.xlabel("Bedroom Group")
    plt.title("Mean Listing Price: 3 Bedrooms vs. 4+ Bedrooms")

    for position, mean_value in zip(x_positions, grouped["mean"]):
        plt.text(
            position,
            mean_value,
            f" ${mean_value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.text(
        0.98,
        0.95,
        "Error bars: 95% confidence intervals",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85),
    )

    return save_current_figure("15_bedroom_group_comparison.png")


# ============================================================================
# REPORT GENERATION
# ============================================================================

def write_methodology(df):
    """Write the reproducible visualization methodology document."""

    output_path = OUTPUT_DIR / "VISUALIZATION_METHODOLOGY.md"

    text = f"""# Part 8 — Visualization Methodology

## Purpose

Part 8 translates the exploratory and inferential findings from the
San Antonio Zillow Housing Analysis into a reproducible Python visualization
layer.

## Source Dataset

- Source: `data/processed/San_Antonio_Zillow_Feature_Engineered.csv`
- Analytical records: {len(df):,}
- Unique properties: {df["zpid"].nunique():,}
- Duplicate `zpid`: {df["zpid"].duplicated().sum():,}

The feature-engineered analytical dataset is not modified by this script.

## Visualization Principles

The visualizations were selected to:

1. Show the distribution and shape of listing prices.
2. Visualize important relationships identified during EDA and statistical
   testing.
3. Emphasize geographic variation across ZIP codes with at least 10 listings.
4. Compare listing prices across property types.
5. Visualize the relationship between listing price and Zestimate.
6. Visualize Zestimate gaps and potential opportunity properties.
7. Provide visual support for statistically significant group comparisons.

## Transformations

`logPrice` is used for several relationship plots because listing prices are
right-skewed. This allows relationships to be viewed on a transformed scale
that is more consistent with the statistical analysis.

Raw dollar values are retained for business-facing distributions and
Zestimate-gap analysis.

## Geographic Sample Rule

ZIP-level charts use only ZIP codes containing at least 10 listings.
This matches the project's sufficient-sample definition and reduces the
risk of over-interpreting extremely small ZIP-level samples.

## Missing Data

Missing Zestimate and tax-assessed values are not artificially imputed.
Charts using these fields are based only on records with valid values for
the variables required by the specific visualization.

## Statistical Context

The visualizations are descriptive and explanatory. They do not establish
causation. Statistical significance, effect size, sample size, confidence
intervals, and data limitations should be considered alongside the charts.

## Output

The script generates 15 PNG charts in:

`reports/visualization/charts/`

The script clears previously generated PNG files before creating the new
visualizations so stale charts are not retained.

ZIP-level charts use horizontal bars with the same sufficient-sample ZIP
population as the underlying analysis. The Zestimate-gap visualization uses
the middle 98% of observations for the displayed histogram range so extreme
gaps do not obscure the central distribution; the full sample size and
percentile bounds are reported on the chart.
"""

    output_path.write_text(text, encoding="utf-8")

    return output_path


def write_key_findings(df):
    """Write a concise findings report based on the generated visual analysis."""

    price = df["price"].dropna()
    log_price = df["logPrice"].dropna()

    area_log = df[["area", "logPrice"]].dropna()
    tax_log = df[["taxAssessedValue", "logPrice"]].dropna()
    zestimate_pairs = df[["price", "zestimate", "zestimate_gap"]].dropna()

    zip_counts = df["zipcode"].value_counts()
    sufficient_zips = zip_counts[zip_counts >= 10].index

    zip_data = df[df["zipcode"].isin(sufficient_zips)].copy()

    zip_median = (
        zip_data.groupby("zipcode")["price"]
        .median()
        .sort_values(ascending=False)
    )

    property_type_summary = (
        df.groupby("homeType")["price"]
        .agg(["count", "median", "mean"])
        .sort_values("median", ascending=False)
    )

    status = df["opportunity_status"].astype(str).str.lower()

    opportunity_mask = status.isin(
        [
            "potential opportunity",
            "opportunity",
            "potential_opportunity",
        ]
    )

    opportunity_count = int(opportunity_mask.sum())

    if len(area_log) >= 3:
        area_corr = area_log["area"].corr(area_log["logPrice"], method="spearman")
    else:
        area_corr = np.nan

    if len(tax_log) >= 3:
        tax_corr = tax_log["taxAssessedValue"].corr(
            tax_log["logPrice"],
            method="spearman",
        )
    else:
        tax_corr = np.nan

    if len(zestimate_pairs) > 0:
        mean_gap = zestimate_pairs["zestimate_gap"].mean()
        median_gap = zestimate_pairs["zestimate_gap"].median()
    else:
        mean_gap = np.nan
        median_gap = np.nan

    lines = [
        "# Part 8 — Visualization Key Findings",
        "",
        "## Analytical Population",
        "",
        f"- The visualization layer uses {len(df):,} analytical properties.",
        f"- There are {df['zpid'].nunique():,} unique properties and "
        f"{df['zpid'].duplicated().sum():,} duplicate zpids.",
        "",
        "## Listing Price Distribution",
        "",
        f"- Median listing price: {format_currency(price.median())}.",
        f"- Mean listing price: {format_currency(price.mean())}.",
        f"- Median logPrice: {log_price.median():.4f}.",
        "",
        "The raw listing-price distribution is visibly right-skewed, supporting "
        "the use of logPrice for several relationship analyses.",
        "",
        "## Key Feature Relationships",
        "",
        f"- Spearman correlation between living area and logPrice: "
        f"{area_corr:.4f}.",
        f"- Spearman correlation between tax-assessed value and logPrice: "
        f"{tax_corr:.4f}.",
        "",
        "These visuals reinforce the statistical analysis showing that property "
        "size and tax-assessed value are strongly associated with listing price.",
        "",
        "## Geographic Variation",
        "",
        f"- {len(sufficient_zips):,} ZIP codes meet the sufficient-sample "
        "threshold of at least 10 listings.",
        f"- Highest median listing price among sufficient-sample ZIPs: "
        f"{format_currency(zip_median.iloc[0]) if len(zip_median) else 'N/A'}.",
        f"- Lowest median listing price among sufficient-sample ZIPs: "
        f"{format_currency(zip_median.iloc[-1]) if len(zip_median) else 'N/A'}.",
        "",
        "The ZIP-level visuals show substantial geographic variation in both "
        "median listing price and median price per square foot.",
        "",
        "## Property Type",
        "",
        "Property-type price distributions show meaningful differences in "
        "central tendency, while sample sizes vary considerably across types.",
        "",
        "### Property-Type Sample Sizes",
        "",
        "| Property Type | Listings | Median Price | Mean Price |",
        "|---|---:|---:|---:|",
    ]

    for home_type, row in property_type_summary.iterrows():
        lines.append(
            f"| {home_type} | {int(row['count']):,} | "
            f"{format_currency(row['median'])} | "
            f"{format_currency(row['mean'])} |"
        )

    lines.extend(
        [
            "",
            "The large difference in sample sizes should be considered when "
            "interpreting property-type comparisons.",
            "",
            "## Zestimate Analysis",
            "",
            f"- Complete listing-price/Zestimate pairs: "
            f"{len(zestimate_pairs):,}.",
            f"- Mean Zestimate gap: {format_currency(mean_gap)}.",
            f"- Median Zestimate gap: {format_currency(median_gap)}.",
            f"- Potential opportunity records identified by the dataset's "
            f"opportunity-status logic: {opportunity_count:,}.",
            "",
            "The listing-price-versus-Zestimate chart and Zestimate-gap "
            "distribution provide a visual view of where listing prices differ "
            "from Zillow's estimated values.",
            "",
            "## Interpretation Guidance",
            "",
            "These visualizations describe patterns in the observed Zillow "
            "listing dataset. They should not be interpreted as proof of "
            "causation. Geographic differences may reflect differences in "
            "property mix, neighborhood characteristics, and other factors not "
            "captured in the dataset.",
        ]
    )

    output_path = OUTPUT_DIR / "VISUALIZATION_KEY_FINDINGS.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return output_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("PART 8 — VISUALIZATION")
    print("=" * 70)

    print("\nSAN ANTONIO ZILLOW VISUALIZATION")

    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Input file: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    print(f"\nRows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns):,}")

    validate_dataset(df)

    numeric_columns = [
        "price",
        "logPrice",
        "area",
        "beds",
        "baths",
        "lotAreaSqFt",
        "taxAssessedValue",
        "daysOnZillow",
        "zestimate",
        "pricePerSqFt",
        "zestimate_gap",
        "zestimate_gap_pct",
    ]

    df = clean_numeric(df, numeric_columns)

    clear_old_charts()

    print("\nCreating visualization charts...")

    chart_functions = [
        chart_01_listing_price_distribution,
        chart_02_log_price_distribution,
        chart_03_area_vs_log_price,
        chart_04_bedrooms_vs_log_price,
        chart_05_bathrooms_vs_log_price,
        chart_06_lot_area_vs_log_price,
        chart_07_tax_assessed_vs_log_price,
        chart_08_correlation_matrix,
        chart_09_zip_median_price,
        chart_10_zip_median_price_per_sqft,
        chart_11_property_type_distribution,
        chart_12_listing_vs_zestimate,
        chart_13_zestimate_gap_distribution,
        chart_14_opportunities_by_zip,
        chart_15_bedroom_group_comparison,
    ]

    generated_files = []

    for function in chart_functions:
        generated_files.append(function(df))

    methodology_path = write_methodology(df)
    findings_path = write_key_findings(df)

    png_files = sorted(CHART_DIR.glob("*.png"))

    print("\n" + "=" * 70)
    print("PART 8 VISUALIZATION COMPLETE")
    print("=" * 70)

    print(f"\nFinal analytical records: {len(df):,}")
    print(f"Unique properties: {df['zpid'].nunique():,}")
    print(f"Duplicate zpid: {df['zpid'].duplicated().sum():,}")
    print(f"Generated charts: {len(png_files):,}")

    if len(png_files) != 15:
        raise ValueError(
            f"Expected 15 visualization charts, found {len(png_files)}."
        )

    expected_filenames = {
        "01_listing_price_distribution.png",
        "02_log_price_distribution.png",
        "03_area_vs_log_price.png",
        "04_bedrooms_vs_log_price.png",
        "05_bathrooms_vs_log_price.png",
        "06_lot_area_vs_log_price.png",
        "07_tax_assessed_value_vs_log_price.png",
        "08_correlation_matrix.png",
        "09_zip_median_listing_price.png",
        "10_zip_median_price_per_sqft.png",
        "11_property_type_price_distribution.png",
        "12_listing_price_vs_zestimate.png",
        "13_zestimate_gap_distribution.png",
        "14_potential_opportunities_by_zip.png",
        "15_bedroom_group_comparison.png",
    }

    actual_filenames = {file.name for file in png_files}

    if actual_filenames != expected_filenames:
        missing = expected_filenames - actual_filenames
        unexpected = actual_filenames - expected_filenames

        raise ValueError(
            f"Visualization output validation failed. "
            f"Missing: {sorted(missing)}; "
            f"Unexpected: {sorted(unexpected)}"
        )

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            "Final analytical population changed unexpectedly."
        )

    if df["zpid"].nunique() != EXPECTED_UNIQUE_ZPID:
        raise ValueError(
            "Unique property count changed unexpectedly."
        )

    if df["zpid"].duplicated().sum() != 0:
        raise ValueError(
            "Duplicate zpid values detected after visualization processing."
        )

    if not methodology_path.exists():
        raise ValueError("Visualization methodology report was not created.")

    if not findings_path.exists():
        raise ValueError("Visualization key findings report was not created.")

    print("\nAll expected Part 8 visualization outputs validated.")
    print("Visualization methodology and key findings were saved.")
    print("The feature-engineered dataset was not modified.")

    print(f"\nCharts saved to: {CHART_DIR}")
    print(f"Methodology: {methodology_path}")
    print(f"Key findings: {findings_path}")


if __name__ == "__main__":
    main()
