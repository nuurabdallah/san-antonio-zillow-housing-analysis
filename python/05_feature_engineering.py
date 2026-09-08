# ============================================================
# PART 5 — FEATURE ENGINEERING
# San Antonio Zillow Housing Analysis
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/Users/nuurabdallah/Desktop/Capstone Reimagined 2026"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "San_Antonio_Zillow_Analysis.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "feature_engineering"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "San_Antonio_Zillow_Feature_Engineered.csv"
)


# ============================================================
# EXPECTED SOURCE COLUMNS
# ============================================================

REQUIRED_SOURCE_COLUMNS = [
    "zpid",
    "addressZipcode",
    "price",
    "beds",
    "baths",
    "area",
    "homeType",
    "daysOnZillow",
    "zestimate",
    "taxAssessedValue",
    "lotAreaValue",
    "lotAreaUnit",
    "priceChange",
]


# ============================================================
# ML FEATURES
# ============================================================

ML_PREDICTORS = [
    "area",
    "beds",
    "baths",
    "lotAreaSqFt",
    "taxAssessedValue",
    "daysOnZillow",
]

ML_TARGET = "logPrice"


# ============================================================
# PRICE SEGMENTS
# ============================================================

PRICE_SEGMENTS = [
    "Under $150K",
    "$150K-$300K",
    "$300K-$500K",
    "$500K-$1M",
    "$1M-$2M",
    "$2M+",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_zillow_price(value):
    """
    Convert Zillow-style price values into numeric dollars.

    Examples:
        $269,000 -> 269000
        $151K   -> 151000
        $1.5M   -> 1500000
    """

    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    value = str(value).strip().upper()

    if value == "":
        return np.nan

    value = value.replace("$", "").replace(",", "").strip()

    try:
        if value.endswith("K"):
            return float(value[:-1]) * 1_000

        if value.endswith("M"):
            return float(value[:-1]) * 1_000_000

        if value.endswith("B"):
            return float(value[:-1]) * 1_000_000_000

        return float(value)

    except ValueError:
        return np.nan


def parse_zipcode(value):
    """
    Standardize ZIP codes as five-digit strings.
    """

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value == "":
        return np.nan

    # Handle values such as 78205.0
    try:
        numeric_value = float(value)

        if numeric_value.is_integer():
            value = str(int(numeric_value))

    except ValueError:
        pass

    # Keep the first five characters where appropriate.
    value = value.replace(".0", "")

    if value.isdigit():
        return value.zfill(5)[:5]

    return value


def create_price_segment(price):
    """
    Assign each property to a market price segment.
    """

    if pd.isna(price):
        return np.nan

    if price < 150_000:
        return "Under $150K"

    elif price < 300_000:
        return "$150K-$300K"

    elif price < 500_000:
        return "$300K-$500K"

    elif price < 1_000_000:
        return "$500K-$1M"

    elif price < 2_000_000:
        return "$1M-$2M"

    else:
        return "$2M+"


def load_dataset():
    """
    Load and validate the authoritative analysis dataset.
    """

    print("Loading source dataset...")
    print(f"Input file: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file was not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows loaded: {len(df)}")
    print(f"Columns loaded: {len(df.columns)}")

    if len(df) != 810:
        raise ValueError(
            f"Expected 810 rows in the authoritative analysis "
            f"dataset, but found {len(df)}."
        )

    missing_columns = [
        col
        for col in REQUIRED_SOURCE_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required source columns are missing:\n"
            + "\n".join(missing_columns)
        )

    print("\nRequired source columns validated.")

    return df


def validate_dataset_integrity(df):
    """
    Confirm the source population has not changed.
    """

    print("\n" + "=" * 70)
    print("DATASET INTEGRITY VALIDATION")
    print("=" * 70)

    duplicate_zpid = df["zpid"].duplicated().sum()
    unique_zpid = df["zpid"].nunique()

    print(f"Rows: {len(df)}")
    print(f"Unique zpid: {unique_zpid}")
    print(f"Duplicate zpid: {duplicate_zpid}")

    if len(df) != 810:
        raise ValueError("Dataset row count changed.")

    if unique_zpid != 810:
        raise ValueError("Expected 810 unique properties.")

    if duplicate_zpid != 0:
        raise ValueError("Duplicate zpid values detected.")

    print("\nDataset integrity validation passed.")


def create_features(df):
    """
    Create all analytical and business features.
    """

    # --------------------------------------------------------
    # ZIP CODE
    # --------------------------------------------------------

    df["zipcode"] = df["addressZipcode"].apply(parse_zipcode)

    print("\nCreated standardized zipcode field.")
    print(
        f"Missing ZIP codes: {df['zipcode'].isna().sum()}"
    )

    # --------------------------------------------------------
    # NUMERIC NORMALIZATION
    # --------------------------------------------------------

    numeric_columns = [
        "price",
        "beds",
        "baths",
        "area",
        "daysOnZillow",
        "zestimate",
        "taxAssessedValue",
        "lotAreaValue",
        "priceChange",
        "lotAreaSqFt",
        "timeOnZillow",
    ]

    for column in numeric_columns:

        if column not in df.columns:
            continue

        if column == "price":
            df[column] = df[column].apply(
                parse_zillow_price
            )

        else:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    print("\nNumeric fields normalized.")

    # --------------------------------------------------------
    # LOT AREA
    # --------------------------------------------------------

    if "lotAreaSqFt" not in df.columns:
        df["lotAreaSqFt"] = np.nan

    # If lot area is already supplied in square feet,
    # preserve it. Otherwise convert common units.
    if "lotAreaUnit" in df.columns:

        unit = (
            df["lotAreaUnit"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        missing_lot_sqft = df["lotAreaSqFt"].isna()

        acres_mask = (
            missing_lot_sqft
            & unit.str.contains(
                "acre",
                na=False
            )
        )

        sqft_mask = (
            missing_lot_sqft
            & unit.str.contains(
                "sqft|square feet|sq ft",
                na=False
            )
        )

        df.loc[sqft_mask, "lotAreaSqFt"] = (
            df.loc[sqft_mask, "lotAreaValue"]
        )

        df.loc[acres_mask, "lotAreaSqFt"] = (
            df.loc[acres_mask, "lotAreaValue"]
            * 43_560
        )

    # --------------------------------------------------------
    # PRICE PER SQUARE FOOT
    # --------------------------------------------------------

    df["pricePerSqFt"] = np.where(
        (df["area"] > 0) & (df["price"] > 0),
        df["price"] / df["area"],
        np.nan
    )

    print("\nCreated pricePerSqFt.")

    # --------------------------------------------------------
    # LOG PRICE
    # --------------------------------------------------------

    df["logPrice"] = np.where(
        df["price"] > 0,
        np.log(df["price"]),
        np.nan
    )

    print("Created logPrice.")

    # --------------------------------------------------------
    # ZESTIMATE GAP
    # --------------------------------------------------------

    df["zestimate_gap"] = np.where(
        df["zestimate"].notna()
        & df["price"].notna(),
        df["zestimate"] - df["price"],
        np.nan
    )

    df["zestimate_gap_pct"] = np.where(
        df["zestimate"].notna()
        & (df["price"] > 0),
        (
            (df["zestimate"] - df["price"])
            / df["price"]
        ),
        np.nan
    )

    df["opportunity_status"] = np.where(
        df["zestimate"].isna(),
        np.nan,
        np.where(
            df["zestimate"] > df["price"],
            "Potential Opportunity",
            "Above Zestimate"
        )
    )

    print("Created Zestimate gap features.")

    # --------------------------------------------------------
    # DEAL SCORE
    # --------------------------------------------------------

    valid_gap = df["zestimate_gap_pct"].notna()

    df["deal_score"] = np.nan

    if valid_gap.sum() > 0:

        # Percentile rank among properties with
        # available Zestimate information.
        df.loc[valid_gap, "deal_score"] = (
            df.loc[
                valid_gap,
                "zestimate_gap_pct"
            ]
            .rank(
                method="average",
                pct=True
            )
            * 100
        )

    print("Created Deal Score.")

    # --------------------------------------------------------
    # PRICE SEGMENT
    # --------------------------------------------------------

    df["price_segment"] = (
        df["price"]
        .apply(create_price_segment)
    )

    print("Created price segments.")

    # --------------------------------------------------------
    # ZIP SAMPLE SIZE
    # --------------------------------------------------------

    df["zip_listing_count"] = (
        df.groupby("zipcode")["zpid"]
        .transform("count")
    )

    df["zip_sample_status"] = np.where(
        df["zip_listing_count"] >= 10,
        "Sufficient Sample",
        "Sparse Sample"
    )

    print("Created ZIP sample-size features.")

    # --------------------------------------------------------
    # PRICE CHANGE FLAG
    # --------------------------------------------------------

    df["hasPriceChange"] = np.where(
        df["priceChange"].notna()
        & (df["priceChange"] != 0),
        1,
        0
    )

    print("Created hasPriceChange.")

    return df


def validate_engineered_features(df):
    """
    Validate all engineered features.
    """

    print("\n" + "=" * 70)
    print("ENGINEERED FEATURE VALIDATION")
    print("=" * 70)

    print(f"Total rows: {len(df)}")
    print(f"Unique zpids: {df['zpid'].nunique()}")
    print(
        f"Duplicate zpids: "
        f"{df['zpid'].duplicated().sum()}"
    )

    # --------------------------------------------------------
    # CORE INTEGRITY
    # --------------------------------------------------------

    assert len(df) == 810
    assert df["zpid"].nunique() == 810
    assert df["zpid"].duplicated().sum() == 0

    # --------------------------------------------------------
    # ZIP
    # --------------------------------------------------------

    missing_zip = df["zipcode"].isna().sum()

    print(f"Missing ZIP codes: {missing_zip}")

    assert missing_zip == 0

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    invalid_prices = (
        df["price"].isna()
        | (df["price"] <= 0)
    ).sum()

    print(f"Invalid prices: {invalid_prices}")

    assert invalid_prices == 0

    # --------------------------------------------------------
    # PRICE/SQ FT
    # --------------------------------------------------------

    expected_ppsf = (
        df["price"] / df["area"]
    )

    ppsf_mismatch = (
        ~np.isclose(
            df["pricePerSqFt"],
            expected_ppsf,
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True
        )
    ).sum()

    print(
        f"Price/Sq Ft mismatches: "
        f"{ppsf_mismatch}"
    )

    assert ppsf_mismatch == 0

    # --------------------------------------------------------
    # LOG PRICE
    # --------------------------------------------------------

    expected_log = np.log(df["price"])

    log_mismatch = (
        ~np.isclose(
            df["logPrice"],
            expected_log,
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True
        )
    ).sum()

    print(
        f"Log price mismatches: "
        f"{log_mismatch}"
    )

    assert log_mismatch == 0

    # --------------------------------------------------------
    # ZESTIMATE
    # --------------------------------------------------------

    zestimate_count = (
        df["zestimate"].notna().sum()
    )

    gap_count = (
        df["zestimate_gap"].notna().sum()
    )

    print(
        f"Listings with Zestimate: "
        f"{zestimate_count}"
    )

    print(
        f"Listings with Zestimate gap: "
        f"{gap_count}"
    )

    assert zestimate_count == gap_count

    expected_gap = (
        df["zestimate"] - df["price"]
    )

    gap_mismatch = (
        ~np.isclose(
            df["zestimate_gap"],
            expected_gap,
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True
        )
    ).sum()

    print(
        f"Zestimate gap mismatches: "
        f"{gap_mismatch}"
    )

    assert gap_mismatch == 0

    # --------------------------------------------------------
    # GAP PERCENTAGE
    # --------------------------------------------------------

    expected_gap_pct = (
        (df["zestimate"] - df["price"])
        / df["price"]
    )

    gap_pct_mismatch = (
        ~np.isclose(
            df["zestimate_gap_pct"],
            expected_gap_pct,
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True
        )
    ).sum()

    print(
        f"Zestimate gap % mismatches: "
        f"{gap_pct_mismatch}"
    )

    assert gap_pct_mismatch == 0

    # --------------------------------------------------------
    # OPPORTUNITIES
    # --------------------------------------------------------

    opportunities = (
        df["opportunity_status"]
        == "Potential Opportunity"
    ).sum()
    
    print(
        f"Potential opportunities: "
        f"{opportunities}"
    )

    assert opportunities == 16

    # --------------------------------------------------------
    # DEAL SCORE
    # --------------------------------------------------------

    deal_score_count = (
        df["deal_score"].notna().sum()
    )

    print(
        f"Deal Score records: "
        f"{deal_score_count}"
    )

    print(
        f"Deal Score range: "
        f"{df['deal_score'].min():.2f} - "
        f"{df['deal_score'].max():.2f}"
    )

    assert deal_score_count == 576
    assert df["deal_score"].min() >= 0
    assert df["deal_score"].max() <= 100

    # --------------------------------------------------------
    # PRICE SEGMENTS
    # --------------------------------------------------------

    segment_count = (
        df["price_segment"].notna().sum()
    )

    print(
        f"Price segment records: "
        f"{segment_count}"
    )

    print(
        f"Price segment total: "
        f"{df['price_segment'].value_counts().sum()}"
    )

    assert segment_count == 810
    assert set(
        df["price_segment"].dropna().unique()
    ).issubset(set(PRICE_SEGMENTS))

    # --------------------------------------------------------
    # ZIP SAMPLE STATUS
    # --------------------------------------------------------

    zip_status_count = (
        df["zip_sample_status"].notna().sum()
    )

    invalid_zip_status = (
        ~df["zip_sample_status"].isin(
            [
                "Sufficient Sample",
                "Sparse Sample"
            ]
        )
    ).sum()

    print(
        f"ZIP sample-status records: "
        f"{zip_status_count}"
    )

    print(
        f"Invalid ZIP sample statuses: "
        f"{invalid_zip_status}"
    )

    assert zip_status_count == 810
    assert invalid_zip_status == 0

    # --------------------------------------------------------
    # ML PREDICTORS
    # --------------------------------------------------------

    missing_ml_columns = [
        col
        for col in ML_PREDICTORS
        if col not in df.columns
    ]

    if missing_ml_columns:
        raise ValueError(
            "Missing ML predictor columns: "
            + ", ".join(missing_ml_columns)
        )

    if ML_TARGET not in df.columns:
        raise ValueError(
            f"ML target '{ML_TARGET}' is missing."
        )

    print("\nML predictor columns validated.")

    print("\nAll engineered feature validation checks passed.")


def create_feature_summary(df):
    """
    Create a summary of engineered features.
    """

    summary = []

    features = [
        "zipcode",
        "pricePerSqFt",
        "logPrice",
        "zestimate_gap",
        "zestimate_gap_pct",
        "deal_score",
        "opportunity_status",
        "price_segment",
        "zip_listing_count",
        "zip_sample_status",
        "hasPriceChange",
    ]

    for feature in features:

        if feature not in df.columns:
            continue

        summary.append({
            "feature": feature,
            "data_type": str(
                df[feature].dtype
            ),
            "non_null_count": int(
                df[feature].notna().sum()
            ),
            "missing_count": int(
                df[feature].isna().sum()
            ),
            "unique_values": int(
                df[feature].nunique(
                    dropna=True
                )
            ),
        })

    return pd.DataFrame(summary)


def create_feature_dictionary():
    """
    Document engineered features and their purpose.
    """

    rows = [

        {
            "feature": "zipcode",
            "definition":
                "Standardized five-digit ZIP code derived from addressZipcode.",
            "category": "Location",
            "ml_role": "Not used",
        },

        {
            "feature": "pricePerSqFt",
            "definition":
                "Listing price divided by living area in square feet.",
            "category": "Pricing",
            "ml_role": "Excluded due to target leakage",
        },

        {
            "feature": "logPrice",
            "definition":
                "Natural logarithm of listing price.",
            "category": "Target",
            "ml_role": "ML target",
        },

        {
            "feature": "zestimate_gap",
            "definition":
                "Zestimate minus listing price.",
            "category": "Zestimate Analysis",
            "ml_role": "Excluded due to target leakage",
        },

        {
            "feature": "zestimate_gap_pct",
            "definition":
                "(Zestimate minus listing price) divided by listing price.",
            "category": "Zestimate Analysis",
            "ml_role": "Excluded due to target leakage",
        },

        {
            "feature": "deal_score",
            "definition":
                "Percentile-based score from Zestimate gap percentage.",
            "category": "Investment Analysis",
            "ml_role": "Excluded due to target leakage",
        },

        {
            "feature": "opportunity_status",
            "definition":
                "Identifies properties where Zestimate exceeds listing price.",
            "category": "Investment Analysis",
            "ml_role": "Excluded from ML",
        },

        {
            "feature": "price_segment",
            "definition":
                "Categorical market segment based on listing price.",
            "category": "Market Segmentation",
            "ml_role": "Business analysis",
        },

        {
            "feature": "zip_listing_count",
            "definition":
                "Number of listings represented in each ZIP code.",
            "category": "Location",
            "ml_role": "Business analysis",
        },

        {
            "feature": "zip_sample_status",
            "definition":
                "Sufficient Sample for ZIPs with at least 10 listings; otherwise Sparse Sample.",
            "category": "Location",
            "ml_role": "Business analysis",
        },

        {
            "feature": "hasPriceChange",
            "definition":
                "Binary indicator identifying listings with a recorded non-zero price change.",
            "category": "Listing Activity",
            "ml_role": "Potential predictor",
        },
    ]

    return pd.DataFrame(rows)


def create_methodology_notes():
    """
    Create feature engineering methodology documentation.
    """

    return """# Part 5 — Feature Engineering Methodology

## Source Dataset

The authoritative analysis dataset contains 810 San Antonio Zillow
property listings.

The source ZIP field `addressZipcode` is standardized into the
analytical field `zipcode`.

## Pricing Features

### pricePerSqFt

Calculated as:

price / living area

This feature is used for market and property-level analysis.

It is excluded from the machine-learning predictor set because it
contains the target variable (price) directly.

### logPrice

Calculated using the natural logarithm:

ln(price)

This is the machine-learning target used for price prediction.

## Zestimate Features

### zestimate_gap

Calculated as:

Zestimate - Listing Price

### zestimate_gap_pct

Calculated as:

(Zestimate - Listing Price) / Listing Price

These features support Zestimate and investment-opportunity analysis.

They are excluded from machine-learning predictors because they contain
the target listing price.

## Deal Score

Deal Score is calculated as the percentile rank of Zestimate gap
percentage among listings with available Zestimate data.

A higher score indicates that a property is further below its Zestimate
relative to other properties in the dataset.

Deal Score is an analytical ranking metric and is not a guarantee of
investment return.

## Price Segments

Listings are classified into:

- Under $150K
- $150K-$300K
- $300K-$500K
- $500K-$1M
- $1M-$2M
- $2M+

## ZIP Sample Size

ZIP codes with at least 10 listings are classified as:

`Sufficient Sample`

ZIP codes with fewer than 10 listings are classified as:

`Sparse Sample`

Sparse ZIP codes are retained in the dataset but should be interpreted
with caution when making standalone ZIP-level comparisons.

## Machine Learning Predictors

The finalized machine-learning predictor set is:

- area
- beds
- baths
- lotAreaSqFt
- taxAssessedValue
- daysOnZillow

Target:

- logPrice

Business-analysis variables containing listing price or derived price
relationships are intentionally excluded from the ML predictor set to
prevent target leakage.

## Population Preservation

Feature engineering does not remove observations.

The analysis-ready population remains 810 unique properties.
"""


def save_outputs(df):
    """
    Save the feature-engineered dataset and documentation.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # FEATURE SUMMARY
    # --------------------------------------------------------

    feature_summary = create_feature_summary(df)

    feature_summary.to_csv(
        REPORT_DIR / "feature_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # FEATURE DICTIONARY
    # --------------------------------------------------------

    feature_dictionary = (
        create_feature_dictionary()
    )

    feature_dictionary.to_csv(
        REPORT_DIR / "feature_dictionary.csv",
        index=False
    )

    # --------------------------------------------------------
    # METHODOLOGY
    # --------------------------------------------------------

    methodology_file = (
        REPORT_DIR
        / "FEATURE_ENGINEERING_METHODOLOGY.md"
    )

    methodology_file.write_text(
        create_methodology_notes(),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING OUTPUTS SAVED")
    print("=" * 70)

    print("\nFeature-engineered dataset:")
    print(f"  {OUTPUT_FILE}")

    print("\nReports:")
    print("  - feature_summary.csv")
    print("  - feature_dictionary.csv")
    print("  - FEATURE_ENGINEERING_METHODOLOGY.md")


def print_final_summary(df):
    """
    Print final Part 5 summary.
    """

    print("\n" + "=" * 70)
    print("PART 5 FEATURE ENGINEERING COMPLETE")
    print("=" * 70)

    print(f"\nFinal properties: {len(df)}")
    print(
        f"Unique properties: "
        f"{df['zpid'].nunique()}"
    )

    zestimate_coverage = (
        df["zestimate"].notna().mean()
        * 100
    )

    opportunities = (
        df["opportunity_status"]
        == "Potential Opportunity"
    ).sum()

    print(
        f"Zestimate coverage: "
        f"{zestimate_coverage:.2f}%"
    )

    print(
        f"Potential opportunities: "
        f"{opportunities}"
    )

    print("\nML predictors:")

    for feature in ML_PREDICTORS:
        print(f"  - {feature}")

    print("\nML target:")
    print(f"  - {ML_TARGET}")

    print("\nBusiness-analysis features:")

    business_features = [
        "pricePerSqFt",
        "zestimate_gap",
        "zestimate_gap_pct",
        "deal_score",
        "opportunity_status",
        "price_segment",
        "zip_listing_count",
        "zip_sample_status",
    ]

    for feature in business_features:
        print(f"  - {feature}")

    print(
        "\nThe analysis-ready population was "
        "preserved at 810 properties."
    )

    print(
        "Target-leaking business metrics are "
        "separated from ML predictors."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PART 5 — FEATURE ENGINEERING")
    print("=" * 70)

    print("\nSAN ANTONIO ZILLOW FEATURE ENGINEERING")

    print(f"\nProject root: {PROJECT_ROOT}")

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # INTEGRITY
    # --------------------------------------------------------

    validate_dataset_integrity(df)

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    df = create_features(df)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validate_engineered_features(df)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_outputs(df)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print_final_summary(df)


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/San_Antonio_Zillow_Analysis.csv"
)

OUTPUT_FILE = Path(
    "data/processed/San_Antonio_Zillow_Feature_Engineered.csv"
)

OUTPUT_DIR = Path(
    "reports/features"
)

MIN_ZIP_SAMPLE_SIZE = 10

EXPECTED_ROWS = 810


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

ML_PREDICTORS = [
    "area",
    "beds",
    "baths",
    "lotAreaSqFt",
    "taxAssessedValue",
    "daysOnZillow",
]

TARGET_VARIABLE = "logPrice"


BUSINESS_ANALYSIS_FEATURES = [
    "pricePerSqFt",
    "zestimate_gap",
    "zestimate_gap_pct",
    "deal_score",
    "opportunity_status",
    "price_segment",
    "zip_listing_count",
    "zip_sample_status",
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    """Load the cleaned analytical dataset."""

    print("=" * 70)
    print("PART 5 — FEATURE ENGINEERING")
    print("=" * 70)
    print()

    print(
        "SAN ANTONIO ZILLOW FEATURE ENGINEERING"
    )
    print()

    print(
        f"Input file: {INPUT_FILE}"
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
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

    print()

    return df


# ============================================================
# VALIDATE REQUIRED SOURCE COLUMNS
# ============================================================

def validate_required_columns(df):
    """Confirm that required source columns exist."""

    required_columns = [
        "zpid",
        "price",
        "addressZipcode",
        "beds",
        "baths",
        "area",
        "homeType",
        "daysOnZillow",
        "zestimate",
        "taxAssessedValue",
        "lotAreaSqFt",
        "pricePerSqFt",
        "logPrice",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required source columns are missing: "
            + ", ".join(missing_columns)
        )

    print(
        "Required source columns validated."
    )

    print()

    return df


# ============================================================
# CREATE STANDARDIZED ZIP CODE
# ============================================================

def create_zipcode(df):
    """
    Standardize the Zillow addressZipcode field into
    a project-level zipcode field.

    The original source field is preserved.
    """

    df["zipcode"] = (
        df["addressZipcode"]
        .astype("string")
        .str.strip()
    )

    # Convert empty strings to missing values.
    df["zipcode"] = (
        df["zipcode"]
        .replace("", pd.NA)
    )

    missing_zip = (
        df["zipcode"]
        .isna()
        .sum()
    )

    print(
        "Created standardized zipcode field."
    )

    print(
        f"Missing ZIP codes: {missing_zip:,}"
    )

    if missing_zip != 0:
        raise ValueError(
            "Missing ZIP codes detected."
        )

    print()

    return df


# ============================================================
# NORMALIZE NUMERIC COLUMNS
# ============================================================

def normalize_numeric_columns(df):
    """
    Convert relevant fields to numeric values.

    Zillow listing prices can contain formats such as:
        $269,000
        $151K
        $1.5M
    """

    numeric_columns = [
        "beds",
        "baths",
        "area",
        "daysOnZillow",
        "zestimate",
        "taxAssessedValue",
        "lotAreaSqFt",
        "pricePerSqFt",
        "logPrice",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Parse Zillow-formatted listing prices
    # --------------------------------------------------------

    original_price = (
        df["price"].copy()
    )

    numeric_price = pd.to_numeric(
        original_price,
        errors="coerce"
    )

    def parse_price(value):
        """Convert Zillow-formatted price strings to numeric."""

        if pd.isna(value):
            return np.nan

        text = (
            str(value)
            .strip()
            .upper()
        )

        if text == "":
            return np.nan

        text = (
            text
            .replace("$", "")
            .replace(",", "")
            .strip()
        )

        try:

            if text.endswith("M"):
                return (
                    float(text[:-1])
                    * 1_000_000
                )

            if text.endswith("K"):
                return (
                    float(text[:-1])
                    * 1_000
                )

            return float(text)

        except (
            ValueError,
            TypeError
        ):

            return np.nan

    parsed_price = (
        original_price.apply(
            parse_price
        )
    )

    df["price"] = (
        numeric_price
        .fillna(parsed_price)
    )

    print(
        "Numeric fields normalized."
    )

    print()

    return df


# ============================================================
# VALIDATE DATASET INTEGRITY
# ============================================================

def validate_dataset_integrity(df):
    """Validate population size and property identifiers."""

    print("=" * 70)
    print("DATASET INTEGRITY VALIDATION")
    print("=" * 70)
    print()

    row_count = len(df)

    unique_zpid = (
        df["zpid"]
        .nunique()
    )

    duplicate_zpid = (
        df["zpid"]
        .duplicated()
        .sum()
    )

    print(
        f"Rows: {row_count:,}"
    )

    print(
        f"Unique zpid: {unique_zpid:,}"
    )

    print(
        f"Duplicate zpid: {duplicate_zpid:,}"
    )

    if row_count != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {row_count}."
        )

    if unique_zpid != EXPECTED_ROWS:
        raise ValueError(
            "The number of unique properties does not "
            "equal the expected 810-property population."
        )

    if duplicate_zpid != 0:
        raise ValueError(
            "Duplicate zpid values detected."
        )

    print()

    print(
        "Dataset integrity validation passed."
    )

    print()

    return df


# ============================================================
# CREATE PRICE PER SQUARE FOOT
# ============================================================

def create_price_per_sqft(df):
    """
    Calculate listing price per square foot.

    This is a business-analysis metric and is NOT an ML predictor
    because it directly contains the target listing price.
    """

    valid = (
        df["price"].notna()
        & df["area"].notna()
        & (df["area"] > 0)
    )

    df["pricePerSqFt"] = np.where(
        valid,
        df["price"] / df["area"],
        np.nan
    )

    print(
        "Created pricePerSqFt."
    )

    print()

    return df


# ============================================================
# CREATE LOG PRICE
# ============================================================

def create_log_price(df):
    """
    Create natural-log transformed listing price.

    logPrice is the machine-learning target variable.
    """

    valid = (
        df["price"].notna()
        & (df["price"] > 0)
    )

    df["logPrice"] = np.where(
        valid,
        np.log(df["price"]),
        np.nan
    )

    print(
        "Created logPrice."
    )

    print()

    return df


# ============================================================
# CREATE ZESTIMATE FEATURES
# ============================================================

def create_zestimate_features(df):
    """
    Create Zestimate comparison features.

    zestimate_gap:
        Zestimate minus listing price.

    zestimate_gap_pct:
        Percentage difference between Zestimate
        and listing price.

    opportunity_status:
        Potential Opportunity when Zestimate exceeds
        listing price.
    """

    valid_zestimate = (
        df["zestimate"].notna()
        & df["price"].notna()
        & (df["price"] > 0)
    )

    df["zestimate_gap"] = np.where(
        valid_zestimate,
        df["zestimate"] - df["price"],
        np.nan
    )

    df["zestimate_gap_pct"] = np.where(
        valid_zestimate,
        (
            (
                df["zestimate"]
                - df["price"]
            )
            / df["price"]
        ),
        np.nan
    )

    df["opportunity_status"] = np.select(
        [
            df["zestimate"].isna(),
            df["zestimate_gap"] > 0,
        ],
        [
            "Null / missing Zestimate",
            "Potential Opportunity",
        ],
        default="Above Zestimate",
    )

    print(
        "Created Zestimate gap features."
    )

    print()

    return df


# ============================================================
# CREATE DEAL SCORE
# ============================================================

def create_deal_score(df):
    """
    Create a relative Deal Score based on Zestimate
    gap percentage.

    Deal Score:
        100 * percentile rank of zestimate_gap_pct

    Higher values indicate larger positive Zestimate gaps
    relative to other properties with available Zestimate data.

    This is a business-analysis metric and must not be used
    as an ML predictor.
    """

    df["deal_score"] = np.nan

    valid = (
        df["zestimate_gap_pct"]
        .notna()
    )

    if valid.sum() > 0:

        df.loc[
            valid,
            "deal_score"
        ] = (
            df.loc[
                valid,
                "zestimate_gap_pct"
            ]
            .rank(
                method="min",
                pct=True
            )
            * 100
        )

    print(
        "Created Deal Score."
    )

    print()

    return df


# ============================================================
# CREATE PRICE SEGMENTS
# ============================================================

def create_price_segments(df):
    """Create standardized listing-price segments."""

    conditions = [
        df["price"] < 150_000,
        df["price"] < 300_000,
        df["price"] < 500_000,
        df["price"] < 1_000_000,
        df["price"] < 2_000_000,
        df["price"] >= 2_000_000,
    ]

    choices = [
        "Under $150K",
        "$150K-$300K",
        "$300K-$500K",
        "$500K-$1M",
        "$1M-$2M",
        "$2M+",
    ]

    df["price_segment"] = np.select(
        conditions,
        choices,
        default="Unclassified",
    )

    print(
        "Created price segments."
    )

    print()

    return df


# ============================================================
# CREATE ZIP FEATURES
# ============================================================

def create_zip_features(df):
    """
    Create ZIP-level sample-size features.

    ZIPs with fewer than 10 listings are classified
    as sparse.

    Sparse ZIPs remain in the dataset and are not removed.
    """

    zip_counts = (
        df.groupby("zipcode")["zpid"]
        .transform("count")
    )

    df["zip_listing_count"] = (
        zip_counts
    )

    df["zip_sample_status"] = np.where(
        df["zip_listing_count"]
        >= MIN_ZIP_SAMPLE_SIZE,
        "Sufficient Sample",
        "Sparse Sample",
    )

    print(
        "Created ZIP sample-size features."
    )

    print()

    return df


# ============================================================
# CREATE PRICE CHANGE FLAG
# ============================================================

def create_price_change_flag(df):
    """Create a standardized price-change indicator."""

    if "priceChange" in df.columns:

        df["hasPriceChange"] = np.where(
            (
                df["priceChange"].notna()
                & (df["priceChange"] != 0)
            ),
            1,
            0,
        )

    else:

        df["hasPriceChange"] = 0

    print(
        "Created hasPriceChange."
    )

    print()

    return df


# ============================================================
# CREATE FEATURE SUMMARY
# ============================================================

def create_feature_summary(df):
    """Create a summary of engineered features."""

    summary = []

    feature_columns = [
        "zipcode",
        "pricePerSqFt",
        "logPrice",
        "zestimate_gap",
        "zestimate_gap_pct",
        "deal_score",
        "opportunity_status",
        "price_segment",
        "zip_listing_count",
        "zip_sample_status",
        "hasPriceChange",
    ]

    for feature in feature_columns:

        if feature not in df.columns:
            continue

        series = (
            df[feature]
        )

        summary.append(
            {
                "feature": feature,
                "data_type": str(
                    series.dtype
                ),
                "non_null_count": (
                    series.notna().sum()
                ),
                "missing_count": (
                    series.isna().sum()
                ),
                "missing_percentage": (
                    series.isna().mean()
                    * 100
                ),
                "unique_values": (
                    series.nunique(
                        dropna=True
                    )
                ),
            }
        )

    return pd.DataFrame(
        summary
    )


# ============================================================
# CREATE FEATURE DICTIONARY
# ============================================================

def create_feature_dictionary():
    """Create formal documentation for project features."""

    definitions = [

        {
            "feature": "zipcode",
            "definition": (
                "Standardized ZIP code derived from "
                "the Zillow addressZipcode field."
            ),
            "category": "Location",
            "ml_safe": "Potential",
            "reason": (
                "Geographic context; not used in "
                "the finalized ML model."
            ),
        },

        {
            "feature": "pricePerSqFt",
            "definition": (
                "Listing price divided by living area "
                "in square feet."
            ),
            "category": "Business Analysis",
            "ml_safe": "No",
            "reason": (
                "Contains listing price and therefore "
                "introduces target leakage."
            ),
        },

        {
            "feature": "logPrice",
            "definition": (
                "Natural logarithm of listing price."
            ),
            "category": "ML Target",
            "ml_safe": "Target",
            "reason": (
                "This is the dependent variable used "
                "for price prediction."
            ),
        },

        {
            "feature": "zestimate_gap",
            "definition": (
                "Zestimate minus listing price."
            ),
            "category": "Business Analysis",
            "ml_safe": "No",
            "reason": (
                "Directly contains the target "
                "listing price."
            ),
        },

        {
            "feature": "zestimate_gap_pct",
            "definition": (
                "(Zestimate minus listing price) "
                "divided by listing price."
            ),
            "category": "Business Analysis",
            "ml_safe": "No",
            "reason": (
                "Directly contains the target "
                "listing price."
            ),
        },

        {
            "feature": "deal_score",
            "definition": (
                "Percentile-based score representing "
                "relative Zestimate gap percentage."
            ),
            "category": "Business Analysis",
            "ml_safe": "No",
            "reason": (
                "Derived from Zestimate gap percentage "
                "and listing price."
            ),
        },

        {
            "feature": "opportunity_status",
            "definition": (
                "Classification based on whether "
                "Zestimate exceeds listing price."
            ),
            "category": "Business Analysis",
            "ml_safe": "No",
            "reason": (
                "Uses listing price and Zestimate."
            ),
        },

        {
            "feature": "price_segment",
            "definition": (
                "Categorical grouping of properties "
                "by listing price."
            ),
            "category": "Business Analysis",
            "ml_safe": "No",
            "reason": (
                "Derived directly from listing price."
            ),
        },

        {
            "feature": "zip_listing_count",
            "definition": (
                "Number of listings represented "
                "in the same ZIP code."
            ),
            "category": "Market Context",
            "ml_safe": "Potential",
            "reason": (
                "Contextual feature; not used in "
                "the finalized ML model."
            ),
        },

        {
            "feature": "zip_sample_status",
            "definition": (
                "Indicates whether a ZIP has at "
                "least 10 listings."
            ),
            "category": "Market Context",
            "ml_safe": "Potential",
            "reason": (
                "Used for analytical reliability "
                "rather than the finalized ML model."
            ),
        },

        {
            "feature": "area",
            "definition": (
                "Living area in square feet."
            ),
            "category": "ML Predictor",
            "ml_safe": "Yes",
            "reason": (
                "Property characteristic available "
                "independently of target price."
            ),
        },

        {
            "feature": "beds",
            "definition": (
                "Number of bedrooms."
            ),
            "category": "ML Predictor",
            "ml_safe": "Yes",
            "reason": (
                "Property characteristic available "
                "independently of target price."
            ),
        },

        {
            "feature": "baths",
            "definition": (
                "Number of bathrooms."
            ),
            "category": "ML Predictor",
            "ml_safe": "Yes",
            "reason": (
                "Property characteristic available "
                "independently of target price."
            ),
        },

        {
            "feature": "lotAreaSqFt",
            "definition": (
                "Lot area standardized to square feet."
            ),
            "category": "ML Predictor",
            "ml_safe": "Yes",
            "reason": (
                "Property characteristic available "
                "independently of target price."
            ),
        },

        {
            "feature": "taxAssessedValue",
            "definition": (
                "Tax-assessed value reported in "
                "the Zillow data."
            ),
            "category": "ML Predictor",
            "ml_safe": "Yes",
            "reason": (
                "Independent property valuation feature "
                "retained in the finalized model."
            ),
        },

        {
            "feature": "daysOnZillow",
            "definition": (
                "Number of days the property has "
                "been listed on Zillow."
            ),
            "category": "ML Predictor",
            "ml_safe": "Yes",
            "reason": (
                "Listing-duration characteristic retained "
                "in the finalized model."
            ),
        },
    ]

    return pd.DataFrame(
        definitions
    )


# ============================================================
# VALIDATE ENGINEERED FEATURES
# ============================================================

def validate_engineered_features(df):
    """Run final validation checks on engineered features."""

    print("=" * 70)
    print("ENGINEERED FEATURE VALIDATION")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Population
    # --------------------------------------------------------

    print(
        f"Total rows: {len(df):,}"
    )

    print(
        f"Unique zpids: "
        f"{df['zpid'].nunique():,}"
    )

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            "Feature engineering changed "
            "the dataset population."
        )

    if df["zpid"].nunique() != EXPECTED_ROWS:
        raise ValueError(
            "Duplicate or missing zpid values detected."
        )

    # --------------------------------------------------------
    # Duplicate identifiers
    # --------------------------------------------------------

    duplicate_count = (
        df["zpid"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate zpids: "
        f"{duplicate_count:,}"
    )

    if duplicate_count != 0:
        raise ValueError(
            "Duplicate zpid values detected."
        )

    # --------------------------------------------------------
    # ZIP
    # --------------------------------------------------------

    missing_zip = (
        df["zipcode"]
        .isna()
        .sum()
    )

    print(
        f"Missing ZIP codes: "
        f"{missing_zip:,}"
    )

    if missing_zip != 0:
        raise ValueError(
            "Missing ZIP codes detected."
        )

    # --------------------------------------------------------
    # Price
    # --------------------------------------------------------

    invalid_price = (
        df["price"].isna()
        | (df["price"] <= 0)
    ).sum()

    print(
        f"Invalid prices: "
        f"{invalid_price:,}"
    )

    if invalid_price != 0:
        raise ValueError(
            "Invalid listing prices detected."
        )

    # --------------------------------------------------------
    # Price per square foot
    # --------------------------------------------------------

    valid_ppsf = (
        df["pricePerSqFt"].notna()
        & df["area"].notna()
        & (df["area"] > 0)
    )

    expected_ppsf = (
        df.loc[
            valid_ppsf,
            "price"
        ]
        /
        df.loc[
            valid_ppsf,
            "area"
        ]
    )

    actual_ppsf = (
        df.loc[
            valid_ppsf,
            "pricePerSqFt"
        ]
    )

    ppsf_mismatch = (
        np.abs(
            actual_ppsf
            - expected_ppsf
        )
        > 0.01
    ).sum()

    print(
        f"Price/Sq Ft mismatches: "
        f"{ppsf_mismatch:,}"
    )

    if ppsf_mismatch != 0:
        raise ValueError(
            "Price per square foot validation failed."
        )

    # --------------------------------------------------------
    # Log price
    # --------------------------------------------------------

    valid_log = (
        df["logPrice"]
        .notna()
    )

    expected_log = np.log(
        df.loc[
            valid_log,
            "price"
        ]
    )

    actual_log = (
        df.loc[
            valid_log,
            "logPrice"
        ]
    )

    log_mismatch = (
        np.abs(
            actual_log
            - expected_log
        )
        > 1e-10
    ).sum()

    print(
        f"Log price mismatches: "
        f"{log_mismatch:,}"
    )

    if log_mismatch != 0:
        raise ValueError(
            "Log price validation failed."
        )

    # --------------------------------------------------------
    # Zestimate
    # --------------------------------------------------------

    zestimate_count = (
        df["zestimate"]
        .notna()
        .sum()
    )

    gap_count = (
        df["zestimate_gap"]
        .notna()
        .sum()
    )

    print(
        f"Listings with Zestimate: "
        f"{zestimate_count:,}"
    )

    print(
        f"Listings with Zestimate gap: "
        f"{gap_count:,}"
    )

    if zestimate_count != gap_count:
        raise ValueError(
            "Zestimate gap coverage does not "
            "match Zestimate coverage."
        )

    # --------------------------------------------------------
    # Zestimate gap validation
    # --------------------------------------------------------

    valid_gap = (
        df["zestimate_gap"]
        .notna()
    )

    expected_gap = (
        df.loc[
            valid_gap,
            "zestimate"
        ]
        -
        df.loc[
            valid_gap,
            "price"
        ]
    )

    actual_gap = (
        df.loc[
            valid_gap,
            "zestimate_gap"
        ]
    )

    gap_mismatch = (
        np.abs(
            expected_gap
            - actual_gap
        )
        > 0.01
    ).sum()

    print(
        f"Zestimate gap mismatches: "
        f"{gap_mismatch:,}"
    )

    if gap_mismatch != 0:
        raise ValueError(
            "Zestimate gap validation failed."
        )

    # --------------------------------------------------------
    # Zestimate gap percentage validation
    # --------------------------------------------------------

    valid_gap_pct = (
        df["zestimate_gap_pct"]
        .notna()
    )

    expected_gap_pct = (
        (
            df.loc[
                valid_gap_pct,
                "zestimate"
            ]
            -
            df.loc[
                valid_gap_pct,
                "price"
            ]
        )
        /
        df.loc[
            valid_gap_pct,
            "price"
        ]
    )

    actual_gap_pct = (
        df.loc[
            valid_gap_pct,
            "zestimate_gap_pct"
        ]
    )

    gap_pct_mismatch = (
        np.abs(
            expected_gap_pct
            - actual_gap_pct
        )
        > 1e-10
    ).sum()

    print(
        f"Zestimate gap % mismatches: "
        f"{gap_pct_mismatch:,}"
    )

    if gap_pct_mismatch != 0:
        raise ValueError(
            "Zestimate gap percentage "
            "validation failed."
        )

    # --------------------------------------------------------
    # Opportunities
    # --------------------------------------------------------

    opportunity_count = (
        df["opportunity_status"]
        .eq("Potential Opportunity")
        .sum()
    )

    print(
        f"Potential opportunities: "
        f"{opportunity_count:,}"
    )

    if opportunity_count != 16:
        raise ValueError(
            f"Expected 16 potential opportunities, "
            f"found {opportunity_count}."
        )

    # --------------------------------------------------------
    # Deal Score
    # --------------------------------------------------------

    deal_score_count = (
        df["deal_score"]
        .notna()
        .sum()
    )

    print(
        f"Deal Score records: "
        f"{deal_score_count:,}"
    )

    if deal_score_count != zestimate_count:
        raise ValueError(
            "Deal Score coverage does not "
            "match Zestimate coverage."
        )

    # --------------------------------------------------------
    # Deal Score range
    # --------------------------------------------------------

    if deal_score_count > 0:

        minimum_score = (
            df["deal_score"]
            .min()
        )

        maximum_score = (
            df["deal_score"]
            .max()
        )

        print(
            f"Deal Score range: "
            f"{minimum_score:.2f} - "
            f"{maximum_score:.2f}"
        )

        if minimum_score < 0:
            raise ValueError(
                "Deal Score contains values below 0."
            )

        if maximum_score > 100:
            raise ValueError(
                "Deal Score contains values above 100."
            )

    # --------------------------------------------------------
    # Price segments
    # --------------------------------------------------------

    segment_count = (
        df["price_segment"]
        .notna()
        .sum()
    )

    print(
        f"Price segment records: "
        f"{segment_count:,}"
    )

    if segment_count != EXPECTED_ROWS:
        raise ValueError(
            "Price segments do not cover "
            "all properties."
        )

    # --------------------------------------------------------
    # Price segment reconciliation
    # --------------------------------------------------------

    segment_counts = (
        df["price_segment"]
        .value_counts(
            dropna=False
        )
    )

    segment_total = (
        segment_counts.sum()
    )

    print(
        f"Price segment total: "
        f"{segment_total:,}"
    )

    if segment_total != EXPECTED_ROWS:
        raise ValueError(
            "Price segment counts do not "
            "reconcile to 810."
        )

    # --------------------------------------------------------
    # ZIP sample status
    # --------------------------------------------------------

    zip_status_count = (
        df["zip_sample_status"]
        .notna()
        .sum()
    )

    print(
        f"ZIP sample-status records: "
        f"{zip_status_count:,}"
    )

    if zip_status_count != EXPECTED_ROWS:
        raise ValueError(
            "ZIP sample status does not "
            "cover all properties."
        )

    # --------------------------------------------------------
    # ZIP sample-size rule
    # --------------------------------------------------------

    invalid_zip_status = (
        (
            df["zip_listing_count"]
            >= MIN_ZIP_SAMPLE_SIZE
        )
        &
        (
            df["zip_sample_status"]
            != "Sufficient Sample"
        )
    ) | (
        (
            df["zip_listing_count"]
            < MIN_ZIP_SAMPLE_SIZE
        )
        &
        (
            df["zip_sample_status"]
            != "Sparse Sample"
        )
    )

    invalid_zip_status_count = (
        invalid_zip_status.sum()
    )

    print(
        f"Invalid ZIP sample statuses: "
        f"{invalid_zip_status_count:,}"
    )

    if invalid_zip_status_count != 0:
        raise ValueError(
            "ZIP sample status validation failed."
        )

    # --------------------------------------------------------
    # ML predictors
    # --------------------------------------------------------

    missing_ml_columns = [
        column
        for column in ML_PREDICTORS
        if column not in df.columns
    ]

    if missing_ml_columns:
        raise ValueError(
            "Missing ML predictor columns: "
            + ", ".join(
                missing_ml_columns
            )
        )

    print()
    print(
        "ML predictor columns validated."
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print()
    print(
        "All engineered feature validation "
        "checks passed."
    )

    print()

    return df


# ============================================================
# CREATE METHODOLOGY DOCUMENT
# ============================================================

def create_methodology_document(df):
    """Create feature-engineering methodology documentation."""

    zestimate_count = (
        df["zestimate"]
        .notna()
        .sum()
    )

    opportunity_count = (
        df["opportunity_status"]
        .eq("Potential Opportunity")
        .sum()
    )

    sufficient_zip_count = (
        df.loc[
            df["zip_sample_status"]
            == "Sufficient Sample",
            "zipcode",
        ]
        .nunique()
    )

    sparse_zip_count = (
        df.loc[
            df["zip_sample_status"]
            == "Sparse Sample",
            "zipcode",
        ]
        .nunique()
    )

    methodology = f"""
# Part 5 — Feature Engineering Methodology

## Purpose

This stage creates and validates the final engineered features used
throughout the San Antonio Zillow housing analysis.

The feature-engineering process preserves the complete
{len(df):,}-property analytical population.

## Source Dataset

Input:

`data/processed/San_Antonio_Zillow_Analysis.csv`

The source dataset contains:

- 810 properties
- 67 source columns

Output:

`data/processed/San_Antonio_Zillow_Feature_Engineered.csv`

## Population

- Total properties: {len(df):,}
- Unique zpid: {df["zpid"].nunique():,}
- Duplicate zpid: {df["zpid"].duplicated().sum():,}

No observations were removed during feature engineering.

## ZIP Code Standardization

The source Zillow field:

`addressZipcode`

is standardized into:

`zipcode`

The original `addressZipcode` field is preserved.

ZIP codes with at least 10 listings are classified as:

`Sufficient Sample`

ZIP codes with fewer than 10 listings are classified as:

`Sparse Sample`

Sparse ZIPs remain in the dataset and are not removed.

## Engineered Features

### Price Per Square Foot

Calculated as:

`Listing Price / Living Area`

This is used for market analysis and property comparison.

It is **not used as an ML predictor** because it directly contains
the listing price target.

### Log Price

Calculated as:

`ln(Listing Price)`

This is the dependent variable used for the finalized price
prediction model.

### Zestimate Gap

Calculated as:

`Zestimate - Listing Price`

Positive values indicate that the Zestimate is above the asking price.

### Zestimate Gap Percentage

Calculated as:

`(Zestimate - Listing Price) / Listing Price`

This measures the relative difference between Zillow's Zestimate
and the listing price.

### Opportunity Status

Properties are classified as:

- Potential Opportunity — Zestimate exceeds listing price
- Above Zestimate — Zestimate does not exceed listing price
- Null / missing Zestimate — Zestimate unavailable

Properties without Zestimate data remain missing rather than being
imputed.

### Deal Score

Deal Score is a percentile-based relative ranking of
`zestimate_gap_pct`.

Higher scores indicate properties with larger positive Zestimate gaps
relative to other properties with available Zestimate data.

Deal Score is an analytical ranking metric, not a guaranteed measure
of investment return.

### Price Segments

Properties are divided into:

- Under $150K
- $150K-$300K
- $300K-$500K
- $500K-$1M
- $1M-$2M
- $2M+

## Machine-Learning Feature Separation

The finalized ML predictors are:

1. area
2. beds
3. baths
4. lotAreaSqFt
5. taxAssessedValue
6. daysOnZillow

Target:

`logPrice`

## Target Leakage Prevention

The following variables are intentionally excluded from the ML
predictor set because they contain or are directly derived from
listing price:

- pricePerSqFt
- zestimate_gap
- zestimate_gap_pct
- deal_score
- opportunity_status
- price_segment

This separation allows the project to use these variables for
business analysis without contaminating the independent price
prediction model.

## Coverage

- Zestimate records: {zestimate_count:,}
- Potential opportunities: {opportunity_count:,}
- Sufficient-sample ZIP codes: {sufficient_zip_count:,}
- Sparse ZIP codes: {sparse_zip_count:,}

## Data Treatment

Feature engineering does not automatically remove statistical
outliers.

Legitimate high-value, low-value, unusually large, or otherwise
unusual properties remain part of the analytical dataset.

Missing Zestimate and tax-assessed values remain missing.

## Validation

The final feature-engineered dataset is required to maintain:

- 810 properties
- 810 unique zpids
- 0 duplicate zpids
- no missing ZIP codes
- valid positive listing prices
- valid price-per-square-foot calculations
- valid log-price calculations
- Zestimate-gap coverage matching Zestimate availability
- 16 potential Zestimate opportunities
- Deal Score coverage matching Zestimate coverage
- Deal Score values between 0 and 100
- complete price segmentation
- complete ZIP sample classification

All validation checks must pass before this dataset is used
downstream.
"""

    return methodology


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(
    df,
    feature_summary,
    feature_dictionary,
    methodology,
):
    """Save engineered dataset and supporting documentation."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    feature_summary.to_csv(
        OUTPUT_DIR
        / "feature_summary.csv",
        index=False
    )

    feature_dictionary.to_csv(
        OUTPUT_DIR
        / "feature_dictionary.csv",
        index=False
    )

    with open(
        OUTPUT_DIR
        / "FEATURE_ENGINEERING_METHODOLOGY.md",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            methodology
        )

    print("=" * 70)
    print("FEATURE ENGINEERING OUTPUTS SAVED")
    print("=" * 70)
    print()

    print(
        "Feature-engineered dataset:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print()

    print("Reports:")

    print(
        "  - feature_summary.csv"
    )

    print(
        "  - feature_dictionary.csv"
    )

    print(
        "  - FEATURE_ENGINEERING_METHODOLOGY.md"
    )

    print()


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

def print_final_summary(df):
    """Print final Part 5 summary."""

    print("=" * 70)
    print(
        "PART 5 FEATURE ENGINEERING COMPLETE"
    )
    print("=" * 70)
    print()

    print(
        f"Final properties: "
        f"{len(df):,}"
    )

    print(
        f"Unique properties: "
        f"{df['zpid'].nunique():,}"
    )

    zestimate_coverage = (
        df["zestimate"]
        .notna()
        .mean()
        * 100
    )

    opportunity_count = (
        df["opportunity_status"]
        .eq("Potential Opportunity")
        .sum()
    )

    print(
        f"Zestimate coverage: "
        f"{zestimate_coverage:.2f}%"
    )

    print(
        f"Potential opportunities: "
        f"{opportunity_count:,}"
    )

    print()

    print("ML predictors:")

    for predictor in ML_PREDICTORS:

        print(
            f"  - {predictor}"
        )

    print()

    print("ML target:")

    print(
        f"  - {TARGET_VARIABLE}"
    )

    print()

    print(
        "Business-analysis features:"
    )

    for feature in BUSINESS_ANALYSIS_FEATURES:

        print(
            f"  - {feature}"
        )

    print()

    print(
        "The analysis-ready population was "
        "preserved at 810 properties."
    )

    print(
        "The source ZIP field addressZipcode "
        "was standardized to zipcode."
    )

    print(
        "Target-leaking business metrics are "
        "separated from ML predictors."
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Validate source structure
    # --------------------------------------------------------

    df = validate_required_columns(
        df
    )

    # --------------------------------------------------------
    # Create standardized ZIP
    # --------------------------------------------------------

    df = create_zipcode(
        df
    )

    # --------------------------------------------------------
    # Normalize numeric fields
    # --------------------------------------------------------

    df = normalize_numeric_columns(
        df
    )

    # --------------------------------------------------------
    # Validate population
    # --------------------------------------------------------

    df = validate_dataset_integrity(
        df
    )

    # --------------------------------------------------------
    # Engineer features
    # --------------------------------------------------------

    df = create_price_per_sqft(
        df
    )

    df = create_log_price(
        df
    )

    df = create_zestimate_features(
        df
    )

    df = create_deal_score(
        df
    )

    df = create_price_segments(
        df
    )

    df = create_zip_features(
        df
    )

    df = create_price_change_flag(
        df
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    df = validate_engineered_features(
        df
    )

    # --------------------------------------------------------
    # Documentation
    # --------------------------------------------------------

    feature_summary = (
        create_feature_summary(
            df
        )
    )

    feature_dictionary = (
        create_feature_dictionary()
    )

    methodology = (
        create_methodology_document(
            df
        )
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    save_reports(
        df,
        feature_summary,
        feature_dictionary,
        methodology,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_final_summary(
        df
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()


# ============================================================

INPUT_FILE = Path(
    "data/processed/San_Antonio_Zillow_Analysis.csv"
)

OUTPUT_FILE = Path(
    "data/processed/San_Antonio_Zillow_Feature_Engineered.csv"
)

OUTPUT_DIR = Path(
    "reports/features"
)

MIN_ZIP_SAMPLE_SIZE = 10

EXPECTED_ROWS = 810


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

ML_PREDICTORS = [
    "area",
    "beds",
    "baths",
    "lotAreaSqFt",
    "taxAssessedValue",
    "daysOnZillow",
]

TARGET_VARIABLE = "logPrice"

BUSINESS_ANALYSIS_FEATURES = [
    "pricePerSqFt",
    "zestimate_gap",
    "zestimate_gap_pct",
    "deal_score",
    "opportunity_status",
    "price_segment",
    "zip_listing_count",
    "zip_sample_status",
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    """Load the cleaned analytical dataset."""

    print("=" * 70)
    print("PART 5 — FEATURE ENGINEERING")
    print("=" * 70)
    print()
    print("SAN ANTONIO ZILLOW FEATURE ENGINEERING")
    print()
    print(f"Input file: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df):,}")
    print()

    return df


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

def validate_required_columns(df):
    """Confirm that required source columns exist."""

    required_columns = [
        "zpid",
        "price",
        "zipcode",
        "beds",
        "baths",
        "area",
        "homeType",
        "daysOnZillow",
        "zestimate",
        "taxAssessedValue",
        "lotAreaSqFt",
        "pricePerSqFt",
        "logPrice",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing: "
            + ", ".join(missing_columns)
        )

    print("Required source columns validated.")
    print()


# ============================================================
# NORMALIZE NUMERIC COLUMNS
# ============================================================

def normalize_numeric_columns(df):
    """
    Convert relevant fields to numeric values.

    Zillow listing prices can contain formats such as:
        $269,000
        $151K
        $1.5M
    """

    numeric_columns = [
        "beds",
        "baths",
        "area",
        "daysOnZillow",
        "zestimate",
        "taxAssessedValue",
        "lotAreaSqFt",
        "pricePerSqFt",
        "logPrice",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Parse Zillow-formatted listing prices
    # --------------------------------------------------------

    original_price = df["price"].copy()

    numeric_price = pd.to_numeric(
        original_price,
        errors="coerce"
    )

    def parse_price(value):
        """Convert Zillow-formatted price strings to numeric."""

        if pd.isna(value):
            return np.nan

        text = str(value).strip().upper()

        if text == "":
            return np.nan

        text = (
            text
            .replace("$", "")
            .replace(",", "")
            .strip()
        )

        try:
            if text.endswith("M"):
                return float(text[:-1]) * 1_000_000

            if text.endswith("K"):
                return float(text[:-1]) * 1_000

            return float(text)

        except (ValueError, TypeError):
            return np.nan

    parsed_price = original_price.apply(
        parse_price
    )

    df["price"] = numeric_price.fillna(
        parsed_price
    )

    print("Numeric fields normalized.")
    print()

    return df


# ============================================================
# VALIDATE DATASET INTEGRITY
# ============================================================

def validate_dataset_integrity(df):
    """Validate population size and property identifiers."""

    print("=" * 70)
    print("DATASET INTEGRITY VALIDATION")
    print("=" * 70)
    print()

    row_count = len(df)

    unique_zpid = df["zpid"].nunique()

    duplicate_zpid = (
        df["zpid"]
        .duplicated()
        .sum()
    )

    print(f"Rows: {row_count:,}")
    print(f"Unique zpid: {unique_zpid:,}")
    print(f"Duplicate zpid: {duplicate_zpid:,}")

    if row_count != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {row_count}."
        )

    if unique_zpid != EXPECTED_ROWS:
        raise ValueError(
            "The number of unique properties does not "
            "equal the expected 810-property population."
        )

    if duplicate_zpid != 0:
        raise ValueError(
            "Duplicate zpid values detected."
        )

    print()
    print("Dataset integrity validation passed.")
    print()

    return df


# ============================================================
# CREATE / VALIDATE PRICE PER SQUARE FOOT
# ============================================================

def create_price_per_sqft(df):
    """
    Calculate listing price per square foot.

    This is a business-analysis metric and is NOT an ML predictor
    because it directly contains the target listing price.
    """

    valid = (
        df["price"].notna()
        & df["area"].notna()
        & (df["area"] > 0)
    )

    df["pricePerSqFt"] = np.where(
        valid,
        df["price"] / df["area"],
        np.nan
    )

    print("Created pricePerSqFt.")
    print()

    return df


# ============================================================
# CREATE / VALIDATE LOG PRICE
# ============================================================

def create_log_price(df):
    """
    Create natural-log transformed listing price.

    logPrice is the machine-learning target variable.
    """

    valid = (
        df["price"].notna()
        & (df["price"] > 0)
    )

    df["logPrice"] = np.where(
        valid,
        np.log(df["price"]),
        np.nan
    )

    print("Created logPrice.")
    print()

    return df


# ============================================================
# CREATE ZESTIMATE FEATURES
# ============================================================

def create_zestimate_features(df):
    """
    Create Zestimate comparison features.

    zestimate_gap:
        Zestimate minus listing price.

    zestimate_gap_pct:
        Percentage difference between Zestimate
        and listing price.

    opportunity_status:
        Potential Opportunity when Zestimate exceeds
        listing price.
    """

    valid_zestimate = (
        df["zestimate"].notna()
        & df["price"].notna()
        & (df["price"] > 0)
    )

    df["zestimate_gap"] = np.where(
        valid_zestimate,
        df["zestimate"] - df["price"],
        np.nan
    )

    df["zestimate_gap_pct"] = np.where(
        valid_zestimate,
        (
            (df["zestimate"] - df["price"])
            / df["price"]
        ),
        np.nan
    )

    df["opportunity_status"] = np.select(
        [
            df["zestimate"].isna(),
            df["zestimate_gap"] > 0,
        ],
        [
            "Null / missing Zestimate",
            "Potential Opportunity",
        ],
        default="Above Zestimate",
    )

    print("Created Zestimate gap features.")
    print()

    return df


# ============================================================
# CREATE DEAL SCORE
# ============================================================

def create_deal_score(df):
    """
    Create a relative Deal Score based on Zestimate
    gap percentage.

    Deal Score:
        100 * percentile rank of zestimate_gap_pct

    Higher values indicate that a property has a larger
    positive Zestimate gap relative to other properties
    with available Zestimate data.

    This is a business-analysis metric and must not be used
    as an ML predictor.
    """

    df["deal_score"] = np.nan

    valid = (
        df["zestimate_gap_pct"]
        .notna()
    )

    if valid.sum() > 0:
        df.loc[valid, "deal_score"] = (
            df.loc[
                valid,
                "zestimate_gap_pct"
            ]
            .rank(
                method="min",
                pct=True
            )
            * 100
        )

    print("Created Deal Score.")
    print()

    return df


# ============================================================
# CREATE PRICE SEGMENTS
# ============================================================

def create_price_segments(df):
    """Create standardized listing-price segments."""

    conditions = [
        df["price"] < 150_000,
        df["price"] < 300_000,
        df["price"] < 500_000,
        df["price"] < 1_000_000,
        df["price"] < 2_000_000,
        df["price"] >= 2_000_000,
    ]

    choices = [
        "Under $150K",
        "$150K-$300K",
        "$300K-$500K",
        "$500K-$1M",
        "$1M-$2M",
        "$2M+",
    ]

    df["price_segment"] = np.select(
        conditions,
        choices,
        default="Unclassified",
    )

    print("Created price segments.")
    print()

    return df


# ============================================================
# CREATE ZIP FEATURES
# ============================================================

def create_zip_features(df):
    """
    Create ZIP-level sample-size features.

    ZIPs with fewer than 10 listings are classified
    as sparse.

    Sparse ZIPs remain in the dataset and are not removed.
    """

    zip_counts = (
        df.groupby("zipcode")["zpid"]
        .transform("count")
    )

    df["zip_listing_count"] = zip_counts

    df["zip_sample_status"] = np.where(
        df["zip_listing_count"]
        >= MIN_ZIP_SAMPLE_SIZE,
        "Sufficient Sample",
        "Sparse Sample",
    )

    print("Created ZIP sample-size features.")
    print()

    return df


# ============================================================
# CREATE PRICE CHANGE FLAG
# ============================================================

def create_price_change_flag(df):
    """Create a standardized price-change indicator."""

    if "priceChange" in df.columns:

        df["hasPriceChange"] = np.where(
            (
                df["priceChange"].notna()
                & (df["priceChange"] != 0)
            ),
            1,
            0,
        )

    else:

        df["hasPriceChange"] = 0

    print("Created hasPriceChange.")
    print()

    return df


# ============================================================
# CREATE FEATURE SUMMARY
# ============================================================

def create_feature_summary(df):
    """Create a summary of engineered features."""

    summary = []

    feature_columns = [
        "pricePerSqFt",
        "logPrice",
        "zestimate_gap",
        "zestimate_gap_pct",
        "deal_score",
        "price_segment",
        "zip_listing_count",
        "zip_sample_status",
        "hasPriceChange",
    ]

    for feature in feature_columns:

        if feature not in df.columns:
            continue

        series = df[feature]

        summary.append(
            {
                "feature": feature,
                "data_type": str(series.dtype),
                "non_null_count": (
                    series.notna().sum()
                ),
                "missing_count": (
                    series.isna().sum()
                ),
                "missing_percentage": (
                    series.isna().mean() * 100
                ),
                "unique_values": (
                    series.nunique(
                        dropna=True
                    )
                ),
            }
        )

    return pd.DataFrame(summary)


# ============================================================
# CREATE FEATURE DICTIONARY
# ============================================================

def create_feature_dictionary():
    """Create formal documentation for project features."""

    definitions = [
        {
            "feature": "pricePerSqFt",
            "definition": (
                "Listing price divided by living area "
                "in square feet."
            ),
            "category": "Business Analysis",
            "ml_safe": "No",
            "reason": (
                "Contains listing price and therefore "
                "introduces target leakage."
            ),
        },
        {
            "feature": "logPrice",
            "definition": (
                "Natural logarithm of listing price."
            ),
            "category": "ML Target",
            "ml_safe": "Target",
            "reason": (
                "This is the dependent variable used "
                "for price prediction."
            ),
        },
        {
            "feature": "zestimate_gap",
            "definition": (
                "Zestimate minus listing price."
            ),
            "category": "Business Analysis",
            "ml_safe": "No",
            "reason": (
                "Directly contains the target "
                "listing price."
            ),
        },
        {
            "feature": "zestimate_gap_pct",
            "definition": (
                "(Zestimate minus listing price) "
                "divided by listing price."
            ),
            "category": "Business Analysis",
            "ml_safe": "No",
            "reason": (
                "Directly contains the target "
                "listing price."
            ),
        },
        {
            "feature": "deal_score",
            "definition": (
                "Percentile-based score representing "
                "relative Zestimate gap percentage."
            ),
            "category": "Business Analysis",
            "ml_safe": "No",
            "reason": (
                "Derived from Zestimate gap percentage "
                "and listing price."
            ),
        },
        {
            "feature": "opportunity_status",
            "definition": (
                "Classification based on whether "
                "Zestimate exceeds listing price."
            ),
            "category": "Business Analysis",
            "ml_safe": "No",
            "reason": (
                "Uses listing price and Zestimate."
            ),
        },
        {
            "feature": "price_segment",
            "definition": (
                "Categorical grouping of properties "
                "by listing price."
            ),
            "category": "Business Analysis",
            "ml_safe": "No",
            "reason": (
                "Derived directly from listing price."
            ),
        },
        {
            "feature": "zip_listing_count",
            "definition": (
                "Number of listings represented "
                "in the same ZIP code."
            ),
            "category": "Market Context",
            "ml_safe": "Potential",
            "reason": (
                "Contextual feature; not used in "
                "the finalized ML model."
            ),
        },
        {
            "feature": "zip_sample_status",
            "definition": (
                "Indicates whether a ZIP has at "
                "least 10 listings."
            ),
            "category": "Market Context",
            "ml_safe": "Potential",
            "reason": (
                "Used for analytical reliability "
                "rather than the finalized ML model."
            ),
        },
        {
            "feature": "area",
            "definition": (
                "Living area in square feet."
            ),
            "category": "ML Predictor",
            "ml_safe": "Yes",
            "reason": (
                "Property characteristic available "
                "independently of target price."
            ),
        },
        {
            "feature": "beds",
            "definition": (
                "Number of bedrooms."
            ),
            "category": "ML Predictor",
            "ml_safe": "Yes",
            "reason": (
                "Property characteristic available "
                "independently of target price."
            ),
        },
        {
            "feature": "baths",
            "definition": (
                "Number of bathrooms."
            ),
            "category": "ML Predictor",
            "ml_safe": "Yes",
            "reason": (
                "Property characteristic available "
                "independently of target price."
            ),
        },
        {
            "feature": "lotAreaSqFt",
            "definition": (
                "Lot area standardized to square feet."
            ),
            "category": "ML Predictor",
            "ml_safe": "Yes",
            "reason": (
                "Property characteristic available "
                "independently of target price."
            ),
        },
        {
            "feature": "taxAssessedValue",
            "definition": (
                "Tax-assessed value reported in "
                "the Zillow data."
            ),
            "category": "ML Predictor",
            "ml_safe": "Yes",
            "reason": (
                "Independent property valuation feature "
                "retained in the finalized model."
            ),
        },
        {
            "feature": "daysOnZillow",
            "definition": (
                "Number of days the property has "
                "been listed on Zillow."
            ),
            "category": "ML Predictor",
            "ml_safe": "Yes",
            "reason": (
                "Listing-duration characteristic retained "
                "in the finalized model."
            ),
        },
    ]

    return pd.DataFrame(
        definitions
    )


# ============================================================
# VALIDATE ENGINEERED FEATURES
# ============================================================

def validate_engineered_features(df):
    """Run final validation checks on engineered features."""

    print("=" * 70)
    print("ENGINEERED FEATURE VALIDATION")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Population
    # --------------------------------------------------------

    print(
        f"Total rows: {len(df):,}"
    )

    print(
        f"Unique zpids: "
        f"{df['zpid'].nunique():,}"
    )

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            "Feature engineering changed "
            "the dataset population."
        )

    if df["zpid"].nunique() != EXPECTED_ROWS:
        raise ValueError(
            "Duplicate or missing zpid values detected."
        )

    # --------------------------------------------------------
    # Duplicate identifiers
    # --------------------------------------------------------

    duplicate_count = (
        df["zpid"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate zpids: {duplicate_count:,}"
    )

    if duplicate_count != 0:
        raise ValueError(
            "Duplicate zpid values detected."
        )

    # --------------------------------------------------------
    # Price
    # --------------------------------------------------------

    invalid_price = (
        df["price"].isna()
        | (df["price"] <= 0)
    ).sum()

    print(
        f"Invalid prices: {invalid_price:,}"
    )

    if invalid_price != 0:
        raise ValueError(
            "Invalid listing prices detected."
        )

    # --------------------------------------------------------
    # Price per square foot
    # --------------------------------------------------------

    valid_ppsf = (
        df["pricePerSqFt"].notna()
        & df["area"].notna()
        & (df["area"] > 0)
    )

    expected_ppsf = (
        df.loc[valid_ppsf, "price"]
        / df.loc[valid_ppsf, "area"]
    )

    actual_ppsf = (
        df.loc[valid_ppsf, "pricePerSqFt"]
    )

    ppsf_mismatch = (
        np.abs(
            actual_ppsf - expected_ppsf
        ) > 0.01
    ).sum()

    print(
        f"Price/Sq Ft mismatches: "
        f"{ppsf_mismatch:,}"
    )

    if ppsf_mismatch != 0:
        raise ValueError(
            "Price per square foot validation failed."
        )

    # --------------------------------------------------------
    # Log price
    # --------------------------------------------------------

    valid_log = (
        df["logPrice"].notna()
    )

    expected_log = np.log(
        df.loc[valid_log, "price"]
    )

    actual_log = (
        df.loc[valid_log, "logPrice"]
    )

    log_mismatch = (
        np.abs(
            actual_log - expected_log
        ) > 1e-10
    ).sum()

    print(
        f"Log price mismatches: "
        f"{log_mismatch:,}"
    )

    if log_mismatch != 0:
        raise ValueError(
            "Log price validation failed."
        )

    # --------------------------------------------------------
    # Zestimate
    # --------------------------------------------------------

    zestimate_count = (
        df["zestimate"]
        .notna()
        .sum()
    )

    gap_count = (
        df["zestimate_gap"]
        .notna()
        .sum()
    )

    print(
        f"Listings with Zestimate: "
        f"{zestimate_count:,}"
    )

    print(
        f"Listings with Zestimate gap: "
        f"{gap_count:,}"
    )

    if zestimate_count != gap_count:
        raise ValueError(
            "Zestimate gap coverage does not "
            "match Zestimate coverage."
        )

    # --------------------------------------------------------
    # Zestimate gap validation
    # --------------------------------------------------------

    valid_gap = (
        df["zestimate_gap"].notna()
    )

    expected_gap = (
        df.loc[valid_gap, "zestimate"]
        - df.loc[valid_gap, "price"]
    )

    actual_gap = (
        df.loc[valid_gap, "zestimate_gap"]
    )

    gap_mismatch = (
        np.abs(
            expected_gap - actual_gap
        ) > 0.01
    ).sum()

    print(
        f"Zestimate gap mismatches: "
        f"{gap_mismatch:,}"
    )

    if gap_mismatch != 0:
        raise ValueError(
            "Zestimate gap validation failed."
        )

    # --------------------------------------------------------
    # Zestimate gap percentage validation
    # --------------------------------------------------------

    valid_gap_pct = (
        df["zestimate_gap_pct"]
        .notna()
    )

    expected_gap_pct = (
        (
            df.loc[
                valid_gap_pct,
                "zestimate"
            ]
            - df.loc[
                valid_gap_pct,
                "price"
            ]
        )
        / df.loc[
            valid_gap_pct,
            "price"
        ]
    )

    actual_gap_pct = (
        df.loc[
            valid_gap_pct,
            "zestimate_gap_pct"
        ]
    )

    gap_pct_mismatch = (
        np.abs(
            expected_gap_pct
            - actual_gap_pct
        ) > 1e-10
    ).sum()

    print(
        f"Zestimate gap % mismatches: "
        f"{gap_pct_mismatch:,}"
    )

    if gap_pct_mismatch != 0:
        raise ValueError(
            "Zestimate gap percentage validation failed."
        )

    # --------------------------------------------------------
    # Opportunities
    # --------------------------------------------------------

    opportunity_count = (
        df["opportunity_status"]
        .eq("Potential Opportunity")
        .sum()
    )

    print(
        f"Potential opportunities: "
        f"{opportunity_count:,}"
    )

    if opportunity_count != 16:
        raise ValueError(
            f"Expected 16 potential opportunities, "
            f"found {opportunity_count}."
        )

    # --------------------------------------------------------
    # Deal Score
    # --------------------------------------------------------

    deal_score_count = (
        df["deal_score"]
        .notna()
        .sum()
    )

    print(
        f"Deal Score records: "
        f"{deal_score_count:,}"
    )

    if deal_score_count != zestimate_count:
        raise ValueError(
            "Deal Score coverage does not "
            "match Zestimate coverage."
        )

    # --------------------------------------------------------
    # Deal Score range
    # --------------------------------------------------------

    if deal_score_count > 0:

        minimum_score = (
            df["deal_score"]
            .min()
        )

        maximum_score = (
            df["deal_score"]
            .max()
        )

        print(
            f"Deal Score range: "
            f"{minimum_score:.2f} - "
            f"{maximum_score:.2f}"
        )

        if minimum_score < 0:
            raise ValueError(
                "Deal Score contains values below 0."
            )

        if maximum_score > 100:
            raise ValueError(
                "Deal Score contains values above 100."
            )

    # --------------------------------------------------------
    # Price segments
    # --------------------------------------------------------

    segment_count = (
        df["price_segment"]
        .notna()
        .sum()
    )

    print(
        f"Price segment records: "
        f"{segment_count:,}"
    )

    if segment_count != EXPECTED_ROWS:
        raise ValueError(
            "Price segments do not cover "
            "all properties."
        )

    # --------------------------------------------------------
    # ZIP sample status
    # --------------------------------------------------------

    zip_status_count = (
        df["zip_sample_status"]
        .notna()
        .sum()
    )

    print(
        f"ZIP sample-status records: "
        f"{zip_status_count:,}"
    )

    if zip_status_count != EXPECTED_ROWS:
        raise ValueError(
            "ZIP sample status does not cover "
            "all properties."
        )

    # --------------------------------------------------------
    # ML predictors
    # --------------------------------------------------------

    missing_ml_columns = [
        column
        for column in ML_PREDICTORS
        if column not in df.columns
    ]

    if missing_ml_columns:
        raise ValueError(
            "Missing ML predictor columns: "
            + ", ".join(missing_ml_columns)
        )

    print()
    print(
        "ML predictor columns validated."
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print()
    print(
        "All engineered feature validation "
        "checks passed."
    )
    print()

    return df


# ============================================================
# CREATE METHODOLOGY DOCUMENT
# ============================================================

def create_methodology_document(df):
    """Create feature-engineering methodology documentation."""

    zestimate_count = (
        df["zestimate"]
        .notna()
        .sum()
    )

    opportunity_count = (
        df["opportunity_status"]
        .eq("Potential Opportunity")
        .sum()
    )

    sufficient_zip_count = (
        df.loc[
            df["zip_sample_status"]
            == "Sufficient Sample",
            "zipcode",
        ]
        .nunique()
    )

    sparse_zip_count = (
        df.loc[
            df["zip_sample_status"]
            == "Sparse Sample",
            "zipcode",
        ]
        .nunique()
    )

    methodology = f"""# Part 5 — Feature Engineering Methodology

## Purpose

This stage creates and validates the final engineered features used
throughout the San Antonio Zillow housing analysis.

The feature-engineering process preserves the complete
{len(df):,}-property analytical population.

## Source Dataset

Input:

`data/processed/San_Antonio_Zillow_Analysis.csv`

Output:

`data/processed/San_Antonio_Zillow_Feature_Engineered.csv`

## Population

- Total properties: {len(df):,}
- Unique zpid: {df["zpid"].nunique():,}
- Duplicate zpid: {df["zpid"].duplicated().sum():,}

No observations were removed during feature engineering.

## Engineered Features

### Price Per Square Foot

Calculated as:

`Listing Price / Living Area`

This is used for market analysis and property comparison.

It is **not used as an ML predictor** because it directly contains
the listing price target.

### Log Price

Calculated as:

`ln(Listing Price)`

This is the dependent variable used for the finalized price
prediction model.

### Zestimate Gap

Calculated as:

`Zestimate - Listing Price`

Positive values indicate that the Zestimate is above the asking price.

### Zestimate Gap Percentage

Calculated as:

`(Zestimate - Listing Price) / Listing Price`

This measures the relative difference between Zillow's Zestimate
and the listing price.

### Opportunity Status

Properties are classified as:

- Potential Opportunity — Zestimate exceeds listing price
- Above Zestimate — Zestimate does not exceed listing price
- Null / missing Zestimate — Zestimate unavailable

Properties without Zestimate data remain missing rather than being
imputed.

### Deal Score

Deal Score is a percentile-based relative ranking of
`zestimate_gap_pct`.

Higher scores indicate properties with larger positive Zestimate gaps
relative to other properties with available Zestimate data.

Deal Score is an analytical ranking metric, not a guaranteed measure
of investment return.

### Price Segments

Properties are divided into:

- Under $150K
- $150K-$300K
- $300K-$500K
- $500K-$1M
- $1M-$2M
- $2M+

### ZIP Sample Status

ZIP codes with at least {MIN_ZIP_SAMPLE_SIZE} listings are classified
as:

`Sufficient Sample`

ZIP codes with fewer than {MIN_ZIP_SAMPLE_SIZE} listings are classified
as:

`Sparse Sample`

Sparse ZIPs are retained in the dataset and are not deleted.

## Machine-Learning Feature Separation

The finalized ML predictors are:

1. area
2. beds
3. baths
4. lotAreaSqFt
5. taxAssessedValue
6. daysOnZillow

Target:

`logPrice`

## Target Leakage Prevention

The following variables are intentionally excluded from the ML
predictor set because they contain or are directly derived from
listing price:

- pricePerSqFt
- zestimate_gap
- zestimate_gap_pct
- deal_score
- opportunity_status
- price_segment

This separation allows the project to use these variables for
business analysis without contaminating the independent price
prediction model.

## Coverage

- Zestimate records: {zestimate_count:,}
- Potential opportunities: {opportunity_count:,}
- Sufficient-sample ZIP codes: {sufficient_zip_count:,}
- Sparse ZIP codes: {sparse_zip_count:,}

## Data Treatment

Feature engineering does not automatically remove statistical
outliers.

Legitimate high-value, low-value, unusually large, or otherwise
unusual properties remain part of the analytical dataset.

Missing Zestimate and tax-assessed values remain missing.

## Validation

The final feature-engineered dataset was required to maintain:

- 810 properties
- 810 unique zpids
- 0 duplicate zpids
- valid positive listing prices
- valid price-per-square-foot calculations
- valid log-price calculations
- Zestimate-gap coverage matching Zestimate availability
- 16 potential Zestimate opportunities
- complete price segmentation
- complete ZIP sample classification

All validation checks must pass before this dataset is used
downstream.
"""

    return methodology


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(
    df,
    feature_summary,
    feature_dictionary,
    methodology,
):
    """Save engineered dataset and supporting documentation."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    feature_summary.to_csv(
        OUTPUT_DIR / "feature_summary.csv",
        index=False
    )

    feature_dictionary.to_csv(
        OUTPUT_DIR / "feature_dictionary.csv",
        index=False
    )

    with open(
        OUTPUT_DIR
        / "FEATURE_ENGINEERING_METHODOLOGY.md",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            methodology
        )

    print("=" * 70)
    print("FEATURE ENGINEERING OUTPUTS SAVED")
    print("=" * 70)
    print()

    print(
        "Feature-engineered dataset:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print()

    print("Reports:")

    print(
        "  - feature_summary.csv"
    )

    print(
        "  - feature_dictionary.csv"
    )

    print(
        "  - FEATURE_ENGINEERING_METHODOLOGY.md"
    )

    print()


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

def print_final_summary(df):
    """Print final Part 5 summary."""

    print("=" * 70)
    print("PART 5 FEATURE ENGINEERING COMPLETE")
    print("=" * 70)
    print()

    print(
        f"Final properties: "
        f"{len(df):,}"
    )

    print(
        f"Unique properties: "
        f"{df['zpid'].nunique():,}"
    )

    zestimate_coverage = (
        df["zestimate"]
        .notna()
        .mean()
        * 100
    )

    opportunity_count = (
        df["opportunity_status"]
        .eq("Potential Opportunity")
        .sum()
    )

    print(
        f"Zestimate coverage: "
        f"{zestimate_coverage:.2f}%"
    )

    print(
        f"Potential opportunities: "
        f"{opportunity_count:,}"
    )

    print()

    print("ML predictors:")

    for predictor in ML_PREDICTORS:
        print(
            f"  - {predictor}"
        )

    print()

    print("ML target:")

    print(
        f"  - {TARGET_VARIABLE}"
    )

    print()

    print(
        "Business-analysis features:"
    )

    for feature in BUSINESS_ANALYSIS_FEATURES:
        print(
            f"  - {feature}"
        )

    print()

    print(
        "The analysis-ready population was "
        "preserved at 810 properties."
    )

    print(
        "Target-leaking business metrics are "
        "separated from ML predictors."
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Validate source structure
    # --------------------------------------------------------

    validate_required_columns(
        df
    )

    # --------------------------------------------------------
    # Normalize numeric fields
    # --------------------------------------------------------

    df = normalize_numeric_columns(
        df
    )

    # --------------------------------------------------------
    # Validate population
    # --------------------------------------------------------

    df = validate_dataset_integrity(
        df
    )

    # --------------------------------------------------------
    # Engineer features
    # --------------------------------------------------------

    df = create_price_per_sqft(
        df
    )

    df = create_log_price(
        df
    )

    df = create_zestimate_features(
        df
    )

    df = create_deal_score(
        df
    )

    df = create_price_segments(
        df
    )

    df = create_zip_features(
        df
    )

    df = create_price_change_flag(
        df
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    df = validate_engineered_features(
        df
    )

    # --------------------------------------------------------
    # Documentation
    # --------------------------------------------------------

    feature_summary = (
        create_feature_summary(df)
    )

    feature_dictionary = (
        create_feature_dictionary()
    )

    methodology = (
        create_methodology_document(df)
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    save_reports(
        df,
        feature_summary,
        feature_dictionary,
        methodology,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_final_summary(
        df
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
