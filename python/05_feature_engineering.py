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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
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
