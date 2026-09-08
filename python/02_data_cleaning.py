"""
02_data_cleaning.py

San Antonio Zillow Housing Analysis
-----------------------------------

Cleans and validates the processed Zillow dataset and creates
the analysis-ready dataset used by the downstream project.

Workflow:
1. Load the processed Zillow dataset.
2. Inspect the initial dataset.
3. Validate key fields.
4. Remove invalid listing prices.
5. Remove LOT properties.
6. Remove records with invalid living area.
7. Remove implausible bathroom counts above 20.
8. Convert negative daysOnZillow values to missing.
9. Convert invalid Zestimate values of 0 to missing.
10. Recalculate price per square foot.
11. Create log-transformed listing price.
12. Preserve legitimate high-value properties and other
    legitimate outliers.
13. Validate duplicate property identifiers.
14. Report the impact of each cleaning rule.
15. Save the analysis-ready dataset.

Important:
- Missing Zestimate values are retained as missing.
- Missing tax-assessed values are retained as missing.
- Legitimate outliers are retained rather than automatically
  deleted.
- No artificial listing dates are created.
- Price-derived variables are not used as independent ML
  predictors when they contain the target variable.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/San_Antonio_Zillow_Processed.csv"
)

OUTPUT_DIR = Path(
    "data/processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "San_Antonio_Zillow_Analysis.csv"
)


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset():
    """
    Load the processed Zillow dataset.
    """

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nInput dataset was not found:\n"
            f"{INPUT_FILE}\n\n"
            "Make sure the processed dataset is located "
            "inside data/processed/."
        )

    print("=" * 60)
    print("LOADING PROCESSED ZILLOW DATA")
    print("=" * 60)

    print(
        f"Input file: {INPUT_FILE}"
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
# INITIAL VALIDATION
# ============================================================

def validate_required_columns(df):
    """
    Verify that the dataset contains the fields required
    for the cleaning pipeline.
    """

    required_columns = [
        "zpid",
        "addressZipcode",
        "beds",
        "baths",
        "area",
        "unformattedPrice",
        "zestimate",
        "daysOnZillow",
        "homeType",
        "taxAssessedValue",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nThe following required columns "
            "are missing:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    print(
        "\nRequired columns validated successfully."
    )


# ============================================================
# DATA TYPE CONVERSION
# ============================================================

def convert_numeric_columns(df):
    """
    Convert analytical numeric fields to numeric types.
    Invalid values become missing.
    """

    numeric_columns = [
        "beds",
        "baths",
        "area",
        "unformattedPrice",
        "zestimate",
        "daysOnZillow",
        "taxAssessedValue",
        "lotAreaValue",
        "priceChange",
        "lotAreaSqFt",
        "pricePerSqFt",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# CLEANING STEP 1 — PRICE
# ============================================================

def remove_invalid_prices(df):
    """
    Remove records with missing or non-positive listing prices.
    """

    before = len(df)

    invalid_price = (
        df["unformattedPrice"].isna()
        |
        (df["unformattedPrice"] <= 0)
    )

    removed = invalid_price.sum()

    df = df.loc[
        ~invalid_price
    ].copy()

    after = len(df)

    print("\n" + "-" * 60)
    print("CLEANING STEP 1 — LISTING PRICE")
    print("-" * 60)

    print(
        f"Invalid price records removed: {removed:,}"
    )

    print(
        f"Rows before: {before:,}"
    )

    print(
        f"Rows after: {after:,}"
    )

    return df


# ============================================================
# CLEANING STEP 2 — PROPERTY TYPE
# ============================================================

def remove_lot_properties(df):
    """
    Remove LOT properties because they do not represent
    residential structures and therefore do not belong in
    the residential housing price analysis.
    """

    before = len(df)

    lot_mask = (
        df["homeType"]
        .astype(str)
        .str.upper()
        .str.strip()
        == "LOT"
    )

    removed = lot_mask.sum()

    df = df.loc[
        ~lot_mask
    ].copy()

    after = len(df)

    print("\n" + "-" * 60)
    print("CLEANING STEP 2 — PROPERTY TYPE")
    print("-" * 60)

    print(
        f"LOT properties removed: {removed:,}"
    )

    print(
        f"Rows before: {before:,}"
    )

    print(
        f"Rows after: {after:,}"
    )

    return df


# ============================================================
# CLEANING STEP 3 — LIVING AREA
# ============================================================

def remove_invalid_living_area(df):
    """
    Remove records with missing or non-positive living area.
    """

    before = len(df)

    invalid_area = (
        df["area"].isna()
        |
        (df["area"] <= 0)
    )

    removed = invalid_area.sum()

    df = df.loc[
        ~invalid_area
    ].copy()

    after = len(df)

    print("\n" + "-" * 60)
    print("CLEANING STEP 3 — LIVING AREA")
    print("-" * 60)

    print(
        f"Invalid living-area records removed: "
        f"{removed:,}"
    )

    print(
        f"Rows before: {before:,}"
    )

    print(
        f"Rows after: {after:,}"
    )

    return df


# ============================================================
# CLEANING STEP 4 — BATHROOMS
# ============================================================

def remove_implausible_bathrooms(df):
    """
    Remove records with bathroom counts above 20.

    This threshold follows the cleaning rule established
    during the original capstone analysis.
    """

    before = len(df)

    invalid_baths = (
        df["baths"].notna()
        &
        (df["baths"] > 20)
    )

    removed = invalid_baths.sum()

    df = df.loc[
        ~invalid_baths
    ].copy()

    after = len(df)

    print("\n" + "-" * 60)
    print("CLEANING STEP 4 — BATHROOM COUNT")
    print("-" * 60)

    print(
        f"Bathroom records above 20 removed: "
        f"{removed:,}"
    )

    print(
        f"Rows before: {before:,}"
    )

    print(
        f"Rows after: {after:,}"
    )

    return df


# ============================================================
# CLEANING STEP 5 — DAYS ON ZILLOW
# ============================================================

def handle_negative_days_on_zillow(df):
    """
    Convert negative daysOnZillow values to missing.

    Negative values are treated as invalid measurements,
    but the associated property is retained.
    """

    if "daysOnZillow" not in df.columns:

        return df

    negative_mask = (
        df["daysOnZillow"] < 0
    )

    count = negative_mask.sum()

    df.loc[
        negative_mask,
        "daysOnZillow"
    ] = np.nan

    print("\n" + "-" * 60)
    print("CLEANING STEP 5 — DAYS ON ZILLOW")
    print("-" * 60)

    print(
        f"Negative daysOnZillow values "
        f"converted to missing: {count:,}"
    )

    print(
        "Associated properties were retained."
    )

    return df


# ============================================================
# CLEANING STEP 6 — ZESTIMATE
# ============================================================

def handle_invalid_zestimates(df):
    """
    Convert Zestimate values of zero to missing.

    A zero Zestimate is treated as unavailable rather than
    as an actual property valuation.

    Existing missing values remain missing.
    """

    if "zestimate" not in df.columns:

        return df

    zero_mask = (
        df["zestimate"] == 0
    )

    count = zero_mask.sum()

    df.loc[
        zero_mask,
        "zestimate"
    ] = np.nan

    print("\n" + "-" * 60)
    print("CLEANING STEP 6 — ZESTIMATE")
    print("-" * 60)

    print(
        f"Zero Zestimate values converted "
        f"to missing: {count:,}"
    )

    print(
        "Missing Zestimate values were retained."
    )

    return df


# ============================================================
# FEATURE ENGINEERING — PRICE PER SQUARE FOOT
# ============================================================

def create_price_per_sqft(df):
    """
    Recalculate price per square foot from listing price
    and living area.

    Formula:

        pricePerSqFt = listing price / living area
    """

    df["pricePerSqFt"] = np.nan

    valid_mask = (
        df["unformattedPrice"].notna()
        &
        df["area"].notna()
        &
        (df["area"] > 0)
    )

    df.loc[
        valid_mask,
        "pricePerSqFt"
    ] = (
        df.loc[
            valid_mask,
            "unformattedPrice"
        ]
        /
        df.loc[
            valid_mask,
            "area"
        ]
    )

    invalid_price_sqft = (
        ~np.isfinite(
            df["pricePerSqFt"]
        )
    )

    df.loc[
        invalid_price_sqft,
        "pricePerSqFt"
    ] = np.nan

    print("\n" + "-" * 60)
    print("FEATURE ENGINEERING — PRICE PER SQUARE FOOT")
    print("-" * 60)

    print(
        "pricePerSqFt recalculated as "
        "listing price / living area."
    )

    print(
        f"Missing pricePerSqFt values: "
        f"{df['pricePerSqFt'].isna().sum():,}"
    )

    return df


# ============================================================
# FEATURE ENGINEERING — LOG PRICE
# ============================================================

def create_log_price(df):
    """
    Create the natural-log transformation of listing price.

    Formula:

        logPrice = ln(listing price)
    """

    valid_mask = (
        df["unformattedPrice"].notna()
        &
        (df["unformattedPrice"] > 0)
    )

    df["logPrice"] = np.nan

    df.loc[
        valid_mask,
        "logPrice"
    ] = np.log(
        df.loc[
            valid_mask,
            "unformattedPrice"
        ]
    )

    print("\n" + "-" * 60)
    print("FEATURE ENGINEERING — LOG PRICE")
    print("-" * 60)

    print(
        "logPrice created using the natural logarithm "
        "of listing price."
    )

    print(
        f"Missing logPrice values: "
        f"{df['logPrice'].isna().sum():,}"
    )

    return df


# ============================================================
# FEATURE ENGINEERING — PRICE CHANGE FLAG
# ============================================================

def create_price_change_flag(df):
    """
    Create an indicator showing whether a property has
    a recorded price change.
    """

    if "priceChange" not in df.columns:

        df["hasPriceChange"] = 0

        return df

    price_change = pd.to_numeric(
        df["priceChange"],
        errors="coerce"
    )

    df["hasPriceChange"] = (
        price_change.notna()
        .astype(int)
    )

    print("\n" + "-" * 60)
    print("FEATURE ENGINEERING — PRICE CHANGE FLAG")
    print("-" * 60)

    print(
        f"Listings with recorded price changes: "
        f"{df['hasPriceChange'].sum():,}"
    )

    return df


# ============================================================
# DUPLICATE VALIDATION
# ============================================================

def validate_duplicates(df):
    """
    Validate duplicate property identifiers.
    """

    print("\n" + "-" * 60)
    print("DUPLICATE VALIDATION")
    print("-" * 60)

    if "zpid" not in df.columns:

        print(
            "zpid column not available."
        )

        return

    duplicate_count = (
        df["zpid"]
        .duplicated()
        .sum()
    )

    unique_count = (
        df["zpid"]
        .nunique()
    )

    print(
        f"Unique zpids: {unique_count:,}"
    )

    print(
        f"Duplicate zpids: {duplicate_count:,}"
    )

    if duplicate_count == 0:

        print(
            "Duplicate validation passed."
        )

    else:

        print(
            "WARNING: Duplicate property identifiers "
            "remain in the dataset."
        )


# ============================================================
# OUTLIER REPORT
# ============================================================

def report_price_distribution(df):
    """
    Report extreme listing prices for review.

    Legitimate high-value properties are retained.
    """

    price = df["unformattedPrice"]

    print("\n" + "-" * 60)
    print("OUTLIER REVIEW")
    print("-" * 60)

    print(
        f"Minimum listing price: "
        f"${price.min():,.2f}"
    )

    print(
        f"Median listing price: "
        f"${price.median():,.2f}"
    )

    print(
        f"Maximum listing price: "
        f"${price.max():,.2f}"
    )

    print(
        "\nHigh-value properties are retained "
        "unless they fail a separate data-quality rule."
    )


# ============================================================
# FINAL DATA QUALITY CHECK
# ============================================================

def final_validation(df):
    """
    Perform final validation of the analysis-ready dataset.
    """

    print("\n" + "=" * 60)
    print("FINAL DATA QUALITY VALIDATION")
    print("=" * 60)

    print(
        f"Final rows: {len(df):,}"
    )

    print(
        f"Final columns: {len(df.columns):,}"
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

    # --------------------------------------------------------
    # Required analytical fields
    # --------------------------------------------------------

    validation_fields = [
        "unformattedPrice",
        "area",
        "addressZipcode",
        "logPrice",
        "pricePerSqFt"
    ]

    print(
        "\nMissing values in required "
        "analytical fields:"
    )

    for field in validation_fields:

        if field in df.columns:

            missing = (
                df[field]
                .isna()
                .sum()
            )

            print(
                f"  {field}: {missing:,}"
            )

    # --------------------------------------------------------
    # Price validation
    # --------------------------------------------------------

    invalid_prices = (
        df["unformattedPrice"].isna()
        |
        (df["unformattedPrice"] <= 0)
    ).sum()

    print(
        f"\nInvalid listing prices remaining: "
        f"{invalid_prices:,}"
    )

    # --------------------------------------------------------
    # Area validation
    # --------------------------------------------------------

    invalid_area = (
        df["area"].isna()
        |
        (df["area"] <= 0)
    ).sum()

    print(
        f"Invalid living area records remaining: "
        f"{invalid_area:,}"
    )

    # --------------------------------------------------------
    # Price/SF validation
    # --------------------------------------------------------

    invalid_price_sqft = (
        ~np.isfinite(
            df["pricePerSqFt"]
        )
    ).sum()

    print(
        f"Invalid pricePerSqFt values remaining: "
        f"{invalid_price_sqft:,}"
    )

    # --------------------------------------------------------
    # Log price validation
    # --------------------------------------------------------

    invalid_log_price = (
        df["logPrice"].isna()
        |
        ~np.isfinite(
            df["logPrice"]
        )
    ).sum()

    print(
        f"Invalid logPrice values remaining: "
        f"{invalid_log_price:,}"
    )


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(df):
    """
    Save the analysis-ready dataset.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("ANALYSIS DATASET SAVED")
    print("=" * 60)

    print(
        f"Output file: {OUTPUT_FILE}"
    )

    print(
        f"Rows saved: {len(df):,}"
    )

    print(
        f"Columns saved: {len(df.columns):,}"
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)

    print(
        "SAN ANTONIO ZILLOW "
        "DATA CLEANING PIPELINE"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Validate structure
    # --------------------------------------------------------

    validate_required_columns(
        df
    )

    # --------------------------------------------------------
    # Convert data types
    # --------------------------------------------------------

    df = convert_numeric_columns(
        df
    )

    # --------------------------------------------------------
    # Cleaning
    # --------------------------------------------------------

    df = remove_invalid_prices(
        df
    )

    df = remove_lot_properties(
        df
    )

    df = remove_invalid_living_area(
        df
    )

    df = remove_implausible_bathrooms(
        df
    )

    df = handle_negative_days_on_zillow(
        df
    )

    df = handle_invalid_zestimates(
        df
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    df = create_price_per_sqft(
        df
    )

    df = create_log_price(
        df
    )

    df = create_price_change_flag(
        df
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_duplicates(
        df
    )

    report_price_distribution(
        df
    )

    final_validation(
        df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_dataset(
        df
    )

    print("\n" + "=" * 60)
    print(
        "PART 2 DATA CLEANING COMPLETE"
    )
    print("=" * 60)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
