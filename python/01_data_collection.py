"""
01_data_collection.py

San Antonio Zillow Housing Analysis
-----------------------------------

Collects Zillow listing data from San Antonio search-result pages
using Selenium and Chrome.

Workflow:
1. Open Zillow using a real browser session.
2. Navigate through San Antonio search-result pages.
3. Read Zillow's embedded __NEXT_DATA__ JSON.
4. Extract listing records from listResults.
5. Combine records from all pages.
6. Remove duplicate properties using zpid.
7. Extract additional fields from hdpData.homeInfo.
8. Create basic engineered fields.
9. Validate the collected dataset.
10. Save the raw dataset to data/raw/.

Important:
Zillow may restrict automated browsing. This script does not
attempt to bypass access controls. If Zillow presents a block,
challenge, CAPTCHA, or otherwise prevents the page from loading,
the script stops safely and reports the collection status.

The raw collection process is retained as a reproducible data-
collection pipeline. The downstream analytical workflow should
only use a successfully collected and validated dataset.
"""

import json
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException


# ============================================================
# CONFIGURATION
# ============================================================

NUM_PAGES = 20

BASE_URL = (
    "https://www.zillow.com/homes/for_sale/"
    "san-antonio-tx/"
)

OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "san_antonio_zillow_raw.csv"

PAGE_DELAY = 4


# ============================================================
# URL HANDLING
# ============================================================

def build_page_url(page_number):
    """
    Build the Zillow search URL for a given page.
    """

    if page_number == 1:
        return BASE_URL

    return (
        "https://www.zillow.com/homes/for_sale/"
        f"san-antonio-tx/{page_number}_p/"
    )


# ============================================================
# SELENIUM SETUP
# ============================================================

def create_driver():
    """
    Create a Selenium Chrome browser.

    Selenium Manager automatically handles the compatible
    browser driver.
    """

    options = Options()

    options.add_argument("--start-maximized")
    options.add_argument("--disable-notifications")

    driver = webdriver.Chrome(options=options)

    return driver


# ============================================================
# ZILLOW PAGE PARSING
# ============================================================

def extract_listings(html):
    """
    Extract Zillow listing records from the page's
    __NEXT_DATA__ JSON structure.

    Returns:
        list: Zillow listing dictionaries
    """

    soup = BeautifulSoup(html, "html.parser")

    script = soup.find(
        "script",
        id="__NEXT_DATA__"
    )

    if script is None:
        print("  __NEXT_DATA__ script was not found.")
        return []

    if not script.string:
        print("  __NEXT_DATA__ script contains no data.")
        return []

    try:
        data = json.loads(script.string)

    except json.JSONDecodeError:
        print("  Could not decode Zillow JSON.")
        return []

    try:
        listings = (
            data["props"]
            ["pageProps"]
            ["searchPageState"]
            ["cat1"]
            ["searchResults"]
            ["listResults"]
        )

        return listings

    except (KeyError, TypeError):
        print(
            "  Zillow listing structure was not found."
        )

        return []


# ============================================================
# BLOCK DETECTION
# ============================================================

def detect_zillow_block(html):
    """
    Check whether the returned page appears to be a
    Zillow access-block or verification page.

    Returns:
        bool: True if a block indicator is detected.
    """

    page_text = html.lower()

    block_indicators = [
        "access denied",
        "robot",
        "captcha",
        "unusual activity",
        "verify you are human"
    ]

    detected_indicators = [
        indicator
        for indicator in block_indicators
        if indicator in page_text
    ]

    if detected_indicators:

        print("\nZillow access restriction detected.")

        print(
            "Indicators found:",
            ", ".join(detected_indicators)
        )

        print(
            "The collector will stop without attempting "
            "to bypass the restriction."
        )

        return True

    return False


# ============================================================
# DATA COLLECTION
# ============================================================

def collect_listings(driver):
    """
    Collect listings from the configured Zillow pages.

    Returns:
        tuple:
            all_listings: collected listing records
            status: collection status message
    """

    all_listings = []

    collection_status = "Completed"

    for page_number in range(
        1,
        NUM_PAGES + 1
    ):

        url = build_page_url(page_number)

        print("\n" + "=" * 60)

        print(
            f"Collecting page "
            f"{page_number}/{NUM_PAGES}"
        )

        print(f"URL: {url}")

        print("=" * 60)

        try:

            driver.get(url)

            time.sleep(PAGE_DELAY)

            html = driver.page_source

            # ------------------------------------------------
            # Check for access restrictions
            # ------------------------------------------------

            if detect_zillow_block(html):

                collection_status = (
                    "Blocked by Zillow"
                )

                break

            # ------------------------------------------------
            # Extract listing data
            # ------------------------------------------------

            listings = extract_listings(html)

            print(
                f"Listings found: {len(listings)}"
            )

            if not listings:

                print(
                    "No listing records were found "
                    "on this page."
                )

                collection_status = (
                    "No listing data found"
                )

                break

            all_listings.extend(listings)

        except WebDriverException as error:

            print("\nBrowser error:")

            print(error)

            collection_status = (
                "Browser error"
            )

            break

        time.sleep(PAGE_DELAY)

    return all_listings, collection_status


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def remove_duplicate_listings(df):
    """
    Remove duplicate properties using zpid.
    """

    if "zpid" not in df.columns:

        print(
            "\nWarning:"
            "\nzpid column was not found."
        )

        return df

    duplicate_count = (
        df["zpid"].duplicated().sum()
    )

    print(
        f"\nDuplicate zpid records found: "
        f"{duplicate_count}"
    )

    before = len(df)

    df = df.drop_duplicates(
        subset="zpid",
        keep="first"
    )

    after = len(df)

    print(
        f"Records before deduplication: "
        f"{before}"
    )

    print(
        f"Records after deduplication: "
        f"{after}"
    )

    print(
        f"Records removed: "
        f"{before - after}"
    )

    return df


# ============================================================
# HOME INFORMATION EXTRACTION
# ============================================================

def extract_home_info(df):
    """
    Extract selected fields from Zillow's
    hdpData.homeInfo structure.
    """

    if "hdpData" not in df.columns:

        print(
            "\nWarning:"
            "\nhdpData column was not found."
        )

        return df

    home_info_records = []

    for value in df["hdpData"]:

        if isinstance(value, dict):

            home_info = value.get(
                "homeInfo",
                {}
            )

        else:

            home_info = {}

        if not isinstance(
            home_info,
            dict
        ):

            home_info = {}

        home_info_records.append(
            home_info
        )

    home_info_df = pd.json_normalize(
        home_info_records
    )

    fields_to_extract = [

        "daysOnZillow",
        "homeType",
        "homeStatus",
        "taxAssessedValue",
        "lotAreaValue",
        "lotAreaUnit",
        "priceChange",
        "timeOnZillow"

    ]

    for field in fields_to_extract:

        if field in home_info_df.columns:

            df[field] = (
                home_info_df[field].values
            )

        else:

            df[field] = pd.NA

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    """
    Create basic analytical fields from the
    collected Zillow data.
    """

    # --------------------------------------------------------
    # Lot area in square feet
    # --------------------------------------------------------

    df["lotAreaSqFt"] = pd.NA

    if (
        "lotAreaValue" in df.columns
        and
        "lotAreaUnit" in df.columns
    ):

        lot_area = pd.to_numeric(
            df["lotAreaValue"],
            errors="coerce"
        )

        units = (
            df["lotAreaUnit"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        sqft_mask = (
            units == "sqft"
        )

        acre_mask = (
            units == "acres"
        )

        df.loc[
            sqft_mask,
            "lotAreaSqFt"
        ] = lot_area.loc[
            sqft_mask
        ]

        df.loc[
            acre_mask,
            "lotAreaSqFt"
        ] = (
            lot_area.loc[
                acre_mask
            ] * 43560
        )

    # --------------------------------------------------------
    # Price per square foot
    # --------------------------------------------------------

    df["pricePerSqFt"] = pd.NA

    if (
        "unformattedPrice" in df.columns
        and
        "area" in df.columns
    ):

        price = pd.to_numeric(
            df["unformattedPrice"],
            errors="coerce"
        )

        area = pd.to_numeric(
            df["area"],
            errors="coerce"
        )

        valid_area = area > 0

        df.loc[
            valid_area,
            "pricePerSqFt"
        ] = (
            price.loc[valid_area]
            /
            area.loc[valid_area]
        )

    # --------------------------------------------------------
    # Price-change indicator
    # --------------------------------------------------------

    if "priceChange" in df.columns:

        price_change = pd.to_numeric(
            df["priceChange"],
            errors="coerce"
        )

        df["hasPriceChange"] = (
            price_change.notna()
            .astype(int)
        )

    else:

        df["hasPriceChange"] = 0

    return df


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset(df):

    print("\n" + "=" * 60)

    print("DATASET VALIDATION")

    print("=" * 60)

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    if "zpid" in df.columns:

        print(
            f"Unique properties: "
            f"{df['zpid'].nunique()}"
        )

        print(
            f"Duplicate zpids: "
            f"{df['zpid'].duplicated().sum()}"
        )

    if "addressZipcode" in df.columns:

        print(
            f"Unique ZIP codes: "
            f"{df['addressZipcode'].nunique()}"
        )

    print(
        "\nMissing values in key fields:"
    )

    key_fields = [

        "zpid",
        "addressZipcode",
        "beds",
        "baths",
        "area",
        "unformattedPrice",
        "zestimate",
        "daysOnZillow",
        "taxAssessedValue",
        "lotAreaValue",
        "lotAreaSqFt",
        "pricePerSqFt"

    ]

    for field in key_fields:

        if field in df.columns:

            missing = (
                df[field]
                .isna()
                .sum()
            )

            print(
                f"  {field}: {missing}"
            )


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(df):

    if df.empty:

        print(
            "\nNo listings were collected."
        )

        print(
            "The raw dataset was not created."
        )

        return False

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)

    print("RAW DATASET SAVED")

    print("=" * 60)

    print(
        f"File: {OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    return True


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)

    print(
        "SAN ANTONIO ZILLOW "
        "DATA COLLECTION"
    )

    print("=" * 60)

    driver = None

    listings = []

    collection_status = (
        "Not started"
    )

    try:

        print(
            "\nStarting Chrome..."
        )

        driver = create_driver()

        print(
            "Chrome started successfully."
        )

        (
            listings,
            collection_status
        ) = collect_listings(
            driver
        )

    except WebDriverException as error:

        print(
            "\nUnable to start "
            "Selenium/Chrome."
        )

        print(error)

        collection_status = (
            "Browser startup error"
        )

        return

    finally:

        if driver is not None:

            driver.quit()

            print(
                "\nChrome session closed."
            )

    # --------------------------------------------------------
    # Collection summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print(
        "COLLECTION COMPLETE"
    )

    print("=" * 60)

    print(
        f"Collection status: "
        f"{collection_status}"
    )

    print(
        f"Total raw records collected: "
        f"{len(listings)}"
    )

    # --------------------------------------------------------
    # Stop if nothing was collected
    # --------------------------------------------------------

    if not listings:

        print(
            "\nNo listings were collected."
        )

        print(
            "No raw dataset was created."
        )

        if collection_status == (
            "Blocked by Zillow"
        ):

            print(
                "\nZillow prevented automated "
                "collection during this run."
            )

            print(
                "The script does not attempt "
                "to bypass the restriction."
            )

        return

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    print(
        "\nCreating raw DataFrame..."
    )

    df = pd.DataFrame(
        listings
    )

    print(
        f"Rows: {df.shape[0]}"
    )

    print(
        f"Columns: {df.shape[1]}"
    )

    # --------------------------------------------------------
    # Clean structural issues
    # --------------------------------------------------------

    df = remove_duplicate_listings(
        df
    )

    # --------------------------------------------------------
    # Extract nested information
    # --------------------------------------------------------

    df = extract_home_info(
        df
    )

    # --------------------------------------------------------
    # Engineer basic fields
    # --------------------------------------------------------

    df = engineer_features(
        df
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_dataset(
        df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_dataset(
        df
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
