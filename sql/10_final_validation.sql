-- ============================================================
-- 10_final_validation.sql
-- San Antonio Zillow Housing Analysis
-- ============================================================
-- Purpose:
-- Perform final validation of the PostgreSQL analytical layer.
--
-- The validation confirms:
-- 1. The core analytical table contains the expected population.
-- 2. Major analytical views reconcile to the 810-property dataset.
-- 3. The executive ZIP ranking produces a valid business-facing result.
--
-- Authoritative analytical population:
-- 810 properties
-- 810 unique Zillow property identifiers (zpid)
-- ============================================================


-- ============================================================
-- STEP 1 — Validate the Core Analytical Table
-- ============================================================

SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT zpid) AS unique_properties,
    COUNT(*) - COUNT(DISTINCT zpid) AS duplicate_zpid_count,
    COUNT(*) FILTER (
        WHERE price IS NULL
    ) AS missing_price,
    COUNT(*) FILTER (
        WHERE zipcode IS NULL
    ) AS missing_zipcode,
    COUNT(*) FILTER (
        WHERE living_area IS NULL
    ) AS missing_living_area
FROM analytics.zillow_listings;


-- Expected:
-- total_rows = 810
-- unique_properties = 810
-- duplicate_zpid_count = 0
-- missing_price = 0
-- missing_zipcode = 0
-- missing_living_area = 0


-- ============================================================
-- STEP 2 — Reconcile the Major Business Views
-- ============================================================

SELECT
    'Market KPIs' AS validation_area,
    total_listings AS listing_count
FROM analytics.v_market_kpis

UNION ALL

SELECT
    'Price Segments' AS validation_area,
    SUM(listing_count) AS listing_count
FROM analytics.v_price_segment_kpis

UNION ALL

SELECT
    'ZIP Analysis' AS validation_area,
    SUM(listing_count) AS listing_count
FROM analytics.v_zip_kpis

UNION ALL

SELECT
    'Property Types' AS validation_area,
    SUM(listing_count) AS listing_count
FROM analytics.v_property_type_kpis;


-- Expected:
-- Market KPIs    = 810
-- Price Segments = 810
-- ZIP Analysis   = 810
-- Property Types = 810


-- ============================================================
-- STEP 3 — Executive ZIP Ranking
-- ============================================================
-- Rank sufficiently sampled ZIP codes by median listing price
-- while retaining key market, property, and Zestimate metrics.

SELECT
    zipcode,
    listing_count,
    zip_sample_status,
    median_price,
    median_price_per_sqft,
    median_living_area,
    average_days_on_zillow,
    listings_with_zestimate,
    zestimate_coverage_pct,
    potential_opportunities,
    RANK() OVER (
        ORDER BY median_price DESC
    ) AS median_price_rank
FROM analytics.v_zip_kpis
WHERE zip_sample_status = 'Sufficient Sample'
ORDER BY median_price_rank;


-- ============================================================
-- VALIDATION COMPLETE
-- ============================================================
-- The SQL analytical layer should reconcile to the authoritative
-- 810-property San Antonio Zillow analytical dataset.
--
-- ZIP rankings exclude sparse ZIP codes with fewer than 10 listings.
-- Sparse ZIP records remain in the analytical database.
-- ============================================================
