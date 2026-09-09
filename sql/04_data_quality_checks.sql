-- San Antonio Zillow Housing Analysis
-- Analytical data-quality checks
-- Purpose: Validate the structure and completeness of the analytical table.

-- 1. Overall row and duplicate validation

SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT zpid) AS unique_properties,
    COUNT(*) - COUNT(DISTINCT zpid) AS duplicate_zpids
FROM analytics.zillow_listings;


-- 2. Key numeric range validation

SELECT
    MIN(price) AS minimum_price,
    MAX(price) AS maximum_price,
    MIN(living_area) AS minimum_living_area,
    MAX(living_area) AS maximum_living_area,
    MIN(price_per_sqft) AS minimum_price_per_sqft,
    MAX(price_per_sqft) AS maximum_price_per_sqft
FROM analytics.zillow_listings;


-- 3. Missing-value validation for core analytical fields

SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE zpid IS NULL) AS missing_zpid,
    COUNT(*) FILTER (WHERE price IS NULL) AS missing_price,
    COUNT(*) FILTER (WHERE zipcode IS NULL) AS missing_zipcode,
    COUNT(*) FILTER (WHERE living_area IS NULL) AS missing_living_area,
    COUNT(*) FILTER (WHERE latitude IS NULL) AS missing_latitude,
    COUNT(*) FILTER (WHERE longitude IS NULL) AS missing_longitude,
    COUNT(*) FILTER (WHERE zestimate IS NULL) AS missing_zestimate,
    COUNT(*) FILTER (WHERE tax_assessed_value IS NULL) AS missing_tax_assessed_value
FROM analytics.zillow_listings;


-- 4. Zestimate coverage

SELECT
    COUNT(*) AS total_rows,
    COUNT(zestimate) AS listings_with_zestimate,
    COUNT(*) - COUNT(zestimate) AS listings_without_zestimate,
    ROUND(
        100.0 * COUNT(zestimate) / NULLIF(COUNT(*), 0),
        2
    ) AS zestimate_coverage_pct
FROM analytics.zillow_listings;


-- 5. ZIP-code completeness

SELECT
    COUNT(*) AS total_rows,
    COUNT(zipcode) AS listings_with_zipcode,
    COUNT(*) - COUNT(zipcode) AS listings_without_zipcode
FROM analytics.zillow_listings;


-- 6. Property identifier validation

SELECT
    COUNT(*) AS total_rows,
    COUNT(zpid) AS non_null_zpids,
    COUNT(DISTINCT zpid) AS unique_zpids
FROM analytics.zillow_listings;
