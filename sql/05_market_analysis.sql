-- San Antonio Zillow Housing Analysis
-- SQL Market Analysis
-- Purpose: Analyze overall San Antonio listing-market characteristics.

-- ============================================================
-- 1. Overall Market Summary
-- ============================================================

SELECT
    COUNT(*) AS total_listings,

    ROUND(AVG(price), 2) AS average_listing_price,

    ROUND(
        PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY price)::NUMERIC,
        2
    ) AS median_listing_price,

    MIN(price) AS minimum_price,

    MAX(price) AS maximum_price,

    ROUND(AVG(living_area), 2) AS average_living_area,

    ROUND(
        PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY living_area)::NUMERIC,
        2
    ) AS median_living_area,

    ROUND(AVG(price_per_sqft), 2) AS average_price_per_sqft,

    ROUND(
        PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY price_per_sqft)::NUMERIC,
        2
    ) AS median_price_per_sqft,

    ROUND(AVG(days_on_zillow), 2) AS average_days_on_zillow,

    ROUND(
        PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY days_on_zillow)::NUMERIC,
        2
    ) AS median_days_on_zillow

FROM analytics.zillow_listings;


-- ============================================================
-- 2. Price Distribution
-- ============================================================

SELECT
    COUNT(price) AS listings_with_price,

    ROUND(
        PERCENTILE_CONT(0.25)
        WITHIN GROUP (ORDER BY price)::NUMERIC,
        2
    ) AS first_quartile_price,

    ROUND(
        PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY price)::NUMERIC,
        2
    ) AS median_price,

    ROUND(
        PERCENTILE_CONT(0.75)
        WITHIN GROUP (ORDER BY price)::NUMERIC,
        2
    ) AS third_quartile_price,

    ROUND(
        (
            PERCENTILE_CONT(0.75)
            WITHIN GROUP (ORDER BY price)
            -
            PERCENTILE_CONT(0.25)
            WITHIN GROUP (ORDER BY price)
        )::NUMERIC,
        2
    ) AS price_iqr

FROM analytics.zillow_listings
WHERE price IS NOT NULL;


-- ============================================================
-- 3. Zestimate and Tax-Assessment Coverage
-- ============================================================

SELECT
    COUNT(*) AS total_listings,

    COUNT(zestimate) AS listings_with_zestimate,

    COUNT(*) - COUNT(zestimate) AS listings_without_zestimate,

    ROUND(
        100.0 * COUNT(zestimate) / NULLIF(COUNT(*), 0),
        2
    ) AS zestimate_coverage_pct,

    COUNT(tax_assessed_value) AS listings_with_tax_assessment,

    COUNT(*) - COUNT(tax_assessed_value)
        AS listings_without_tax_assessment,

    ROUND(
        100.0 * COUNT(tax_assessed_value) / NULLIF(COUNT(*), 0),
        2
    ) AS tax_assessment_coverage_pct

FROM analytics.zillow_listings;


-- ============================================================
-- 4. Property-Type Market Composition
-- ============================================================

SELECT
    home_type,

    COUNT(*) AS listing_count,

    ROUND(
        100.0 * COUNT(*) /
        SUM(COUNT(*)) OVER (),
        2
    ) AS percentage_of_listings,

    ROUND(AVG(price), 2) AS mean_price,

    ROUND(
        PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY price)::NUMERIC,
        2
    ) AS median_price

FROM analytics.zillow_listings

GROUP BY home_type

ORDER BY listing_count DESC;


--- ============================================================
-- 5. Price Segment Distribution
-- ============================================================

WITH price_segments AS (
    SELECT
        CASE
            WHEN price < 150000
                THEN 'Under $150K'

            WHEN price < 300000
                THEN '$150K-$300K'

            WHEN price < 500000
                THEN '$300K-$500K'

            WHEN price < 1000000
                THEN '$500K-$1M'

            WHEN price < 2000000
                THEN '$1M-$2M'

            ELSE '$2M+'
        END AS price_segment,

        price,
        price_per_sqft,
        days_on_zillow

    FROM analytics.zillow_listings

    WHERE price IS NOT NULL
)

SELECT
    price_segment,

    COUNT(*) AS listing_count,

    ROUND(
        100.0 * COUNT(*) /
        SUM(COUNT(*)) OVER (),
        2
    ) AS percentage_of_listings,

    ROUND(AVG(price), 2) AS average_price,

    ROUND(
        PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY price)::NUMERIC,
        2
    ) AS median_price,

    ROUND(AVG(price_per_sqft), 2) AS average_price_per_sqft,

    ROUND(AVG(days_on_zillow), 2) AS average_days_on_zillow

FROM price_segments

GROUP BY price_segment

ORDER BY
    CASE price_segment
        WHEN 'Under $150K' THEN 1
        WHEN '$150K-$300K' THEN 2
        WHEN '$300K-$500K' THEN 3
        WHEN '$500K-$1M' THEN 4
        WHEN '$1M-$2M' THEN 5
        WHEN '$2M+' THEN 6
    END;
