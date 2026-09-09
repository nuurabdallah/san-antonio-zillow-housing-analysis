-- ============================================================
-- 08_advanced_rankings.sql
-- San Antonio Zillow Housing Analysis
-- ============================================================
-- Purpose:
-- Perform advanced ZIP-level and property-level ranking analysis
-- using PostgreSQL window functions.
--
-- ZIP codes with fewer than 10 listings are excluded from formal
-- ZIP rankings but remain in the analytical dataset.
-- ============================================================


-- ============================================================
-- 1. ZIP MARKET RANKINGS
-- ============================================================

WITH zip_metrics AS (
    SELECT
        zipcode,
        COUNT(*) AS listing_count,

        PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY price)::NUMERIC
        AS median_price,

        PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY price_per_sqft)::NUMERIC
        AS median_price_per_sqft,

        PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY living_area)::NUMERIC
        AS median_living_area,

        PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY days_on_zillow)::NUMERIC
        AS median_days_on_zillow

    FROM analytics.zillow_listings

    WHERE zipcode IS NOT NULL

    GROUP BY zipcode

    HAVING COUNT(*) >= 10
),

ranked_zip AS (
    SELECT
        zipcode,
        listing_count,
        median_price,
        median_price_per_sqft,
        median_living_area,
        median_days_on_zillow,

        RANK() OVER (
            ORDER BY median_price DESC
        ) AS price_rank,

        RANK() OVER (
            ORDER BY median_price_per_sqft DESC
        ) AS price_per_sqft_rank,

        RANK() OVER (
            ORDER BY median_living_area DESC
        ) AS size_rank,

        RANK() OVER (
            ORDER BY median_days_on_zillow DESC
        ) AS days_on_zillow_rank

    FROM zip_metrics
)

SELECT
    *
FROM ranked_zip
ORDER BY price_rank;


-- ============================================================
-- 2. ZIP × PROPERTY-TYPE COMPOSITION
-- ============================================================

SELECT
    zipcode,
    home_type,

    COUNT(*) AS listing_count,

    ROUND(
        (
            100.0 * COUNT(*)
            / SUM(COUNT(*)) OVER (
                PARTITION BY zipcode
            )
        )::NUMERIC,
        2
    ) AS pct_of_zip,

    ROUND(
        PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY price)::NUMERIC,
        2
    ) AS median_price,

    ROUND(
        PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY price_per_sqft)::NUMERIC,
        2
    ) AS median_price_per_sqft

FROM analytics.zillow_listings

WHERE zipcode IS NOT NULL
  AND home_type IS NOT NULL
  AND price IS NOT NULL

GROUP BY
    zipcode,
    home_type

ORDER BY
    zipcode,
    listing_count DESC;


-- ============================================================
-- 3. PROPERTY-LEVEL ZESTIMATE OPPORTUNITY RANKING
-- ============================================================
-- Potential Opportunity:
-- Zestimate > Listing Price
--
-- Zestimate Gap:
-- Zestimate - Listing Price
--
-- Zestimate Gap Percentage:
-- (Zestimate - Listing Price) / Listing Price * 100
--
-- Opportunities are ranked by relative gap percentage.


WITH opportunity_data AS (
    SELECT
        zpid,
        zipcode,
        city,
        home_type,
        price,
        zestimate,
        living_area,
        bedrooms,
        bathrooms,

        zestimate - price
            AS zestimate_gap,

        100.0 * (zestimate - price)
        / NULLIF(price, 0)
            AS zestimate_gap_pct

    FROM analytics.zillow_listings

    WHERE zestimate IS NOT NULL
      AND price > 0
      AND zestimate > price
),

ranked_opportunities AS (
    SELECT
        *,

        ROW_NUMBER() OVER (
            ORDER BY zestimate_gap_pct DESC
        ) AS opportunity_rank,

        PERCENT_RANK() OVER (
            ORDER BY zestimate_gap_pct
        ) AS opportunity_percentile

    FROM opportunity_data
)

SELECT
    zpid,
    zipcode,
    city,
    home_type,
    price,
    zestimate,

    ROUND(
        zestimate_gap::NUMERIC,
        2
    ) AS zestimate_gap,

    ROUND(
        zestimate_gap_pct::NUMERIC,
        2
    ) AS zestimate_gap_pct,

    opportunity_rank,

    ROUND(
        (100.0 * opportunity_percentile)::NUMERIC,
        2
    ) AS opportunity_percentile,

    living_area,
    bedrooms,
    bathrooms

FROM ranked_opportunities

ORDER BY opportunity_rank;
