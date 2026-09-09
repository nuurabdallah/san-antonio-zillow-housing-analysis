-- San Antonio Zillow Housing Analysis
-- Zestimate Analysis
-- Purpose: Evaluate the relationship between listing price and Zestimate
-- and identify potential pricing opportunities.

-- ============================================================
-- 1. Zestimate Coverage and Overall Gap Summary
-- ============================================================

SELECT
    COUNT(*) AS total_listings,

    COUNT(zestimate) AS listings_with_zestimate,

    COUNT(*) - COUNT(zestimate)
        AS listings_without_zestimate,

    ROUND(
        100.0 * COUNT(zestimate)
        / NULLIF(COUNT(*), 0),
        2
    ) AS zestimate_coverage_pct,

    ROUND(
        AVG(zestimate - price)
        FILTER (
            WHERE zestimate IS NOT NULL
        ),
        2
    ) AS average_zestimate_gap,

    ROUND(
        PERCENTILE_CONT(0.50)
        WITHIN GROUP (
            ORDER BY zestimate - price
        ) FILTER (
            WHERE zestimate IS NOT NULL
        )::NUMERIC,
        2
    ) AS median_zestimate_gap

FROM analytics.zillow_listings;


-- ============================================================
-- 2. Potential Pricing Opportunities
-- ============================================================
-- Potential Opportunity:
-- Zestimate is greater than the listing price.
--
-- Zestimate Gap:
-- Zestimate - Listing Price
--
-- Zestimate Gap %:
-- (Zestimate - Listing Price) / Listing Price * 100

WITH deal_analysis AS (
    SELECT
        zpid,
        zipcode,
        city,
        price,
        zestimate,
        living_area,
        bedrooms,
        bathrooms,
        home_type,
        price_per_sqft,

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

scored_deals AS (
    SELECT
        *,

        ROUND(
            100.0 * PERCENT_RANK()
            OVER (
                ORDER BY zestimate_gap_pct
            )::NUMERIC,
            2
        ) AS deal_score

    FROM deal_analysis
)

SELECT
    zpid,
    zipcode,
    city,
    home_type,
    price,
    zestimate,

    ROUND(zestimate_gap, 2)
        AS zestimate_gap,

    ROUND(zestimate_gap_pct, 2)
        AS zestimate_gap_pct,

    deal_score,

    living_area,
    bedrooms,
    bathrooms,
    price_per_sqft

FROM scored_deals

ORDER BY deal_score DESC

LIMIT 25;


-- ============================================================
-- 3. Potential Opportunities by ZIP Code
-- ============================================================

SELECT
    zipcode,

    COUNT(*) AS potential_opportunities,

    ROUND(
        100.0 * COUNT(*)
        / SUM(COUNT(*)) OVER (),
        2
    ) AS pct_of_opportunities,

    ROUND(
        AVG(zestimate - price),
        2
    ) AS average_positive_gap,

    ROUND(
        PERCENTILE_CONT(0.50)
        WITHIN GROUP (
            ORDER BY zestimate - price
        )::NUMERIC,
        2
    ) AS median_positive_gap,

    ROUND(
        AVG(
            100.0 * (zestimate - price)
            / NULLIF(price, 0)
        ),
        2
    ) AS average_positive_gap_pct

FROM analytics.zillow_listings

WHERE zestimate IS NOT NULL
  AND price > 0
  AND zestimate > price

GROUP BY zipcode

ORDER BY potential_opportunities DESC;


-- ============================================================
-- 4. Potential Opportunities by Property Type
-- ============================================================

SELECT
    home_type,

    COUNT(*) AS potential_opportunities,

    ROUND(
        100.0 * COUNT(*)
        / SUM(COUNT(*)) OVER (),
        2
    ) AS pct_of_opportunities,

    ROUND(
        AVG(price),
        2
    ) AS mean_listing_price,

    ROUND(
        PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY price)::NUMERIC,
        2
    ) AS median_listing_price,

    ROUND(
        AVG(zestimate - price),
        2
    ) AS mean_positive_gap,

    ROUND(
        PERCENTILE_CONT(0.50)
        WITHIN GROUP (
            ORDER BY zestimate - price
        )::NUMERIC,
        2
    ) AS median_positive_gap,

    ROUND(
        AVG(
            100.0 * (zestimate - price)
            / NULLIF(price, 0)
        ),
        2
    ) AS mean_positive_gap_pct

FROM analytics.zillow_listings

WHERE zestimate IS NOT NULL
  AND price > 0
  AND zestimate > price

GROUP BY home_type

ORDER BY potential_opportunities DESC;
