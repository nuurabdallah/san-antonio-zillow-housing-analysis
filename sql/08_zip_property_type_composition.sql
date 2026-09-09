-- San Antonio Zillow Housing Analysis
-- ZIP × Property-Type Composition
-- Purpose: Analyze the composition of each ZIP-code housing market.
-- The percentage represents each property type's share of its ZIP.

SELECT
    zipcode,
    home_type,

    COUNT(*) AS listing_count,

    ROUND(
        100.0 * COUNT(*)
        / SUM(COUNT(*)) OVER (PARTITION BY zipcode),
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
