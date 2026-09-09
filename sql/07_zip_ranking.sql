-- San Antonio Zillow Housing Analysis
-- ZIP-Code Ranking Analysis
-- Purpose: Rank sufficiently sampled ZIP codes across multiple
-- housing-market dimensions using SQL window functions.

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

    GROUP BY zipcode

    HAVING COUNT(*) >= 10
),

ranked_zip AS (
    SELECT
        *,

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

SELECT *
FROM ranked_zip
ORDER BY price_rank;
