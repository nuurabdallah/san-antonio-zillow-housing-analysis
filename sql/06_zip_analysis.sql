-- San Antonio Zillow Housing Analysis
-- ZIP-Code Analysis
-- Purpose: Analyze housing-market characteristics by ZIP code.
-- Methodology: ZIP codes with fewer than 10 listings are classified
-- as Sparse Sample but remain in the analytical dataset.

-- ============================================================
-- 1. ZIP-Code Market Analysis
-- ============================================================

WITH zip_base AS (
    SELECT
        zipcode,

        COUNT(*) AS listing_count,

        AVG(price) AS average_price,

        PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY price) AS median_price,

        AVG(living_area) AS average_living_area,

        PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY living_area)
        AS median_living_area,

        AVG(price_per_sqft) AS average_price_per_sqft,

        PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY price_per_sqft)
        AS median_price_per_sqft,

        AVG(days_on_zillow) AS average_days_on_zillow,

        COUNT(zestimate) AS listings_with_zestimate,

        COUNT(
            CASE
                WHEN zestimate > price THEN 1
            END
        ) AS potential_opportunities

    FROM analytics.zillow_listings

    WHERE zipcode IS NOT NULL

    GROUP BY zipcode
)

SELECT
    zipcode,

    listing_count,

    CASE
        WHEN listing_count >= 10
            THEN 'Sufficient Sample'
        ELSE 'Sparse Sample'
    END AS zip_sample_status,

    ROUND(average_price::NUMERIC, 2)
        AS average_price,

    ROUND(median_price::NUMERIC, 2)
        AS median_price,

    ROUND(average_living_area::NUMERIC, 2)
        AS average_living_area,

    ROUND(median_living_area::NUMERIC, 2)
        AS median_living_area,

    ROUND(average_price_per_sqft::NUMERIC, 2)
        AS average_price_per_sqft,

    ROUND(median_price_per_sqft::NUMERIC, 2)
        AS median_price_per_sqft,

    ROUND(average_days_on_zillow::NUMERIC, 2)
        AS average_days_on_zillow,

    listings_with_zestimate,

    ROUND(
        (
            100.0 * listings_with_zestimate / listing_count
        )::NUMERIC,
        2
    ) AS zestimate_coverage_pct,

    potential_opportunities

FROM zip_base

ORDER BY median_price DESC;


-- ============================================================
-- 2. Sufficient-Sample ZIP Rankings
-- ============================================================

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
    potential_opportunities

FROM (
    WITH zip_base AS (
        SELECT
            zipcode,

            COUNT(*) AS listing_count,

            PERCENTILE_CONT(0.50)
            WITHIN GROUP (ORDER BY price)
            AS median_price,

            PERCENTILE_CONT(0.50)
            WITHIN GROUP (ORDER BY price_per_sqft)
            AS median_price_per_sqft,

            PERCENTILE_CONT(0.50)
            WITHIN GROUP (ORDER BY living_area)
            AS median_living_area,

            AVG(days_on_zillow)
            AS average_days_on_zillow,

            COUNT(zestimate)
            AS listings_with_zestimate

        FROM analytics.zillow_listings

        WHERE zipcode IS NOT NULL

        GROUP BY zipcode
    )

    SELECT
        zipcode,
        listing_count,

        CASE
            WHEN listing_count >= 10
                THEN 'Sufficient Sample'
            ELSE 'Sparse Sample'
        END AS zip_sample_status,

        median_price,
        median_price_per_sqft,
        median_living_area,
        average_days_on_zillow,
        listings_with_zestimate,

        ROUND(
            (
                100.0 * listings_with_zestimate / listing_count
            )::NUMERIC,
            2
        ) AS zestimate_coverage_pct,

        (
            SELECT COUNT(*)
            FROM analytics.zillow_listings z
            WHERE z.zipcode = zip_base.zipcode
              AND z.zestimate > z.price
        ) AS potential_opportunities

    FROM zip_base
) AS ranked_zip_data

WHERE zip_sample_status = 'Sufficient Sample'

ORDER BY median_price DESC;


-- ============================================================
-- 3. ZIP Sample-Size Summary
-- ============================================================

SELECT
    zip_sample_status,
    COUNT(*) AS zip_code_count,
    SUM(listing_count) AS total_listings

FROM (
    SELECT
        zipcode,
        COUNT(*) AS listing_count,

        CASE
            WHEN COUNT(*) >= 10
                THEN 'Sufficient Sample'
            ELSE 'Sparse Sample'
        END AS zip_sample_status

    FROM analytics.zillow_listings

    WHERE zipcode IS NOT NULL

    GROUP BY zipcode
) AS zip_summary

GROUP BY zip_sample_status

ORDER BY zip_sample_status;
