/* ============================================================
   SAN ANTONIO ZILLOW HOUSING ANALYSIS
   10_final_validation.sql

   Purpose:
   Final validation of the PostgreSQL analytical layer.

   This script validates:
   1. Core analytical table integrity
   2. Major analytical view reconciliation
   3. Tableau-ready view integrity
   4. Zestimate coverage and opportunity metrics
   5. ZIP sample-size methodology
   6. Final executive business query

   Expected analytical population:
   810 properties
   810 unique Zillow property identifiers (zpid)
   0 duplicate zpids
   ============================================================ */


/* ============================================================
   1. VALIDATE THE CORE ANALYTICAL TABLE
   ============================================================ */

SELECT
    COUNT(*) AS total_rows,

    COUNT(DISTINCT zpid) AS unique_properties,

    COUNT(*) - COUNT(DISTINCT zpid)
        AS duplicate_zpid_count,

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


/* ============================================================
   EXPECTED:
   
   total_rows              = 810
   unique_properties       = 810
   duplicate_zpid_count    = 0
   missing_price           = 0
   missing_zipcode         = 0
   missing_living_area     = 0
   ============================================================ */


/* ============================================================
   2. RECONCILE MAJOR ANALYTICAL VIEWS
   ============================================================ */

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

FROM analytics.v_property_type_kpis

ORDER BY
    validation_area;


/* ============================================================
   EXPECTED:
   
   Market KPIs       = 810
   Price Segments    = 810
   ZIP Analysis      = 810
   Property Types    = 810
   ============================================================ */


/* ============================================================
   3. VALIDATE THE TABLEAU-READY VIEW
   ============================================================ */

SELECT
    COUNT(*) AS total_rows,

    COUNT(DISTINCT zpid) AS unique_properties,

    COUNT(*) - COUNT(DISTINCT zpid)
        AS duplicate_zpid_count

FROM analytics.v_tableau_listings;


/* ============================================================
   EXPECTED:
   
   total_rows              = 810
   unique_properties       = 810
   duplicate_zpid_count    = 0
   ============================================================ */


/* ============================================================
   4. VALIDATE TABLEAU BUSINESS DIMENSIONS
   ============================================================ */

SELECT
    price_segment,

    zip_sample_status,

    opportunity_status,

    COUNT(*) AS listing_count

FROM analytics.v_tableau_listings

GROUP BY
    price_segment,
    zip_sample_status,
    opportunity_status

ORDER BY
    price_segment,
    zip_sample_status,
    opportunity_status;


/* ============================================================
   PURPOSE:

   Confirms that the Tableau-ready view contains the expected
   business dimensions without dropping the analytical population.

   Sparse ZIP codes remain labeled as:
   'Sparse Sample'

   Missing Zestimate values remain distinguishable rather than
   being treated as zero.
   ============================================================ */


/* ============================================================
   5. VALIDATE ZESTIMATE COVERAGE
   ============================================================ */

SELECT
    COUNT(*) AS total_listings,

    COUNT(zestimate) AS listings_with_zestimate,

    COUNT(*) - COUNT(zestimate)
        AS listings_without_zestimate,

    ROUND(
        (
            100.0 * COUNT(zestimate)
            / NULLIF(COUNT(*), 0)
        )::NUMERIC,
        2
    ) AS zestimate_coverage_pct

FROM analytics.zillow_listings;


/* ============================================================
   EXPECTED:

   Total listings             = 810
   Listings with Zestimate    = 576
   Listings without Zestimate = 234
   Zestimate coverage         = 71.11%
   ============================================================ */


/* ============================================================
   6. VALIDATE ZESTIMATE OPPORTUNITY METRICS
   ============================================================ */

SELECT
    COUNT(*) AS listings_with_zestimate,

    COUNT(*) FILTER (
        WHERE zestimate > price
    ) AS potential_opportunities,

    COUNT(*) FILTER (
        WHERE zestimate < price
    ) AS above_zestimate,

    COUNT(*) FILTER (
        WHERE zestimate = price
    ) AS equal_to_zestimate,

    ROUND(
        (
            100.0 *
            COUNT(*) FILTER (
                WHERE zestimate > price
            )
            / NULLIF(COUNT(*), 0)
        )::NUMERIC,
        2
    ) AS opportunity_rate_pct,

    ROUND(
        AVG(
            CASE
                WHEN zestimate > price
                THEN zestimate - price
            END
        )::NUMERIC,
        2
    ) AS average_opportunity_gap,

    ROUND(
        PERCENTILE_CONT(0.50)
        WITHIN GROUP (
            ORDER BY
                CASE
                    WHEN zestimate > price
                    THEN zestimate - price
                END
        )::NUMERIC,
        2
    ) AS median_opportunity_gap,

    ROUND(
        AVG(
            CASE
                WHEN zestimate > price
                     AND price > 0
                THEN
                    (
                        (zestimate - price)
                        / price
                    ) * 100
            END
        )::NUMERIC,
        2
    ) AS average_opportunity_gap_pct,

    ROUND(
        PERCENTILE_CONT(0.50)
        WITHIN GROUP (
            ORDER BY
                CASE
                    WHEN zestimate > price
                         AND price > 0
                    THEN
                        (
                            (zestimate - price)
                            / price
                        ) * 100
                END
        )::NUMERIC,
        2
    ) AS median_opportunity_gap_pct,

    ROUND(
        SUM(
            CASE
                WHEN zestimate > price
                THEN zestimate - price
                ELSE 0
            END
        )::NUMERIC,
        2
    ) AS total_positive_gap

FROM analytics.zillow_listings

WHERE zestimate IS NOT NULL;


/* ============================================================
   EXPECTED:

   Listings with Zestimate      = 576
   Potential opportunities      = 16
   Above Zestimate              = 560
   Equal to Zestimate           = 0
   Opportunity rate             = 2.78%
   Average opportunity gap      = $57,498.19
   Median opportunity gap       = $30,700
   Average opportunity gap %    = 43.87%
   Median opportunity gap %     = 14.53%
   Total positive gap           = $919,971
   ============================================================ */


/* ============================================================
   7. VALIDATE ZIP SAMPLE-SIZE METHODOLOGY
   ============================================================ */

SELECT
    zip_sample_status,

    COUNT(*) AS total_listings,

    COUNT(DISTINCT zipcode) AS zip_code_count

FROM analytics.v_tableau_listings

GROUP BY
    zip_sample_status

ORDER BY
    zip_sample_status;


/* ============================================================
   EXPECTED:

   Sparse Sample:
       119 listings
       28 ZIP codes

   Sufficient Sample:
       691 listings
       31 ZIP codes
   ============================================================ */


/* ============================================================
   8. FINAL EXECUTIVE ZIP-CODE ANALYSIS
   ============================================================
   
   Business question:

   Which San Antonio ZIP codes represent the highest-priced
   housing markets, and what does the market look like within
   those ZIP codes?

   Only ZIP codes meeting the minimum sample-size requirement
   of 10 listings are included.
   ============================================================ */

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

ORDER BY
    median_price_rank;


/* ============================================================
   9. FINAL PROPERTY-LEVEL OPPORTUNITY CHECK
   ============================================================ */

SELECT
    COUNT(*) AS potential_opportunity_count,

    ROUND(
        SUM(zestimate_gap)::NUMERIC,
        2
    ) AS total_positive_gap,

    ROUND(
        AVG(zestimate_gap)::NUMERIC,
        2
    ) AS average_positive_gap,

    ROUND(
        PERCENTILE_CONT(0.50)
        WITHIN GROUP (
            ORDER BY zestimate_gap
        )::NUMERIC,
        2
    ) AS median_positive_gap

FROM analytics.v_tableau_listings

WHERE opportunity_status = 'Potential Opportunity';


/* ============================================================
   EXPECTED:

   potential_opportunity_count = 16

   The total, average, and median gaps should reconcile with
   the previously validated Zestimate opportunity analysis.
   ============================================================ */


/* ============================================================
   END OF FINAL SQL VALIDATION
   ============================================================ */
