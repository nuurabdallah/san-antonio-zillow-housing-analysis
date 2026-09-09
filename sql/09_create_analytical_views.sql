/* ============================================================
   SAN ANTONIO ZILLOW HOUSING ANALYSIS
   09_create_analytical_views.sql

   Purpose:
   Create the final reusable analytical view used by Tableau.

   Source:
   analytics.zillow_listings

   Final Tableau layer:
   analytics.v_tableau_listings

   Key principles:
   - Preserve all valid listings
   - One row per Zillow property
   - Retain sparse ZIP codes
   - Do not treat missing Zestimate as zero
   - Keep Zestimate opportunity metrics separate from ML
   ============================================================ */


/* ============================================================
   1. CREATE TABLEAU-READY LISTING VIEW
   ============================================================ */

CREATE OR REPLACE VIEW analytics.vv_tableau_listings AS

WITH listing_base AS (

    SELECT
        zpid,
        zipcode,
        city,
        state,
        latitude,
        longitude,

        price,
        bedrooms,
        bathrooms,
        living_area,

        home_type,
        home_status,
        days_on_zillow,

        zestimate,

        tax_assessed_value,

        lot_area_value,
        lot_area_unit,
        lot_area_sqft,

        price_change,
        has_price_change,

        price_per_sqft,
        price_quality_flag,
        log_price,

        /* ----------------------------------------------------
           Price segmentation
           ---------------------------------------------------- */
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
        END AS price_segment

    FROM analytics.zillow_listings

    WHERE price IS NOT NULL
),


/* ============================================================
   2. CALCULATE ZIP-LEVEL LISTING COUNTS
   ============================================================ */

zip_counts AS (

    SELECT
        zipcode,
        COUNT(*) AS zip_listing_count

    FROM analytics.zillow_listings

    WHERE zipcode IS NOT NULL

    GROUP BY zipcode
),


/* ============================================================
   3. ADD ZIP SAMPLE STATUS AND PROPERTY RANKINGS
   ============================================================ */

ranked_listings AS (

    SELECT
        lb.*,

        zc.zip_listing_count,

        /* ----------------------------------------------------
           ZIP sample-size methodology

           10+ listings = Sufficient Sample
           <10 listings = Sparse Sample
           ---------------------------------------------------- */
        CASE
            WHEN zc.zip_listing_count >= 10
                THEN 'Sufficient Sample'

            ELSE 'Sparse Sample'
        END AS zip_sample_status,


        /* ----------------------------------------------------
           Property price rank within ZIP
           ---------------------------------------------------- */
        ROW_NUMBER() OVER (
            PARTITION BY lb.zipcode
            ORDER BY lb.price DESC
        ) AS price_rank_within_zip,


        /* ----------------------------------------------------
           Property price-per-square-foot rank within ZIP
           ---------------------------------------------------- */
        RANK() OVER (
            PARTITION BY lb.zipcode
            ORDER BY lb.price_per_sqft DESC NULLS LAST
        ) AS price_per_sqft_rank_within_zip

    FROM listing_base lb

    LEFT JOIN zip_counts zc
        ON lb.zipcode = zc.zipcode
)


/* ============================================================
   4. ADD ZESTIMATE OPPORTUNITY INFORMATION
   ============================================================ */

SELECT

    r.*,

    z.zestimate_gap,
    z.zestimate_gap_pct,
    z.opportunity_status,
    z.deal_score

FROM ranked_listings r

LEFT JOIN analytics.v_zestimate_opportunities z
    ON r.zpid = z.zpid;


/* ============================================================
   END OF ANALYTICAL VIEW CREATION
   ============================================================ */
