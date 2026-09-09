-- San Antonio Zillow Housing Analysis
-- Analytical table creation
-- Purpose: Convert the raw staging data into a typed analytical table.

DROP TABLE IF EXISTS analytics.zillow_listings;

CREATE TABLE analytics.zillow_listings AS
SELECT
    NULLIF(TRIM(zpid), '')::BIGINT AS zpid,
    NULLIF(TRIM(addresszipcode), '') AS zipcode,
    NULLIF(TRIM(addresscity), '') AS city,
    NULLIF(TRIM(addressstate), '') AS state,

    CASE
        WHEN latlong ~ 'latitude:\s*-?[0-9]+(\.[0-9]+)?'
        THEN substring(
            latlong
            FROM 'latitude:\s*(-?[0-9]+(\.[0-9]+)?)'
        )::NUMERIC(10,6)
        ELSE NULL
    END AS latitude,

    CASE
        WHEN latlong ~ 'longitude:\s*-?[0-9]+(\.[0-9]+)?'
        THEN substring(
            latlong
            FROM 'longitude:\s*(-?[0-9]+(\.[0-9]+)?)'
        )::NUMERIC(10,6)
        ELSE NULL
    END AS longitude,

    CASE
        WHEN UPPER(TRIM(price)) ~ '^\$?[0-9]+(\.[0-9]+)?K$'
        THEN REPLACE(
            REPLACE(UPPER(TRIM(price)), '$', ''),
            'K',
            ''
        )::NUMERIC * 1000

        WHEN UPPER(TRIM(price)) ~ '^\$?[0-9]+(\.[0-9]+)?M$'
        THEN REPLACE(
            REPLACE(UPPER(TRIM(price)), '$', ''),
            'M',
            ''
        )::NUMERIC * 1000000

        WHEN TRIM(price) ~ '^\$?[0-9,]+(\.[0-9]+)?$'
        THEN REPLACE(
            REPLACE(TRIM(price), '$', ''),
            ',',
            ''
        )::NUMERIC

        ELSE NULL
    END AS price,

    NULLIF(TRIM(beds), '')::NUMERIC AS bedrooms,
    NULLIF(TRIM(baths), '')::NUMERIC AS bathrooms,
    NULLIF(TRIM(area), '')::NUMERIC AS living_area,
    NULLIF(TRIM(hometype), '') AS home_type,
    NULLIF(TRIM(rawhomestatuscd), '') AS home_status,
    NULLIF(TRIM(daysonzillow), '')::NUMERIC::INTEGER AS days_on_zillow,
    NULLIF(TRIM(zestimate), '')::NUMERIC AS zestimate,
    NULLIF(TRIM(taxassessedvalue), '')::NUMERIC AS tax_assessed_value,
    NULLIF(TRIM(lotareavalue), '')::NUMERIC AS lot_area_value,
    NULLIF(TRIM(lotareaunit), '') AS lot_area_unit,
    NULLIF(TRIM(pricechange), '')::NUMERIC AS price_change,
    NULLIF(TRIM(timeonzillow), '')::NUMERIC::BIGINT AS time_on_zillow,
    NULLIF(TRIM(lotareasqft), '')::NUMERIC AS lot_area_sqft,
    NULLIF(TRIM(pricepersqft), '')::NUMERIC AS price_per_sqft,
    NULLIF(TRIM(haspricechange), '')::NUMERIC::INTEGER AS has_price_change,
    NULLIF(TRIM(pricequalityflag), '') AS price_quality_flag,
    NULLIF(TRIM(logprice), '')::NUMERIC AS log_price

FROM staging.zillow_analysis_raw;
