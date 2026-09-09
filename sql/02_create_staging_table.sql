-- San Antonio Zillow Housing Analysis
-- Staging table for the PostgreSQL analytical workflow
-- Purpose: Preserve source-format fields before analytical type conversion.

DROP TABLE IF EXISTS staging.zillow_analysis_raw;

CREATE TABLE staging.zillow_analysis_raw (
    zpid BIGINT,
    zipcode VARCHAR(10),
    city TEXT,
    state VARCHAR(2),
    latitude NUMERIC(10,6),
    longitude NUMERIC(10,6),
    price NUMERIC(15,2),
    bedrooms NUMERIC(5,1),
    bathrooms NUMERIC(5,1),
    livingArea NUMERIC(12,2),
    homeType TEXT,
    homeStatus TEXT,
    daysOnZillow INTEGER,
    zestimate NUMERIC(15,2),
    rentZestimate NUMERIC(15,2),
    taxAssessedValue NUMERIC(15,2),
    lotAreaValue NUMERIC(15,4),
    lotAreaUnit TEXT,
    priceChange NUMERIC(15,2),
    timeOnZillow BIGINT,
    lotAreaSqFt NUMERIC(15,2),
    pricePerSqFt NUMERIC(15,4),
    hasPriceChange INTEGER,
    priceQualityFlag TEXT,
    logPrice NUMERIC(15,6)
);
