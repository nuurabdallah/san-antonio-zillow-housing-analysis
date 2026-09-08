# Part 8 — Visualization Key Findings

## Analytical Population

- The visualization layer uses 810 analytical properties.
- There are 810 unique properties and 0 duplicate zpids.

## Listing Price Distribution

- Median listing price: $309,950.
- Mean listing price: $418,156.
- Median logPrice: 12.6442.

The raw listing-price distribution is visibly right-skewed, supporting the use of logPrice for several relationship analyses.

## Key Feature Relationships

- Spearman correlation between living area and logPrice: 0.7813.
- Spearman correlation between tax-assessed value and logPrice: 0.8970.

These visuals reinforce the statistical analysis showing that property size and tax-assessed value are strongly associated with listing price.

## Geographic Variation

- 31 ZIP codes meet the sufficient-sample threshold of at least 10 listings.
- Highest median listing price among sufficient-sample ZIPs: $3,475,000.
- Lowest median listing price among sufficient-sample ZIPs: $179,000.

The ZIP-level visuals show substantial geographic variation in both median listing price and median price per square foot.

## Property Type

Property-type price distributions show meaningful differences in central tendency, while sample sizes vary considerably across types.

### Property-Type Sample Sizes

| Property Type | Listings | Median Price | Mean Price |
|---|---:|---:|---:|
| MULTI_FAMILY | 22 | $387,500 | $374,720 |
| SINGLE_FAMILY | 761 | $314,900 | $426,538 |
| CONDO | 13 | $249,777 | $232,775 |
| TOWNHOUSE | 10 | $232,000 | $245,740 |
| MANUFACTURED | 4 | $57,498 | $95,999 |

The large difference in sample sizes should be considered when interpreting property-type comparisons.

## Zestimate Analysis

- Complete listing-price/Zestimate pairs: 576.
- Mean Zestimate gap: $-11,800.
- Median Zestimate gap: $-5,300.
- Potential opportunity records identified by the dataset's opportunity-status logic: 16.

The listing-price-versus-Zestimate chart and Zestimate-gap distribution provide a visual view of where listing prices differ from Zillow's estimated values.

## Interpretation Guidance

These visualizations describe patterns in the observed Zillow listing dataset. They should not be interpreted as proof of causation. Geographic differences may reflect differences in property mix, neighborhood characteristics, and other factors not captured in the dataset.
