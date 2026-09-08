# Part 6 — Exploratory Data Analysis Findings

## Dataset

The feature-engineered analytical dataset contains **810 unique
San Antonio Zillow property listings**.

No observations were removed during exploratory analysis.

## Market Overview

- Median listing price: $309,950
- Mean listing price: $418,156
- Median living area: 2,000 sq ft
- Mean living area: 2,251 sq ft
- Median price per square foot: $160.38
- Median days on Zillow: 23 days

## Price Segmentation

The largest price segment is **$150K-$300K**,
containing 324 listings
(40.00% of the dataset).

## Property Types

The most represented property type is **SINGLE_FAMILY**,
with 761 listings
(93.95% of the dataset).

## ZIP Code Analysis

ZIP-level comparisons use the established minimum sample size of
10 listings.

The highest-median-price sufficient-sample ZIP is 78257, with a median listing price of $3,475,000.

The lowest-median-price sufficient-sample ZIP is 78223, with a median listing price of $179,000.

## Zestimate Analysis

Zestimate information is available for
576 listings
(71.11% coverage).

There are 16 listings where Zestimate exceeds
listing price and therefore meet the project's definition of a
Potential Opportunity.

## Correlation Analysis

The strongest absolute Pearson correlation among the analyzed variables
is between **price** and
**zestimate**, with a correlation of
0.9986.

Correlation measures describe association and do not establish
causation.

## Analytical Considerations

Several variables are strongly related to listing price, but these
relationships should not automatically be interpreted as causal.

ZIP-level comparisons should account for sample size and
property-type composition.

Zestimate-based opportunity analysis is limited to listings with
available Zestimate information.

The exploratory analysis provides the foundation for subsequent
statistical modeling, machine learning, visualization, and business
analysis.
