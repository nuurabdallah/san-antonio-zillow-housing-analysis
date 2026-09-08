# Part 8 — Visualization Methodology

## Purpose

Part 8 translates the exploratory and inferential findings from the
San Antonio Zillow Housing Analysis into a reproducible Python visualization
layer.

## Source Dataset

- Source: `data/processed/San_Antonio_Zillow_Feature_Engineered.csv`
- Analytical records: 810
- Unique properties: 810
- Duplicate `zpid`: 0

The feature-engineered analytical dataset is not modified by this script.

## Visualization Principles

The visualizations were selected to:

1. Show the distribution and shape of listing prices.
2. Visualize important relationships identified during EDA and statistical
   testing.
3. Emphasize geographic variation across ZIP codes with at least 10 listings.
4. Compare listing prices across property types.
5. Visualize the relationship between listing price and Zestimate.
6. Visualize Zestimate gaps and potential opportunity properties.
7. Provide visual support for statistically significant group comparisons.

## Transformations

`logPrice` is used for several relationship plots because listing prices are
right-skewed. This allows relationships to be viewed on a transformed scale
that is more consistent with the statistical analysis.

Raw dollar values are retained for business-facing distributions and
Zestimate-gap analysis.

## Geographic Sample Rule

ZIP-level charts use only ZIP codes containing at least 10 listings.
This matches the project's sufficient-sample definition and reduces the
risk of over-interpreting extremely small ZIP-level samples.

## Missing Data

Missing Zestimate and tax-assessed values are not artificially imputed.
Charts using these fields are based only on records with valid values for
the variables required by the specific visualization.

## Statistical Context

The visualizations are descriptive and explanatory. They do not establish
causation. Statistical significance, effect size, sample size, confidence
intervals, and data limitations should be considered alongside the charts.

## Output

The script generates 15 PNG charts in:

`reports/visualization/charts/`

The script clears previously generated PNG files before creating the new
visualizations so stale charts are not retained.

ZIP-level charts use horizontal bars with the same sufficient-sample ZIP
population as the underlying analysis. The Zestimate-gap visualization uses
the middle 98% of observations for the displayed histogram range so extreme
gaps do not obscure the central distribution; the full sample size and
percentile bounds are reported on the chart.
