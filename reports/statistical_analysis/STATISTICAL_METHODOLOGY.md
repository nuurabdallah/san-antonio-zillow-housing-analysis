# Part 7 — Statistical Analysis & Hypothesis Testing Methodology

## Purpose
Part 7 evaluates whether important relationships and group differences identified during EDA are statistically supported.

Source: `data/processed/San_Antonio_Zillow_Feature_Engineered.csv`

Analytical population: 810 unique properties. No observations are removed.

## Significance Level
Alpha = 0.05. A p-value below 0.05 is classified as statistically significant. Statistical significance does not establish causation.

## Price Variable
`logPrice = ln(Listing Price)` is used for price-based inferential analysis where appropriate because listing prices are right-skewed.

## Tests
- Pearson and Spearman correlations between `logPrice` and area, beds, baths, lotAreaSqFt, taxAssessedValue, and daysOnZillow.
- Welch independent-samples t-test and Mann–Whitney U test for 3-bedroom versus 4+ bedroom properties.
- One-way ANOVA, Kruskal–Wallis, and Levene's test across property types.
- Paired t-test and Wilcoxon signed-rank test comparing log Zestimate with log listing price for properties with Zestimate data.
- One-way ANOVA, Kruskal–Wallis, and Levene's test across ZIP codes with at least 10 listings.

## Effect Sizes
Cohen's d is reported for the bedroom comparison. Eta squared and epsilon squared are reported for multi-group comparisons.

## Confidence Intervals
95% confidence intervals are reported for key means and group comparisons.

## Limitations
This is observational Zillow listing data. Statistical association does not establish causation. ZIP-level results should be interpreted with sample size and property-type composition in mind. Zestimate analysis is limited to listings with available Zestimate values.
