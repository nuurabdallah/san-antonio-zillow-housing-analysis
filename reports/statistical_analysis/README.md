# Part 7 — Statistical Analysis & Hypothesis Testing Findings# Part 7 — Statistical Analysis & Hypothesis Testing Findings

## Dataset

- Analytical records: 810
- Unique properties: 810
- Duplicate zpid values: 0
- Significance level: alpha = 0.05

## Correlation Tests

- **area vs logPrice:** Pearson r = 0.7189, p = 9.71816e-130; Spearman rho = 0.7813, p = 1.43185e-167.
- **beds vs logPrice:** Pearson r = 0.4503, p = 1.08501e-41; Spearman rho = 0.4496, p = 1.45991e-41.
- **baths vs logPrice:** Pearson r = 0.6831, p = 2.05044e-112; Spearman rho = 0.6142, p = 3.75436e-85.
- **lotAreaSqFt vs logPrice:** Pearson r = 0.3605, p = 1.031e-25; Spearman rho = 0.4928, p = 1.07857e-49.
- **taxAssessedValue vs logPrice:** Pearson r = 0.7361, p = 1.54221e-127; Spearman rho = 0.8970, p = 1.56569e-264.
- **daysOnZillow vs logPrice:** Pearson r = 0.0341, p = 0.333264; Spearman rho = 0.0084, p = 0.811073.

## Bedroom Group Comparison

- 3 Bedrooms: n = 360; 4+ Bedrooms: n = 403.
- Welch t-test: p = 3.91907e-29 (Statistically Significant).
- Mann–Whitney U: p = 2.98519e-29 (Statistically Significant).
- Cohen's d = -0.8348 (Large).

## Property Type Comparison

- ANOVA: p = 2.3754e-08 (Statistically Significant).
- Kruskal–Wallis: p = 0.00109765 (Statistically Significant).
- Eta squared = 0.0499 (Small).

## Zestimate vs Listing Price

- Complete Zestimate pairs: 576.
- Paired t-test: p = 4.88271e-05 (Statistically Significant).
- Wilcoxon: p = 9.16968e-81 (Statistically Significant).
- Mean raw Zestimate gap: $-11,800.
- Median raw Zestimate gap: $-5,300.

## Sufficient-Sample ZIP Comparison

- Sufficient-sample ZIP codes tested: 31.
- ANOVA: p = 4.97411e-88 (Statistically Significant).
- Kruskal–Wallis: p = 2.86826e-59 (Statistically Significant).
- Eta squared = 0.5290 (Large).
- Levene's test: p = 2.86976e-05 (Statistically Significant), indicating unequal variances across ZIP-code groups.
- Because the equal-variance assumption is violated, the Kruskal–Wallis result provides important nonparametric support for the observed geographic differences.

## Interpretation Guidance

Statistically significant results indicate evidence of an association or difference within this dataset. They do not prove causation. Effect sizes, confidence intervals, sample sizes, and robustness tests should be considered before drawing business conclusions.
