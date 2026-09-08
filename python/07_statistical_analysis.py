# ============================================================
# PART 7 — STATISTICAL ANALYSIS & HYPOTHESIS TESTING
# San Antonio Zillow Housing Analysis
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# PROJECT PATHS / SETTINGS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "San_Antonio_Zillow_Feature_Engineered.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "statistical_analysis"
ALPHA = 0.05
MIN_ZIP_SAMPLE = 10

REQUIRED_COLUMNS = [
    "zpid", "zipcode", "price", "logPrice", "area", "beds", "baths",
    "homeType", "lotAreaSqFt", "taxAssessedValue", "daysOnZillow",
    "zestimate", "zestimate_gap", "zestimate_gap_pct", "opportunity_status"
]

# ============================================================
# LOAD / VALIDATE
# ============================================================
def load_dataset():
    print("=" * 70)
    print("PART 7 — STATISTICAL ANALYSIS & HYPOTHESIS TESTING")
    print("=" * 70)
    print("\nSAN ANTONIO ZILLOW STATISTICAL ANALYSIS")
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Input file: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Feature-engineered dataset not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    print(f"\nRows loaded: {len(df)}")
    print(f"Columns loaded: {len(df.columns)}")

    if len(df) != 810:
        raise ValueError(f"Expected 810 rows, found {len(df)}.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Required columns are missing:\n" + "\n".join(missing))

    print("\nRequired statistical-analysis columns validated.")
    return df


def validate_integrity(df):
    print("\n" + "=" * 70)
    print("DATASET INTEGRITY VALIDATION")
    print("=" * 70)
    unique_zpid = df["zpid"].nunique()
    duplicate_zpid = df["zpid"].duplicated().sum()
    print(f"Rows: {len(df)}")
    print(f"Unique zpid: {unique_zpid}")
    print(f"Duplicate zpid: {duplicate_zpid}")
    if unique_zpid != 810 or duplicate_zpid != 0:
        raise ValueError("Dataset integrity validation failed.")
    print("\nDataset integrity validation passed.")

# ============================================================
# HELPERS
# ============================================================
def sig_label(p):
    return "Statistically Significant" if pd.notna(p) and p < ALPHA else "Not Statistically Significant"


def mean_ci(values, confidence=0.95):
    x = pd.Series(values).dropna().astype(float)
    if len(x) < 2:
        return np.nan, np.nan
    return stats.t.interval(confidence, len(x) - 1, loc=x.mean(), scale=stats.sem(x))


def pooled_sd(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))


def cohens_d(a, b):
    sd = pooled_sd(a, b)
    if pd.isna(sd) or sd == 0:
        return np.nan
    return (pd.Series(a).mean() - pd.Series(b).mean()) / sd


def effect_label(value, kind):
    if pd.isna(value):
        return "Not available"
    v = abs(value)
    if kind == "d":
        return "Negligible" if v < .20 else "Small" if v < .50 else "Medium" if v < .80 else "Large"
    return "Negligible" if v < .01 else "Small" if v < .06 else "Medium" if v < .14 else "Large"


def eta_squared(f, k, n):
    den = f * (k - 1) + (n - k)
    return np.nan if den == 0 else (f * (k - 1)) / den


def epsilon_squared(h, k, n):
    return np.nan if n <= k else (h - k + 1) / (n - k)

# ============================================================
# 1. CORRELATION HYPOTHESIS TESTS
# ============================================================
def correlation_tests(df):
    print("\nRunning correlation hypothesis tests...")
    rows = []
    for variable in ["area", "beds", "baths", "lotAreaSqFt", "taxAssessedValue", "daysOnZillow"]:
        x = df[[variable, "logPrice"]].dropna()
        if len(x) < 3:
            continue
        r, rp = stats.pearsonr(x[variable], x.logPrice)
        rho, rhop = stats.spearmanr(x[variable], x.logPrice)
        rows.append({
            "variable": variable, "dependent_variable": "logPrice", "n": len(x),
            "pearson_r": r, "pearson_p_value": rp, "pearson_result": sig_label(rp),
            "spearman_rho": rho, "spearman_p_value": rhop, "spearman_result": sig_label(rhop)
        })
    return pd.DataFrame(rows)

# ============================================================
# 2. BEDROOM GROUP COMPARISON
# ============================================================
def bedroom_comparison(df):
    print("\nRunning bedroom-group hypothesis tests...")
    a = df.loc[df.beds == 3, "logPrice"].dropna()
    b = df.loc[df.beds >= 4, "logPrice"].dropna()
    t, tp = stats.ttest_ind(a, b, equal_var=False)
    u, up = stats.mannwhitneyu(a, b, alternative="two-sided")
    d = cohens_d(a, b)
    ca = mean_ci(a); cb = mean_ci(b)
    return pd.DataFrame([{
        "group_1": "3 Bedrooms", "group_2": "4+ Bedrooms", "n_group_1": len(a), "n_group_2": len(b),
        "mean_log_price_group_1": a.mean(), "mean_log_price_group_2": b.mean(),
        "median_log_price_group_1": a.median(), "median_log_price_group_2": b.median(),
        "welch_t_statistic": t, "welch_t_p_value": tp, "welch_t_result": sig_label(tp),
        "mann_whitney_u": u, "mann_whitney_p_value": up, "mann_whitney_result": sig_label(up),
        "cohens_d": d, "cohens_d_interpretation": effect_label(d, "d"),
        "group_1_mean_ci_95_lower": ca[0], "group_1_mean_ci_95_upper": ca[1],
        "group_2_mean_ci_95_lower": cb[0], "group_2_mean_ci_95_upper": cb[1]
    }])

# ============================================================
# 3. PROPERTY TYPE COMPARISON
# ============================================================
def property_type_comparison(df):
    print("\nRunning property-type hypothesis tests...")
    groups = [g.logPrice.dropna() for _, g in df.groupby("homeType", dropna=True) if g.logPrice.notna().sum() >= 2]
    all_values = pd.concat(groups)
    lev, levp = stats.levene(*groups, center="median")
    f, fp = stats.f_oneway(*groups)
    h, hp = stats.kruskal(*groups)
    eta = eta_squared(f, len(groups), len(all_values))
    eps = epsilon_squared(h, len(groups), len(all_values))
    return pd.DataFrame([{
        "number_of_property_type_groups": len(groups), "total_n": len(all_values),
        "levene_statistic": lev, "levene_p_value": levp, "levene_result": sig_label(levp),
        "anova_f_statistic": f, "anova_p_value": fp, "anova_result": sig_label(fp),
        "eta_squared": eta, "eta_squared_interpretation": effect_label(eta, "eta"),
        "kruskal_h_statistic": h, "kruskal_p_value": hp, "kruskal_result": sig_label(hp),
        "epsilon_squared": eps, "epsilon_squared_interpretation": effect_label(eps, "epsilon")
    }])

# ============================================================
# 4. ZESTIMATE PAIRED COMPARISON
# ============================================================
def zestimate_comparison(df):
    print("\nRunning Zestimate paired hypothesis tests...")
    x = df[["price", "zestimate"]].dropna()
    log_price = np.log(x.price)
    log_z = np.log(x.zestimate)
    diff = log_z - log_price
    raw_gap = x.zestimate - x.price
    t, tp = stats.ttest_rel(log_z, log_price)
    w, wp = stats.wilcoxon(log_z, log_price, alternative="two-sided")
    ci = mean_ci(diff)
    return pd.DataFrame([{
        "n": len(x), "mean_listing_price": x.price.mean(), "median_listing_price": x.price.median(),
        "mean_zestimate": x.zestimate.mean(), "median_zestimate": x.zestimate.median(),
        "mean_raw_gap": raw_gap.mean(), "median_raw_gap": raw_gap.median(),
        "mean_gap_percentage": (raw_gap / x.price).mean(), "median_gap_percentage": (raw_gap / x.price).median(),
        "mean_log_difference": diff.mean(), "median_log_difference": diff.median(),
        "mean_log_difference_ci_95_lower": ci[0], "mean_log_difference_ci_95_upper": ci[1],
        "paired_t_statistic": t, "paired_t_p_value": tp, "paired_t_result": sig_label(tp),
        "wilcoxon_statistic": w, "wilcoxon_p_value": wp, "wilcoxon_result": sig_label(wp)
    }])

# ============================================================
# 5. ZIP CODE COMPARISON
# ============================================================
def zip_comparison(df):
    print("\nRunning sufficient-sample ZIP-code hypothesis tests...")
    counts = df.groupby("zipcode").size()
    sufficient = counts[counts >= MIN_ZIP_SAMPLE].index
    groups = [df.loc[df.zipcode == z, "logPrice"].dropna() for z in sufficient]
    groups = [g for g in groups if len(g) >= MIN_ZIP_SAMPLE]
    all_values = pd.concat(groups)
    lev, levp = stats.levene(*groups, center="median")
    f, fp = stats.f_oneway(*groups)
    h, hp = stats.kruskal(*groups)
    eta = eta_squared(f, len(groups), len(all_values))
    eps = epsilon_squared(h, len(groups), len(all_values))
    return pd.DataFrame([{
        "minimum_zip_sample_size": MIN_ZIP_SAMPLE, "number_of_sufficient_sample_zips": len(groups), "total_n": len(all_values),
        "levene_statistic": lev, "levene_p_value": levp, "levene_result": sig_label(levp),
        "anova_f_statistic": f, "anova_p_value": fp, "anova_result": sig_label(fp),
        "eta_squared": eta, "eta_squared_interpretation": effect_label(eta, "eta"),
        "kruskal_h_statistic": h, "kruskal_p_value": hp, "kruskal_result": sig_label(hp),
        "epsilon_squared": eps, "epsilon_squared_interpretation": effect_label(eps, "epsilon")
    }])

# ============================================================
# 6. GROUP DESCRIPTIVES / CONFIDENCE INTERVALS
# ============================================================
def group_descriptives(df):
    rows = []
    definitions = {"3 Bedrooms": df.beds == 3, "4+ Bedrooms": df.beds >= 4}
    for name, mask in definitions.items():
        x = df.loc[mask, "logPrice"].dropna(); ci = mean_ci(x)
        rows.append({"analysis": "Bedroom Group", "group": name, "n": len(x), "mean_log_price": x.mean(),
                     "median_log_price": x.median(), "std_log_price": x.std(), "mean_price": np.exp(x).mean(),
                     "median_price": np.exp(x).median(), "mean_ci_95_lower_log": ci[0], "mean_ci_95_upper_log": ci[1]})
    for name, g in df.groupby("homeType", dropna=True):
        x = g.logPrice.dropna(); ci = mean_ci(x)
        rows.append({"analysis": "Property Type", "group": name, "n": len(x), "mean_log_price": x.mean(),
                     "median_log_price": x.median(), "std_log_price": x.std(), "mean_price": np.exp(x).mean(),
                     "median_price": np.exp(x).median(), "mean_ci_95_lower_log": ci[0], "mean_ci_95_upper_log": ci[1]})
    return pd.DataFrame(rows)


def confidence_intervals(df):
    rows = []
    for v in ["price", "logPrice", "area", "beds", "baths", "lotAreaSqFt", "taxAssessedValue", "daysOnZillow", "pricePerSqFt", "zestimate"]:
        x = df[v].dropna(); ci = mean_ci(x)
        rows.append({"variable": v, "n": len(x), "mean": x.mean(), "median": x.median(), "std": x.std(),
                     "mean_ci_95_lower": ci[0], "mean_ci_95_upper": ci[1]})
    return pd.DataFrame(rows)


def effect_summary(bed, prop, zips):
    rows = [{"analysis": "3 Bedrooms vs 4+ Bedrooms", "effect_measure": "Cohen's d", "effect_size": bed.iloc[0].cohens_d,
             "interpretation": bed.iloc[0].cohens_d_interpretation}]
    for label, frame in [("Property Type", prop), ("Sufficient-Sample ZIP Codes", zips)]:
        r = frame.iloc[0]
        rows += [
            {"analysis": label, "effect_measure": "Eta squared", "effect_size": r.eta_squared, "interpretation": r.eta_squared_interpretation},
            {"analysis": label, "effect_measure": "Epsilon squared", "effect_size": r.epsilon_squared, "interpretation": r.epsilon_squared_interpretation},
        ]
    return pd.DataFrame(rows)

# ============================================================
# DOCUMENTATION
# ============================================================
def methodology():
    return """# Part 7 — Statistical Analysis & Hypothesis Testing Methodology

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
"""


def findings(df, corr, bed, prop, zest, zips):
    lines = [
        "# Part 7 — Statistical Analysis & Hypothesis Testing Findings\n",
        "## Dataset\n",
        f"- Analytical records: {len(df):,}",
        f"- Unique properties: {df.zpid.nunique():,}",
        f"- Duplicate zpid values: {df.zpid.duplicated().sum():,}",
        f"- Significance level: alpha = {ALPHA}\n",
        "## Correlation Tests\n",
    ]
    for _, r in corr.iterrows():
        lines.append(f"- **{r.variable} vs logPrice:** Pearson r = {r.pearson_r:.4f}, p = {r.pearson_p_value:.6g}; Spearman rho = {r.spearman_rho:.4f}, p = {r.spearman_p_value:.6g}.")
    b = bed.iloc[0]; lines += [
        "\n## Bedroom Group Comparison\n",
        f"- 3 Bedrooms: n = {int(b.n_group_1):,}; 4+ Bedrooms: n = {int(b.n_group_2):,}.",
        f"- Welch t-test: p = {b.welch_t_p_value:.6g} ({b.welch_t_result}).",
        f"- Mann–Whitney U: p = {b.mann_whitney_p_value:.6g} ({b.mann_whitney_result}).",
        f"- Cohen's d = {b.cohens_d:.4f} ({b.cohens_d_interpretation}).",
    ]
    p = prop.iloc[0]; lines += [
        "\n## Property Type Comparison\n",
        f"- ANOVA: p = {p.anova_p_value:.6g} ({p.anova_result}).",
        f"- Kruskal–Wallis: p = {p.kruskal_p_value:.6g} ({p.kruskal_result}).",
        f"- Eta squared = {p.eta_squared:.4f} ({p.eta_squared_interpretation}).",
    ]
    z = zest.iloc[0]; lines += [
        "\n## Zestimate vs Listing Price\n",
        f"- Complete Zestimate pairs: {int(z.n):,}.",
        f"- Paired t-test: p = {z.paired_t_p_value:.6g} ({z.paired_t_result}).",
        f"- Wilcoxon: p = {z.wilcoxon_p_value:.6g} ({z.wilcoxon_result}).",
        f"- Mean raw Zestimate gap: ${z.mean_raw_gap:,.0f}.",
        f"- Median raw Zestimate gap: ${z.median_raw_gap:,.0f}.",
    ]
    q = zips.iloc[0]
    zip_lines = [
        "\n## Sufficient-Sample ZIP Comparison\n",
        f"- Sufficient-sample ZIP codes tested: {int(q.number_of_sufficient_sample_zips):,}.",
        f"- ANOVA: p = {q.anova_p_value:.6g} ({q.anova_result}).",
        f"- Kruskal–Wallis: p = {q.kruskal_p_value:.6g} ({q.kruskal_result}).",
        f"- Eta squared = {q.eta_squared:.4f} ({q.eta_squared_interpretation}).",
        f"- Levene's test: p = {q.levene_p_value:.6g} ({q.levene_result}), indicating unequal variances across ZIP-code groups."
    ]

    if q.levene_p_value < ALPHA:
        zip_lines.append(
            "- Because the equal-variance assumption is violated, the Kruskal–Wallis result provides important nonparametric support for the observed geographic differences."
        )

    lines += zip_lines + [
        "\n## Interpretation Guidance\n",
        "Statistically significant results indicate evidence of an association or difference within this dataset. They do not prove causation. Effect sizes, confidence intervals, sample sizes, and robustness tests should be considered before drawing business conclusions."
    ]
    return "\n".join(lines)

# ============================================================
# MAIN
# ============================================================
def main():
    df = load_dataset()
    validate_integrity(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nCreating statistical analyses...")
    corr = correlation_tests(df)
    bed = bedroom_comparison(df)
    prop = property_type_comparison(df)
    zest = zestimate_comparison(df)
    zips = zip_comparison(df)
    groups = group_descriptives(df)
    cis = confidence_intervals(df)
    effects = effect_summary(bed, prop, zips)

    groups.to_csv(OUTPUT_DIR / "group_descriptive_statistics.csv", index=False)
    corr.to_csv(OUTPUT_DIR / "correlation_hypothesis_tests.csv", index=False)
    bed.to_csv(OUTPUT_DIR / "bedroom_group_comparison.csv", index=False)
    prop.to_csv(OUTPUT_DIR / "property_type_comparison.csv", index=False)
    zest.to_csv(OUTPUT_DIR / "zestimate_paired_comparison.csv", index=False)
    zips.to_csv(OUTPUT_DIR / "zip_group_comparison.csv", index=False)
    cis.to_csv(OUTPUT_DIR / "confidence_intervals.csv", index=False)
    effects.to_csv(OUTPUT_DIR / "effect_sizes.csv", index=False)
    (OUTPUT_DIR / "STATISTICAL_METHODOLOGY.md").write_text(methodology(), encoding="utf-8")
    (OUTPUT_DIR / "STATISTICAL_KEY_FINDINGS.md").write_text(findings(df, corr, bed, prop, zest, zips), encoding="utf-8")

    expected = [
        "group_descriptive_statistics.csv", "correlation_hypothesis_tests.csv",
        "bedroom_group_comparison.csv", "property_type_comparison.csv",
        "zestimate_paired_comparison.csv", "zip_group_comparison.csv",
        "confidence_intervals.csv", "effect_sizes.csv",
        "STATISTICAL_METHODOLOGY.md", "STATISTICAL_KEY_FINDINGS.md"
    ]
    missing = [f for f in expected if not (OUTPUT_DIR / f).exists()]
    if missing:
        raise FileNotFoundError("Expected outputs were not created:\n" + "\n".join(missing))

    print("\n" + "=" * 70)
    print("PART 7 STATISTICAL ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nFinal analytical records: {len(df):,}")
    print(f"Unique properties: {df.zpid.nunique():,}")
    print(f"Significance level: alpha = {ALPHA}")
    print("\nAll expected Part 7 output files validated.")
    print("Statistical results, effect sizes, confidence intervals, methodology, and findings were saved.")
    print("\nThe feature-engineered dataset was not modified.")


if __name__ == "__main__":
    main()
