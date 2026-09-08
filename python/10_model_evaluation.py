"""
PART 10 — MODEL EVALUATION & DIAGNOSTICS
San Antonio Zillow Housing Analysis
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "San_Antonio_Zillow_Feature_Engineered.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"
CHART_DIR = OUTPUT_DIR / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "logPrice"
PREDICTORS = ["area", "beds", "baths", "lotAreaSqFt", "taxAssessedValue", "daysOnZillow"]
LEAKAGE_COLUMNS = ["price", "pricePerSqFt", "zestimate", "zestimate_gap",
                   "zestimate_gap_pct", "deal_score", "opportunity_status"]
RANDOM_STATE = 42
TEST_SIZE = 0.20
MODEL_PARAMS = {
    "n_estimators": 100, "learning_rate": 0.05, "max_depth": 3,
    "min_samples_split": 5, "min_samples_leaf": 1, "subsample": 1.0,
    "random_state": RANDOM_STATE
}

def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def mape(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100

print("=" * 70)
print("PART 10 — MODEL EVALUATION & DIAGNOSTICS")
print("=" * 70)
print("\nSAN ANTONIO ZILLOW MODEL EVALUATION")
print(f"\nProject root: {PROJECT_ROOT}")
print(f"Input file: {INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)
print(f"\nRows loaded: {len(df):,}")
print(f"Columns loaded: {df.shape[1]:,}")

required = ["zpid", "price", TARGET] + PREDICTORS
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

section("DATASET INTEGRITY VALIDATION")
unique_zpid = df["zpid"].nunique(dropna=True)
duplicates = int(df["zpid"].duplicated().sum())
print(f"Rows: {len(df):,}")
print(f"Unique zpid: {unique_zpid:,}")
print(f"Duplicate zpid: {duplicates:,}")
if len(df) != 810 or unique_zpid != 810 or duplicates != 0:
    raise ValueError("Dataset integrity validation failed.")
print("\nDataset integrity validation passed.")

section("TARGET-LEAKAGE VALIDATION")
print(f"Target: {TARGET}")
print("Predictors: " + ", ".join(PREDICTORS))
print("\nExcluded target-derived / leakage-prone variables:")
for c in LEAKAGE_COLUMNS:
    print(f"  - {c}")
if set(PREDICTORS) & set(LEAKAGE_COLUMNS) or TARGET in PREDICTORS:
    raise ValueError("Target leakage detected.")
print("\nNo target leakage detected in the approved predictor set.")

section("MACHINE-LEARNING POPULATION")
work = df[required].copy()
for c in PREDICTORS + [TARGET, "price"]:
    work[c] = pd.to_numeric(work[c], errors="coerce")
complete = work[PREDICTORS + [TARGET, "price"]].notna().all(axis=1)
ml = work.loc[complete].copy()
print(f"Full analytical population: {len(work):,}")
print(f"Complete ML cases: {len(ml):,}")
print(f"Excluded incomplete cases: {len(work)-len(ml):,}")
print(f"ML complete-case coverage: {len(ml)/len(work)*100:.2f}%")
if len(ml) != 731:
    raise ValueError(f"Expected 731 ML cases, found {len(ml)}.")

section("TRAIN / TEST SPLIT")
# Reproduce the Part 9 holdout design with sklearn's train_test_split.
# For 731 observations and test_size=0.20, sklearn allocates 147
# observations to the test set and 584 to training.
train, test = train_test_split(
    ml,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train = train.copy()
test = test.copy()

print(f"Total ML records: {len(ml):,}")
print(f"Training records: {len(train):,}")
print(f"Test records: {len(test):,}")
print(f"Test proportion: {len(test)/len(ml):.0%}")
print(f"Random state: {RANDOM_STATE}")

if len(train) != 584 or len(test) != 147:
    raise ValueError(
        f"Unexpected train/test sizes: train={len(train)}, test={len(test)}."
    )

section("SELECTED MODEL")
print("Development candidate: Tuned Gradient Boosting")
for k, v in MODEL_PARAMS.items():
    print(f"  {k}: {v}")

model = GradientBoostingRegressor(**MODEL_PARAMS)
model.fit(train[PREDICTORS], train[TARGET])
pred_log = model.predict(test[PREDICTORS])
actual_log = test[TARGET].to_numpy()
actual_price = test["price"].to_numpy()
pred_price = np.exp(pred_log)

section("HOLDOUT PERFORMANCE — LOG PRICE")
test_r2 = r2_score(actual_log, pred_log)
test_rmse_log = np.sqrt(mean_squared_error(actual_log, pred_log))
test_mae_log = mean_absolute_error(actual_log, pred_log)
print(f"R²: {test_r2:.6f}")
print(f"RMSE: {test_rmse_log:.6f}")
print(f"MAE: {test_mae_log:.6f}")
print("MAPE: reported on dollar-price scale below; logPrice is not a dollar-scale quantity.")

abs_error = np.abs(actual_price - pred_price)
signed_error = pred_price - actual_price
pct_error = abs_error / actual_price * 100
dollar_rmse = np.sqrt(mean_squared_error(actual_price, pred_price))
dollar_mae = mean_absolute_error(actual_price, pred_price)
median_ae = np.median(abs_error)
dollar_mape = mape(actual_price, pred_price)
median_ape = np.median(pct_error)

section("HOLDOUT PERFORMANCE — DOLLAR SCALE")
print(f"RMSE: ${dollar_rmse:,.2f}")
print(f"MAE: ${dollar_mae:,.2f}")
print(f"Median absolute error: ${median_ae:,.2f}")
print(f"MAPE: {dollar_mape:.2f}%")
print(f"Median APE: {median_ape:.2f}%")

section("MODEL CALIBRATION")
slope, intercept, r, p, stderr = stats.linregress(pred_price, actual_price)
print(f"Calibration slope: {slope:.6f}")
print(f"Calibration intercept: ${intercept:,.2f}")
print(f"Calibration R²: {r*r:.6f}")
print(f"Calibration slope p-value: {p:.6g}")

residual = actual_log - pred_log
pearson = stats.pearsonr(pred_log, residual)
spearman = stats.spearmanr(pred_log, np.abs(residual))
pearson_r, pearson_p = float(pearson[0]), float(pearson[1])
spearman_rho, spearman_p = float(spearman[0]), float(spearman[1])
shapiro_p = float(stats.shapiro(residual)[1])

section("RESIDUAL DIAGNOSTICS")
print(f"Residual mean: {residual.mean():.6f}")
print(f"Residual median: {np.median(residual):.6f}")
print(f"Residual standard deviation: {residual.std(ddof=1):.6f}")
print(f"Residual vs predicted Pearson r: {pearson_r:.6f}")
print(f"Residual vs predicted Pearson p-value: {pearson_p:.6g}")
print(f"Absolute residual vs predicted Spearman rho: {spearman_rho:.6f}")
print(f"Absolute residual vs predicted Spearman p-value: {spearman_p:.6g}")
print(f"Shapiro-Wilk residual normality p-value: {shapiro_p:.6g}")

eval_df = test[["zpid", "price", "logPrice"]].copy()
eval_df["predicted_logPrice"] = pred_log
eval_df["predicted_price"] = pred_price
eval_df["error"] = signed_error
eval_df["absolute_error"] = abs_error
eval_df["absolute_percentage_error"] = pct_error
eval_df["residual_logPrice"] = residual

# Price segments use training quartiles, preventing holdout-derived boundaries.
edges = np.unique(train["price"].quantile([0, .25, .50, .75, 1]).to_numpy())
labels = ["Lower Price", "Lower-Middle Price", "Upper-Middle Price", "Higher Price"]
eval_df["price_segment"] = pd.cut(eval_df["price"], bins=edges, labels=labels, include_lowest=True)

segment = eval_df.groupby("price_segment", observed=False).agg(
    records=("zpid", "count"),
    mean_actual_price=("price", "mean"),
    mean_predicted_price=("predicted_price", "mean"),
    mean_absolute_error=("absolute_error", "mean"),
    median_absolute_error=("absolute_error", "median"),
    mean_absolute_percentage_error=("absolute_percentage_error", "mean"),
    median_absolute_percentage_error=("absolute_percentage_error", "median"),
    mean_signed_error=("error", "mean"),
    rmse_price=("error", lambda x: np.sqrt(np.mean(x**2)))
).reset_index()

section("ERROR BY PRICE SEGMENT")
print(segment.to_string(index=False))

worst = eval_df.sort_values("absolute_error", ascending=False).head(20).copy()
worst["error_direction"] = np.where(worst["error"] > 0, "Overprediction", "Underprediction")

section("LARGEST HOLDOUT PREDICTION ERRORS")
print(worst[["zpid", "price", "predicted_price", "error",
             "absolute_error", "absolute_percentage_error",
             "price_segment", "error_direction"]].head(10).to_string(index=False))

fi = pd.DataFrame({"feature": PREDICTORS, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
perm = permutation_importance(model, test[PREDICTORS], actual_log, scoring="r2",
                              n_repeats=30, random_state=RANDOM_STATE)
pi = pd.DataFrame({"feature": PREDICTORS,
                   "mean_decrease_in_r2": perm.importances_mean,
                   "std_decrease_in_r2": perm.importances_std}).sort_values("mean_decrease_in_r2", ascending=False)

section("FEATURE IMPORTANCE")
print(fi.to_string(index=False))
section("PERMUTATION IMPORTANCE")
print(pi.to_string(index=False))

eval_df.to_csv(OUTPUT_DIR / "holdout_predictions.csv", index=False)
worst.to_csv(OUTPUT_DIR / "largest_prediction_errors.csv", index=False)
segment.to_csv(OUTPUT_DIR / "price_segment_error_analysis.csv", index=False)
fi.to_csv(OUTPUT_DIR / "selected_model_feature_importance.csv", index=False)
pi.to_csv(OUTPUT_DIR / "selected_model_permutation_importance.csv", index=False)

metrics = pd.DataFrame([
    ("analytical_population", len(df)),
    ("complete_ml_population", len(ml)),
    ("training_records", len(train)),
    ("holdout_test_records", len(test)),
    ("test_r2_log", test_r2),
    ("test_rmse_log", test_rmse_log),
    ("test_mae_log", test_mae_log),
    ("test_rmse_dollars", dollar_rmse),
    ("test_mae_dollars", dollar_mae),
    ("test_median_absolute_error_dollars", median_ae),
    ("test_mape_dollars_percent", dollar_mape),
    ("test_median_ape_dollars_percent", median_ape),
    ("calibration_slope", slope),
    ("calibration_intercept_dollars", intercept),
    ("calibration_r_squared", r*r),
    ("calibration_slope_p_value", p),
    ("residual_mean_log", residual.mean()),
    ("residual_std_log", residual.std(ddof=1)),
    ("residual_vs_predicted_pearson_r", pearson_r),
    ("residual_vs_predicted_pearson_p", pearson_p),
    ("absolute_residual_vs_predicted_spearman_rho", spearman_rho),
    ("absolute_residual_vs_predicted_spearman_p", spearman_p),
    ("shapiro_wilk_p", shapiro_p),
], columns=["metric", "value"])
metrics.to_csv(OUTPUT_DIR / "model_evaluation_metrics.csv", index=False)

# ------------------------------ CHARTS ------------------------------

def finish(fig, ax, title, xlabel, ylabel, filename):
    ax.set_title(title, fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    ax.grid(alpha=.20)
    fig.tight_layout()
    fig.savefig(CHART_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 9))
ax.scatter(actual_price, pred_price, alpha=.65)
lo, hi = min(actual_price.min(), pred_price.min()), max(actual_price.max(), pred_price.max())
ax.plot([lo, hi], [lo, hi], "--", linewidth=2, label="Perfect prediction")
ax.legend()
finish(fig, ax, "Held-Out Test Predictions — Dollar Price",
       "Observed Listing Price ($)", "Predicted Listing Price ($)",
       "01_actual_vs_predicted_price.png")

fig, ax = plt.subplots(figsize=(12, 9))
ax.scatter(pred_log, residual, alpha=.65)
ax.axhline(0, linestyle="--", linewidth=2)
finish(fig, ax, "Held-Out Residuals vs Predicted logPrice",
       "Predicted logPrice", "Residual", "02_residuals_vs_predicted.png")

fig, ax = plt.subplots(figsize=(12, 8))
ax.hist(residual, bins=30, edgecolor="black")
ax.axvline(0, linestyle="--", linewidth=2)
finish(fig, ax, "Held-Out Residual Distribution",
       "Residual on logPrice", "Frequency", "03_residual_distribution.png")

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(actual_price, abs_error, alpha=.65)
finish(fig, ax, "Absolute Prediction Error by Observed Listing Price",
       "Observed Listing Price ($)", "Absolute Error ($)",
       "04_absolute_error_vs_price.png")

fig, ax = plt.subplots(figsize=(12, 8))
ax.bar(segment["price_segment"].astype(str), segment["mean_absolute_error"])
ax.tick_params(axis="x", rotation=15)
finish(fig, ax, "Mean Absolute Error by Price Segment",
       "Price Segment", "Mean Absolute Error ($)",
       "05_mae_by_price_segment.png")

fig, ax = plt.subplots(figsize=(12, 8))
ax.bar(segment["price_segment"].astype(str), segment["mean_absolute_percentage_error"])
ax.tick_params(axis="x", rotation=15)
finish(fig, ax, "Mean Absolute Percentage Error by Price Segment",
       "Price Segment", "Mean Absolute Percentage Error (%)",
       "06_mape_by_price_segment.png")

fig, ax = plt.subplots(figsize=(12, 9))
ax.scatter(pred_price, actual_price, alpha=.65)
lo, hi = min(pred_price.min(), actual_price.min()), max(pred_price.max(), actual_price.max())
ax.plot([lo, hi], [lo, hi], "--", linewidth=2, label="Perfect calibration")
x = np.linspace(lo, hi, 100)
ax.plot(x, intercept + slope*x, linewidth=2, label="Calibration line")
ax.legend()
finish(fig, ax, "Model Calibration — Held-Out Test Set",
       "Predicted Listing Price ($)", "Observed Listing Price ($)",
       "07_model_calibration.png")

def horizontal_bars(data, value_col, title, xlabel, filename):
    s = data.set_index("feature")[value_col].sort_values()
    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(len(s))
    ax.barh(y, s.values)
    ax.set_yticks(y)
    ax.set_yticklabels(s.index)
    pad = max(abs(s.values).max() * .04, .01)
    for i, v in enumerate(s.values):
        ax.text(v + pad if v >= 0 else v - pad, i, f"{v:.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=11)
    finish(fig, ax, title, xlabel, "", filename)

horizontal_bars(fi, "importance", "Feature Importance — Tuned Gradient Boosting",
                "Importance", "08_feature_importance.png")
horizontal_bars(pi, "mean_decrease_in_r2", "Permutation Importance — Tuned Gradient Boosting",
                "Mean Decrease in R²", "09_permutation_importance.png")

methodology = f"""# Part 10 — Model Evaluation & Diagnostics

Part 10 formally evaluates the Tuned Gradient Boosting development candidate from Part 9.

**Source:** `data/processed/San_Antonio_Zillow_Feature_Engineered.csv`

**Population:** 810 analytical properties; 731 complete ML cases; 584 training records; 147 holdout records.

**Target:** `logPrice`

**Predictors:** `area`, `beds`, `baths`, `lotAreaSqFt`, `taxAssessedValue`, `daysOnZillow`

Target-derived/leakage-prone fields were excluded: `price`, `pricePerSqFt`, `zestimate`, `zestimate_gap`, `zestimate_gap_pct`, `deal_score`, and `opportunity_status`.

The holdout split reproduces Part 9 using random state {RANDOM_STATE}. Evaluation includes log-price and dollar-scale metrics, calibration, residual diagnostics, price-segment error analysis, largest prediction errors, feature importance, and permutation importance.

Price segments use training-population quartiles to avoid deriving evaluation categories from the holdout set.

Residual normality is reported descriptively; normal residuals are not required for a tree-based predictive model. Statistical diagnostics do not establish causation.

The feature-engineered dataset was not modified.
"""
(OUTPUT_DIR / "MODEL_EVALUATION_METHODOLOGY.md").write_text(methodology, encoding="utf-8")

top = fi.iloc[0]
findings = f"""# Part 10 — Model Evaluation Key Findings

## Holdout Performance

- R²: **{test_r2:.4f}**
- RMSE on logPrice: **{test_rmse_log:.4f}**
- MAE on logPrice: **{test_mae_log:.4f}**
- Dollar RMSE: **${dollar_rmse:,.2f}**
- Dollar MAE: **${dollar_mae:,.2f}**
- Median absolute dollar error: **${median_ae:,.2f}**
- Dollar MAPE: **{dollar_mape:.2f}%**
- Median APE: **{median_ape:.2f}%**

## Calibration

- Slope: **{slope:.4f}**
- Intercept: **${intercept:,.2f}**
- Calibration R²: **{r*r:.4f}**

## Residual Diagnostics

- Mean residual: **{residual.mean():.4f}**
- Residual standard deviation: **{residual.std(ddof=1):.4f}**
- Pearson residual/predicted p-value: **{pearson_p:.6g}**
- Spearman absolute-residual/predicted p-value: **{spearman_p:.6g}**
- Shapiro-Wilk p-value: **{shapiro_p:.6g}**

## Feature Importance

The strongest built-in model feature was **{top["feature"]}** with importance **{top["importance"]:.3f}**.

Feature importance indicates predictive contribution within this fitted model; it does not establish causal influence.

## Price-Segment Performance

The price-segment output provides mean absolute and percentage errors across four holdout price ranges. Dollar error and percentage error should be considered together because higher-priced properties can naturally have larger dollar errors.

## Largest Errors

The `largest_prediction_errors.csv` file contains the 20 holdout observations with the largest absolute prediction errors for investigation.

## Overall Assessment

The selected model demonstrates meaningful predictive performance on unseen holdout properties. The evaluation also identifies residual behavior, calibration, price ranges, and observations where predictions are less accurate.

The model should be presented as a portfolio predictive-analysis model, not as a guaranteed property valuation system.
"""
(OUTPUT_DIR / "MODEL_EVALUATION_KEY_FINDINGS.md").write_text(findings, encoding="utf-8")

expected = [
    "holdout_predictions.csv", "largest_prediction_errors.csv",
    "price_segment_error_analysis.csv", "selected_model_feature_importance.csv",
    "selected_model_permutation_importance.csv", "model_evaluation_metrics.csv",
    "MODEL_EVALUATION_METHODOLOGY.md", "MODEL_EVALUATION_KEY_FINDINGS.md"
]
charts = [f"{i:02d}_{name}.png" for i, name in enumerate([
    "actual_vs_predicted_price", "residuals_vs_predicted", "residual_distribution",
    "absolute_error_vs_price", "mae_by_price_segment", "mape_by_price_segment",
    "model_calibration", "feature_importance", "permutation_importance"
], 1)]

missing_outputs = [str(OUTPUT_DIR / f) for f in expected] + [
    str(CHART_DIR / f) for f in charts
]
missing_outputs = [p for p in missing_outputs if not Path(p).exists()]
if missing_outputs:
    raise RuntimeError("Missing expected outputs:\n" + "\n".join(missing_outputs))

section("PART 10 MODEL EVALUATION COMPLETE")
print(f"Full analytical population: {len(df):,}")
print(f"Complete ML population: {len(ml):,}")
print(f"Training records: {len(train):,}")
print(f"Holdout test records: {len(test):,}")
print("Selected model: Tuned Gradient Boosting")
print(f"Holdout R²: {test_r2:.4f}")
print(f"Holdout RMSE (logPrice): {test_rmse_log:.4f}")
print(f"Holdout MAE (logPrice): {test_mae_log:.4f}")
print(f"Dollar RMSE: ${dollar_rmse:,.2f}")
print(f"Dollar MAE: ${dollar_mae:,.2f}")
print(f"Dollar MAPE: {dollar_mape:.2f}%")
print(f"\nGenerated charts: {len(charts)}")
print("\nAll expected Part 10 model-evaluation outputs validated.")
print("The feature-engineered dataset was not modified.")
print(f"\nOutputs saved to: {OUTPUT_DIR}")
