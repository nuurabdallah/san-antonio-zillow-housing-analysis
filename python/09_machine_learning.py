"""
Part 9 — Machine Learning
San Antonio Zillow Housing Analysis

Purpose:
    Develop and compare machine-learning models for predicting log-transformed
    Zillow listing prices using the finalized feature-engineered dataset.

Input:
    data/processed/San_Antonio_Zillow_Feature_Engineered.csv

Target:
    logPrice

Predictors:
    area
    beds
    baths
    lotAreaSqFt
    taxAssessedValue
    daysOnZillow

Important modeling rules:
    - The finalized analytical dataset is never modified.
    - Only complete cases for the six approved predictors and logPrice are
      used for machine-learning development.
    - Target-derived variables such as pricePerSqFt, zestimate_gap,
      zestimate_gap_pct, and deal_score are excluded to prevent target leakage.
    - The final analytical population remains 810 properties; the ML modeling
      population is the complete-case subset.
    - Models predict logPrice rather than raw price because listing price is
      strongly right-skewed.
    - Part 9 focuses on model development and comparison.
      Final model evaluation and interpretation belong to Part 10.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "San_Antonio_Zillow_Feature_Engineered.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "machine_learning"
CHART_DIR = OUTPUT_DIR / "charts"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 10


# ============================================================================
# MODELING VARIABLES
# ============================================================================

TARGET = "logPrice"

PREDICTORS = [
    "area",
    "beds",
    "baths",
    "lotAreaSqFt",
    "taxAssessedValue",
    "daysOnZillow",
]

LEAKAGE_COLUMNS = [
    "price",
    "pricePerSqFt",
    "zestimate",
    "zestimate_gap",
    "zestimate_gap_pct",
    "deal_score",
    "opportunity_status",
]


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

MODELS = {
    "Linear Regression": Pipeline(
        [
            ("model", LinearRegression()),
        ]
    ),
    "Ridge": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    ),
    "Lasso": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.001, max_iter=10000)),
        ]
    ),
    "Elastic Net": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNet(
                    alpha=0.001,
                    l1_ratio=0.5,
                    max_iter=10000,
                ),
            ),
        ]
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=1.0,
        random_state=RANDOM_STATE,
    ),
    "Tuned Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=1,
        subsample=1.0,
        random_state=RANDOM_STATE,
    ),
}


# ============================================================================
# HELPERS
# ============================================================================

def save_current_figure(filename):
    """Save the current matplotlib figure and close it."""

    output_path = CHART_DIR / filename

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    return output_path


def format_metric(value, decimals=4):
    """Format a metric for report output."""

    if pd.isna(value):
        return "N/A"

    return f"{value:.{decimals}f}"


def validate_dataset(df):
    """Validate the finalized analytical dataset and required ML columns."""

    required_columns = [
        "zpid",
        TARGET,
        *PREDICTORS,
        *LEAKAGE_COLUMNS,
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required machine-learning columns: "
            + ", ".join(missing_columns)
        )

    duplicate_zpid = int(df["zpid"].duplicated().sum())

    print("\n" + "=" * 70)
    print("DATASET INTEGRITY VALIDATION")
    print("=" * 70)

    print(f"Rows: {len(df):,}")
    print(f"Unique zpid: {df['zpid'].nunique():,}")
    print(f"Duplicate zpid: {duplicate_zpid:,}")

    if len(df) != 810:
        raise ValueError(
            f"Expected 810 analytical records, found {len(df):,}."
        )

    if df["zpid"].nunique() != 810:
        raise ValueError(
            "Expected 810 unique properties."
        )

    if duplicate_zpid != 0:
        raise ValueError(
            "Duplicate zpid values detected."
        )

    print("\nDataset integrity validation passed.")


def prepare_ml_data(df):
    """Prepare the complete-case ML modeling population."""

    ml_columns = ["zpid", TARGET, *PREDICTORS]

    model_data = df[ml_columns].copy()

    for column in [TARGET, *PREDICTORS]:
        model_data[column] = pd.to_numeric(
            model_data[column],
            errors="coerce",
        )

    before = len(model_data)

    model_data = model_data.dropna(
        subset=[TARGET, *PREDICTORS]
    ).copy()

    after = len(model_data)

    print("\n" + "=" * 70)
    print("MACHINE-LEARNING POPULATION")
    print("=" * 70)

    print(f"Full analytical population: {before:,}")
    print(f"Complete ML cases: {after:,}")
    print(f"Excluded incomplete cases: {before - after:,}")
    print(
        f"ML complete-case coverage: "
        f"{after / before:.2%}"
    )

    if model_data["zpid"].duplicated().any():
        raise ValueError(
            "Duplicate zpid values detected in ML population."
        )

    if after == 0:
        raise ValueError(
            "No complete machine-learning cases remain."
        )

    return model_data


def validate_no_target_leakage(df):
    """Confirm that target-derived variables are not model predictors."""

    print("\n" + "=" * 70)
    print("TARGET-LEAKAGE VALIDATION")
    print("=" * 70)

    print(f"Target: {TARGET}")
    print(f"Predictors: {', '.join(PREDICTORS)}")

    overlapping = set(PREDICTORS).intersection(
        set(LEAKAGE_COLUMNS)
    )

    if overlapping:
        raise ValueError(
            "Target leakage detected in predictor list: "
            + ", ".join(sorted(overlapping))
        )

    print("\nExcluded target-derived / leakage-prone variables:")
    for column in LEAKAGE_COLUMNS:
        print(f"  - {column}")

    print("\nNo target leakage detected in the approved predictor set.")


def create_train_test_split(model_data):
    """Create a reproducible holdout split for model development."""

    X = model_data[PREDICTORS].copy()
    y = model_data[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    print(f"Total ML records: {len(model_data):,}")
    print(f"Training records: {len(X_train):,}")
    print(f"Test records: {len(X_test):,}")
    print(f"Test proportion: {TEST_SIZE:.0%}")
    print(f"Random state: {RANDOM_STATE}")

    return X_train, X_test, y_train, y_test


def cv_scoring(model, X, y):
    """Run shuffled K-fold cross-validation."""

    cv = KFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )

    return results


def evaluate_model_for_development(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """Fit one candidate model and calculate development metrics."""

    fitted_model = clone(model)

    cv_results = cv_scoring(
        fitted_model,
        X_train,
        y_train,
    )

    fitted_model.fit(X_train, y_train)

    train_predictions = fitted_model.predict(X_train)
    test_predictions = fitted_model.predict(X_test)

    result = {
        "model": name,
        "cv_mean_r2": np.mean(cv_results["test_r2"]),
        "cv_std_r2": np.std(cv_results["test_r2"], ddof=1),
        "cv_mean_rmse_log": -np.mean(cv_results["test_rmse"]),
        "cv_mean_mae_log": -np.mean(cv_results["test_mae"]),
        "cv_mean_train_r2": np.mean(cv_results["train_r2"]),
        "cv_generalization_gap": (
            np.mean(cv_results["train_r2"])
            - np.mean(cv_results["test_r2"])
        ),
        "train_r2": r2_score(
            y_train,
            train_predictions,
        ),
        "test_r2": r2_score(
            y_test,
            test_predictions,
        ),
        "test_rmse_log": np.sqrt(
            mean_squared_error(
                y_test,
                test_predictions,
            )
        ),
        "test_mae_log": mean_absolute_error(
            y_test,
            test_predictions,
        ),
    }

    return fitted_model, result, test_predictions


def get_linear_coefficients(model):
    """Extract coefficients from a linear model pipeline."""

    if not hasattr(model, "named_steps"):
        return None

    underlying_model = model.named_steps.get("model")

    if underlying_model is None:
        return None

    if not hasattr(underlying_model, "coef_"):
        return None

    return underlying_model.coef_


def get_tree_feature_importance(model):
    """Extract native feature importance from tree-based models."""

    if not hasattr(model, "feature_importances_"):
        return None

    return model.feature_importances_


# ============================================================================
# DEVELOPMENT CHARTS
# ============================================================================

def chart_model_cv_comparison(results_df):
    """Visualize mean cross-validated R-squared by model."""

    data = results_df.sort_values(
        "cv_mean_r2",
        ascending=True,
    ).copy()

    plt.figure(figsize=(10, 7))

    plt.barh(
        data["model"],
        data["cv_mean_r2"],
    )

    plt.xlabel("Mean Cross-Validated R²")
    plt.ylabel("Model")
    plt.title("Machine-Learning Development: Cross-Validated R²")

    for index, value in enumerate(data["cv_mean_r2"]):
        plt.text(
            value,
            index,
            f" {value:.3f}",
            va="center",
        )

    return save_current_figure(
        "01_model_cross_validated_r2.png"
    )


def chart_model_rmse_comparison(results_df):
    """Visualize mean cross-validated RMSE."""

    data = results_df.sort_values(
        "cv_mean_rmse_log",
        ascending=False,
    ).copy()

    plt.figure(figsize=(10, 7))

    plt.barh(
        data["model"],
        data["cv_mean_rmse_log"],
    )

    plt.xlabel("Mean Cross-Validated RMSE (logPrice)")
    plt.ylabel("Model")
    plt.title("Machine-Learning Development: Cross-Validated RMSE")

    for index, value in enumerate(data["cv_mean_rmse_log"]):
        plt.text(
            value,
            index,
            f" {value:.3f}",
            va="center",
        )

    return save_current_figure(
        "02_model_cross_validated_rmse.png"
    )


def chart_model_generalization_gap(results_df):
    """Visualize training-to-validation R² generalization gap."""

    data = results_df.sort_values(
        "cv_generalization_gap",
        ascending=True,
    ).copy()

    plt.figure(figsize=(10, 7))

    plt.barh(
        data["model"],
        data["cv_generalization_gap"],
    )

    plt.xlabel("Training R² − Validation R²")
    plt.ylabel("Model")
    plt.title("Model Generalization Gap")

    for index, value in enumerate(data["cv_generalization_gap"]):
        plt.text(
            value,
            index,
            f" {value:.3f}",
            va="center",
        )

    return save_current_figure(
        "03_model_generalization_gap.png"
    )


def chart_test_actual_vs_predicted(
    y_test,
    predictions,
    selected_model_name,
):
    """Plot held-out test predictions against observed log prices."""

    plt.figure(figsize=(9, 7))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.55,
        s=24,
    )

    minimum = min(
        y_test.min(),
        predictions.min(),
    )

    maximum = max(
        y_test.max(),
        predictions.max(),
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=2,
        label="Perfect prediction",
    )

    plt.xlabel("Observed logPrice")
    plt.ylabel("Predicted logPrice")
    plt.title(
        f"Held-Out Test Predictions — {selected_model_name}"
    )
    plt.legend()

    return save_current_figure(
        "04_selected_model_test_predictions.png"
    )


def chart_selected_model_residuals(
    y_test,
    predictions,
    selected_model_name,
):
    """Plot held-out residuals against predicted log price."""

    residuals = y_test - predictions

    plt.figure(figsize=(10, 6))

    plt.scatter(
        predictions,
        residuals,
        alpha=0.55,
        s=24,
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=2,
    )

    plt.xlabel("Predicted logPrice")
    plt.ylabel("Residual")
    plt.title(
        f"Held-Out Residuals — {selected_model_name}"
    )

    return save_current_figure(
        "05_selected_model_residuals.png"
    )


def chart_feature_importance(
    selected_model,
    selected_model_name,
):
    """Create feature-importance visualization when supported."""

    importance = get_tree_feature_importance(
        selected_model
    )

    if importance is None:
        coefficients = get_linear_coefficients(
            selected_model
        )

        if coefficients is None:
            return None

        importance = np.abs(coefficients)

    data = pd.DataFrame(
        {
            "feature": PREDICTORS,
            "importance": importance,
        }
    ).sort_values(
        "importance",
        ascending=True,
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        data["feature"],
        data["importance"],
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(
        f"Feature Importance — {selected_model_name}"
    )

    for index, value in enumerate(data["importance"]):
        plt.text(
            value,
            index,
            f" {value:.3f}",
            va="center",
        )

    return save_current_figure(
        "06_selected_model_feature_importance.png"
    )


def chart_permutation_importance(
    selected_model,
    X_test,
    y_test,
    selected_model_name,
):
    """Calculate and plot permutation importance on the holdout set."""

    permutation = permutation_importance(
        selected_model,
        X_test,
        y_test,
        scoring="r2",
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    data = pd.DataFrame(
        {
            "feature": PREDICTORS,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values(
        "importance_mean",
        ascending=True,
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        data["feature"],
        data["importance_mean"],
        xerr=data["importance_std"],
        capsize=3,
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("Mean Decrease in R²")
    plt.ylabel("Feature")
    plt.title(
        f"Permutation Importance — {selected_model_name}"
    )

    return save_current_figure(
        "07_selected_model_permutation_importance.png"
    ), data


# ============================================================================
# REPORTS
# ============================================================================

def write_model_development_report(
    model_data,
    results_df,
    selected_model_name,
):
    """Write model-development methodology and decisions."""

    output_path = OUTPUT_DIR / "MACHINE_LEARNING_METHODOLOGY.md"

    selected_row = results_df.loc[
        results_df["model"] == selected_model_name
    ].iloc[0]

    text = f"""# Part 9 — Machine Learning Methodology

## Purpose

Part 9 develops and compares machine-learning models for predicting
Zillow listing prices in San Antonio.

The modeling target is `logPrice`, the natural-log transformation of
listing price.

## Source Dataset

- Source: `data/processed/San_Antonio_Zillow_Feature_Engineered.csv`
- Full analytical population: 810 properties
- Complete machine-learning population: {len(model_data):,} properties
- Target: `{TARGET}`

## Approved Predictors

The model-development feature set is:

- `area`
- `beds`
- `baths`
- `lotAreaSqFt`
- `taxAssessedValue`
- `daysOnZillow`

## Target Leakage Controls

The following variables are intentionally excluded from the predictor set:

- `price`
- `pricePerSqFt`
- `zestimate`
- `zestimate_gap`
- `zestimate_gap_pct`
- `deal_score`
- `opportunity_status`

These variables either directly contain the target, are calculated from
the target, or are business metrics derived from listing price and therefore
could create target leakage.

## Missing Data

Machine-learning development uses complete cases for the target and all six
approved predictors. The complete-case subset is used only for model
training and comparison.

The original analytical dataset is not modified, and missing values remain
missing in the source dataset.

## Train/Test Design

- Holdout test proportion: {TEST_SIZE:.0%}
- Random state: {RANDOM_STATE}
- Cross-validation: {CV_FOLDS}-fold shuffled K-fold
- Cross-validation random state: {RANDOM_STATE}

Cross-validation is performed on the training portion of the data. The
holdout test set is reserved for model-development checking and subsequent
evaluation.

## Candidate Models

The following models were developed:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Elastic Net
5. Random Forest Regressor
6. Gradient Boosting Regressor
7. Tuned Gradient Boosting Regressor

Linear models use standardization where appropriate. Tree-based models do
not require feature scaling.

## Development Metrics

Models are compared using:

- Cross-validated R²
- Cross-validated RMSE on logPrice
- Cross-validated MAE on logPrice
- Cross-validated training R²
- Generalization gap
- Holdout-test R²
- Holdout-test RMSE
- Holdout-test MAE

## Development Candidate

The development candidate selected for subsequent model evaluation is:

**{selected_model_name}**

Its cross-validated development results were:

- Mean CV R²: {selected_row["cv_mean_r2"]:.4f}
- CV R² standard deviation: {selected_row["cv_std_r2"]:.4f}
- Mean CV RMSE: {selected_row["cv_mean_rmse_log"]:.4f}
- Mean CV MAE: {selected_row["cv_mean_mae_log"]:.4f}
- Mean training R²: {selected_row["cv_mean_train_r2"]:.4f}
- Generalization gap: {selected_row["cv_generalization_gap"]:.4f}

Selection in Part 9 is a development decision. Final model evaluation,
residual diagnostics, error analysis, calibration, and business interpretation
are addressed in Part 10.

## Reproducibility

The random seed is fixed at `{RANDOM_STATE}`. The complete model-development
process is implemented in `python/09_machine_learning.py`.

The script does not modify the analytical dataset.
"""

    output_path.write_text(
        text,
        encoding="utf-8",
    )

    return output_path


def write_model_comparison_csv(results_df):
    """Save the model comparison table."""

    output_path = OUTPUT_DIR / "model_development_comparison.csv"

    results_df.to_csv(
        output_path,
        index=False,
    )

    return output_path


def write_feature_importance_csv(
    selected_model,
    selected_model_name,
):
    """Save native feature importance when available."""

    importance = get_tree_feature_importance(
        selected_model
    )

    source = "native_tree_importance"

    if importance is None:
        coefficients = get_linear_coefficients(
            selected_model
        )

        if coefficients is None:
            return None

        importance = np.abs(coefficients)
        source = "absolute_linear_coefficient"

    data = pd.DataFrame(
        {
            "model": selected_model_name,
            "feature": PREDICTORS,
            "importance": importance,
            "importance_source": source,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    output_path = OUTPUT_DIR / "selected_model_feature_importance.csv"

    data.to_csv(
        output_path,
        index=False,
    )

    return output_path


def write_permutation_importance_csv(
    permutation_data,
    selected_model_name,
):
    """Save permutation importance results."""

    data = permutation_data.copy()

    data.insert(
        0,
        "model",
        selected_model_name,
    )

    output_path = (
        OUTPUT_DIR
        / "selected_model_permutation_importance.csv"
    )

    data.to_csv(
        output_path,
        index=False,
    )

    return output_path


def write_key_findings(
    model_data,
    results_df,
    selected_model_name,
):
    """Write concise machine-learning development findings."""

    selected = results_df.loc[
        results_df["model"] == selected_model_name
    ].iloc[0]

    best_cv_r2 = results_df.loc[
        results_df["cv_mean_r2"].idxmax()
    ]

    lowest_cv_rmse = results_df.loc[
        results_df["cv_mean_rmse_log"].idxmin()
    ]

    lowest_gap = results_df.loc[
        results_df["cv_generalization_gap"].idxmin()
    ]

    text = f"""# Part 9 — Machine Learning Key Findings

## Modeling Population

- Full analytical population: 810 properties.
- Complete machine-learning cases: {len(model_data):,}.
- ML complete-case coverage: {len(model_data) / 810:.2%}.
- Target: `logPrice`.
- Predictors: `area`, `beds`, `baths`, `lotAreaSqFt`,
  `taxAssessedValue`, and `daysOnZillow`.

## Model Development

Seven candidate models were developed and compared using shuffled
{CV_FOLDS}-fold cross-validation on the training data.

### Best Cross-Validated R²

**{best_cv_r2["model"]}**

- Mean CV R²: {best_cv_r2["cv_mean_r2"]:.4f}
- CV R² standard deviation: {best_cv_r2["cv_std_r2"]:.4f}

### Lowest Cross-Validated RMSE

**{lowest_cv_rmse["model"]}**

- Mean CV RMSE: {lowest_cv_rmse["cv_mean_rmse_log"]:.4f}

### Smallest Generalization Gap

**{lowest_gap["model"]}**

- Generalization gap: {lowest_gap["cv_generalization_gap"]:.4f}

## Development Candidate

The model carried forward to Part 10 is:

**{selected_model_name}**

Development metrics:

- Mean CV R²: {selected["cv_mean_r2"]:.4f}
- CV R² standard deviation: {selected["cv_std_r2"]:.4f}
- Mean CV RMSE: {selected["cv_mean_rmse_log"]:.4f}
- Mean CV MAE: {selected["cv_mean_mae_log"]:.4f}
- Mean training R²: {selected["cv_mean_train_r2"]:.4f}
- Generalization gap: {selected["cv_generalization_gap"]:.4f}
- Holdout-test R²: {selected["test_r2"]:.4f}
- Holdout-test RMSE: {selected["test_rmse_log"]:.4f}
- Holdout-test MAE: {selected["test_mae_log"]:.4f}

## Interpretation

The development results show how well the candidate models capture
variation in log-transformed listing price using the approved structural
and property-level predictors.

The model selected for Part 10 is a development candidate rather than a
final business-decision model. Final evaluation should consider residual
behavior, error magnitude, generalization, calibration, feature importance,
and other diagnostics.

## Target Leakage

No approved predictor is directly derived from listing price or another
target-derived business metric.

Variables such as `pricePerSqFt`, `zestimate_gap`, `zestimate_gap_pct`,
and `deal_score` were intentionally excluded from the predictor set.
"""

    output_path = OUTPUT_DIR / "MACHINE_LEARNING_KEY_FINDINGS.md"

    output_path.write_text(
        text,
        encoding="utf-8",
    )

    return output_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("PART 9 — MACHINE LEARNING")
    print("=" * 70)

    print("\nSAN ANTONIO ZILLOW MACHINE LEARNING")

    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Input file: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHART_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    old_charts = list(CHART_DIR.glob("*.png"))

    for chart in old_charts:
        chart.unlink()

    if old_charts:
        print(
            f"\nCleared {len(old_charts):,} existing PNG chart(s)."
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"\nRows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns):,}")

    validate_dataset(df)

    validate_no_target_leakage(df)

    model_data = prepare_ml_data(df)

    X_train, X_test, y_train, y_test = create_train_test_split(
        model_data
    )

    print("\n" + "=" * 70)
    print("MODEL DEVELOPMENT")
    print("=" * 70)

    print(
        f"\nTraining {len(MODELS):,} candidate models "
        f"using {CV_FOLDS}-fold cross-validation..."
    )

    fitted_models = {}
    results = {}
    test_predictions = {}

    for name, model in MODELS.items():
        print(f"\nRunning: {name}")

        (
            fitted_model,
            result,
            predictions,
        ) = evaluate_model_for_development(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        fitted_models[name] = fitted_model
        results[name] = result
        test_predictions[name] = predictions

        print(
            f"  CV R²: {result['cv_mean_r2']:.4f}"
        )
        print(
            f"  CV RMSE: {result['cv_mean_rmse_log']:.4f}"
        )
        print(
            f"  CV MAE: {result['cv_mean_mae_log']:.4f}"
        )

    results_df = pd.DataFrame(
        list(results.values())
    ).sort_values(
        "cv_mean_r2",
        ascending=False,
    ).reset_index(drop=True)

    # The tuned Gradient Boosting model is the designated development
    # candidate because it was previously tuned using the approved predictor
    # set and is carried forward for formal evaluation in Part 10.
    selected_model_name = "Tuned Gradient Boosting"

    if selected_model_name not in fitted_models:
        raise ValueError(
            "Designated development candidate was not fitted."
        )

    selected_model = fitted_models[selected_model_name]
    selected_predictions = test_predictions[selected_model_name]

    print("\n" + "=" * 70)
    print("MODEL DEVELOPMENT COMPARISON")
    print("=" * 70)

    display_columns = [
        "model",
        "cv_mean_r2",
        "cv_std_r2",
        "cv_mean_rmse_log",
        "cv_mean_mae_log",
        "cv_generalization_gap",
        "test_r2",
        "test_rmse_log",
        "test_mae_log",
    ]

    print(
        results_df[display_columns].to_string(
            index=False
        )
    )

    print(
        f"\nDevelopment candidate: {selected_model_name}"
    )

    print("\nCreating development charts...")

    chart_model_cv_comparison(results_df)
    chart_model_rmse_comparison(results_df)
    chart_model_generalization_gap(results_df)

    chart_test_actual_vs_predicted(
        y_test,
        selected_predictions,
        selected_model_name,
    )

    chart_selected_model_residuals(
        y_test,
        selected_predictions,
        selected_model_name,
    )

    chart_feature_importance(
        selected_model,
        selected_model_name,
    )

    (
        permutation_chart_path,
        permutation_data,
    ) = chart_permutation_importance(
        selected_model,
        X_test,
        y_test,
        selected_model_name,
    )

    model_comparison_path = write_model_comparison_csv(
        results_df
    )

    feature_importance_path = write_feature_importance_csv(
        selected_model,
        selected_model_name,
    )

    permutation_importance_path = write_permutation_importance_csv(
        permutation_data,
        selected_model_name,
    )

    methodology_path = write_model_development_report(
        model_data,
        results_df,
        selected_model_name,
    )

    findings_path = write_key_findings(
        model_data,
        results_df,
        selected_model_name,
    )

    # Save the holdout predictions for Part 10 evaluation.
    prediction_data = pd.DataFrame(
        {
            "zpid": model_data.loc[
                X_test.index,
                "zpid",
            ],
            "observed_logPrice": y_test,
            "predicted_logPrice": selected_predictions,
            "residual_logPrice": (
                y_test - selected_predictions
            ),
        }
    ).sort_values("zpid")

    prediction_path = (
        OUTPUT_DIR
        / "selected_model_holdout_predictions.csv"
    )

    prediction_data.to_csv(
        prediction_path,
        index=False,
    )

    # Save the training/test membership for reproducibility.
    split_data = pd.DataFrame(
        {
            "zpid": model_data["zpid"],
            "dataset_split": np.where(
                model_data.index.isin(X_train.index),
                "train",
                "test",
            ),
        }
    )

    split_path = (
        OUTPUT_DIR
        / "ml_train_test_split.csv"
    )

    split_data.to_csv(
        split_path,
        index=False,
    )

    # Final output validation.
    chart_files = sorted(
        CHART_DIR.glob("*.png")
    )

    expected_chart_files = {
        "01_model_cross_validated_r2.png",
        "02_model_cross_validated_rmse.png",
        "03_model_generalization_gap.png",
        "04_selected_model_test_predictions.png",
        "05_selected_model_residuals.png",
        "06_selected_model_feature_importance.png",
        "07_selected_model_permutation_importance.png",
    }

    actual_chart_files = {
        file.name for file in chart_files
    }

    if actual_chart_files != expected_chart_files:
        missing = expected_chart_files - actual_chart_files
        unexpected = actual_chart_files - expected_chart_files

        raise ValueError(
            "Machine-learning chart validation failed. "
            f"Missing: {sorted(missing)}; "
            f"Unexpected: {sorted(unexpected)}"
        )

    expected_report_files = [
        model_comparison_path,
        feature_importance_path,
        permutation_importance_path,
        methodology_path,
        findings_path,
        prediction_path,
        split_path,
    ]

    missing_reports = [
        path for path in expected_report_files
        if path is None or not path.exists()
    ]

    if missing_reports:
        raise ValueError(
            "One or more machine-learning output files were not created."
        )

    if len(results_df) != len(MODELS):
        raise ValueError(
            "Expected one development result per candidate model."
        )

    if len(model_data) <= 0:
        raise ValueError(
            "ML modeling population is empty."
        )

    if (
        len(prediction_data) != len(X_test)
        or len(split_data) != len(model_data)
    ):
        raise ValueError(
            "Holdout prediction or train/test split output is incomplete."
        )

    print("\n" + "=" * 70)
    print("PART 9 MACHINE LEARNING COMPLETE")
    print("=" * 70)

    print(
        f"\nFull analytical population: {len(df):,}"
    )
    print(
        f"Complete ML population: {len(model_data):,}"
    )
    print(
        f"Training records: {len(X_train):,}"
    )
    print(
        f"Holdout test records: {len(X_test):,}"
    )
    print(
        f"Candidate models developed: {len(MODELS):,}"
    )
    print(
        f"Development candidate: {selected_model_name}"
    )
    print(
        f"Mean CV R²: "
        f"{results_df.loc[results_df['model'] == selected_model_name, 'cv_mean_r2'].iloc[0]:.4f}"
    )
    print(
        f"Mean CV RMSE: "
        f"{results_df.loc[results_df['model'] == selected_model_name, 'cv_mean_rmse_log'].iloc[0]:.4f}"
    )
    print(
        f"Mean CV MAE: "
        f"{results_df.loc[results_df['model'] == selected_model_name, 'cv_mean_mae_log'].iloc[0]:.4f}"
    )

    print(
        f"\nGenerated charts: {len(chart_files):,}"
    )

    print(
        "\nAll expected Part 9 machine-learning outputs validated."
    )

    print(
        "The feature-engineered dataset was not modified."
    )

    print(
        "\nFinal model evaluation and diagnostics continue in Part 10."
    )


if __name__ == "__main__":
    main()
