"""Held-out, non-ensembled comparison of experimental and observational controls.

Run from the workspace root with ``python -m src.placebo_validation``.
The source contrast is experimental controls minus observational controls,
not an effect of actual treatment. No existing treatment caches are read.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from flaml import AutoML
from lightgbm import Booster, LGBMClassifier, LGBMRegressor
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import brier_score_loss, log_loss, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .data_loading import (
    COVARIATES,
    OUTCOME_BINARY,
    OUTCOME_CONTINUOUS,
    TREATMENT,
    difference_in_means,
)
from .estimators import (
    doubly_robust_ate,
    ipw_ate,
    outcome_modeling_ate,
    regression_ate,
)
from .trimming import optimal_trim_threshold


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
OUTCOMES = {"continuous": OUTCOME_CONTINUOUS, "binary": OUTCOME_BINARY}
SOURCE = "source"
SOURCE_ROW = "source_row"
PLACEBO = "placebo_treated"
PROPENSITY = "placebo_propensity"
TREATMENT_PROPENSITY = "treatment_propensity"
FOLD = "outer_fold"
KEEP = "keep_crump"
PROVENANCE = [SOURCE, SOURCE_ROW, TREATMENT, PLACEBO]
SOURCE_FILES = [
    "placebo_validation.py", "data_loading.py", "estimators.py", "trimming.py", "__init__.py",
]
Z_975 = float(norm.ppf(0.975))


@dataclass(frozen=True)
class Config:
    folds: int = 3
    inner_folds: int = 3
    time_budget: float = 30.0
    max_iter: int = 30
    seed: int = 0
    jobs: int = 4
    continuous_upper: float | None = None

    def validate(self) -> None:
        _require(self.folds >= 2 and self.inner_folds >= 2, "CV needs at least two folds.")
        _require(np.isfinite(self.time_budget) and self.time_budget > 0, "Invalid time budget.")
        _require(self.max_iter >= 2, "At least two FLAML iterations are required for tuning.")
        _require(self.jobs >= 1, "jobs must be positive.")
        _require(0 <= self.seed < 2**32 - 10000, "seed is outside the supported range.")
        if self.continuous_upper is not None:
            _require(
                np.isfinite(self.continuous_upper) and self.continuous_upper >= 0,
                "The common continuous-outcome upper bound must be finite and nonnegative.",
            )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _vector(values: np.ndarray, name: str, n: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    _require(result.ndim == 1 and result.size > 0, f"{name} must be a nonempty vector.")
    _require(n is None or len(result) == n, f"{name} has the wrong length.")
    _require(bool(np.isfinite(result).all()), f"{name} contains missing/nonfinite values.")
    return result


def _groups(values: np.ndarray) -> np.ndarray:
    result = _vector(values, PLACEBO)
    _require(bool(np.isin(result, [0, 1]).all()), "Placebo labels must be 0 or 1.")
    _require(
        bool((np.bincount(result.astype(int), minlength=2) >= 2).all()),
        "Both placebo groups need at least two observations.",
    )
    return result.astype(np.int8)


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _index_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def save_source_snapshot(output: Path) -> dict[str, str]:
    snapshot = output / "source_snapshot" / "src"
    snapshot.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in SOURCE_FILES:
        source = ROOT / "src" / name
        expected = _file_hash(source)
        shutil.copy2(source, snapshot / name)
        _require(_file_hash(snapshot / name) == expected, f"{name}: source changed during archiving.")
        hashes[name] = expected
    return hashes


def build_placebo_sample(
    experimental: pd.DataFrame, observed: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter actual exposure first; retain positional provenance in each input."""
    required = [TREATMENT, *OUTCOMES.values(), *COVARIATES]
    _require(len(COVARIATES) == 41 and len(set(COVARIATES)) == 41, "Expected 41 unique covariates.")
    _require(
        not set(COVARIATES).intersection([*PROVENANCE, *OUTCOMES.values(), FOLD, PROPENSITY, KEEP]),
        "An exposure, outcome, or analysis indicator is in the adjustment covariates.",
    )
    pieces, audit = [], []
    for source, frame, label in [("experimental", experimental, 1), ("observed", observed, 0)]:
        _require(frame.columns.is_unique, f"{source}: duplicate column names.")
        _require(set(frame.columns) == set(required), f"{source}: expected the 44-column public schema.")
        exposure = frame[TREATMENT]
        _require(
            pd.api.types.is_bool_dtype(exposure.dtype) and not exposure.isna().any(),
            f"{source}: {TREATMENT} must be nonmissing Boolean actual exposure.",
        )
        positions = np.flatnonzero(exposure.eq(False).to_numpy())
        controls = frame.iloc[positions][required].copy()
        _require(len(controls) >= 2, f"{source}: fewer than two genuinely unexposed rows.")
        for column in [*COVARIATES, *OUTCOMES.values()]:
            _require(pd.api.types.is_numeric_dtype(controls[column]), f"{source}: {column} is not numeric.")
            _vector(controls[column].to_numpy(), f"{source}.{column}", len(controls))
        _require(
            controls[OUTCOME_BINARY].isin([0, 1]).all(),
            f"{source}: {OUTCOME_BINARY} is not binary.",
        )
        controls[SOURCE] = source
        controls[SOURCE_ROW] = positions
        controls[PLACEBO] = np.int8(label)
        pieces.append(controls)
        audit.append({
            SOURCE: source,
            PLACEBO: label,
            "original_rows": len(frame),
            "excluded_actual_exposure": int(exposure.sum()),
            "unexposed_rows": len(controls),
            "excluded_missing_or_nonfinite": 0,
        })
    sample = pd.concat(pieces, ignore_index=True)
    _require(not sample[TREATMENT].any(), "Actual exposure leaked into the placebo sample.")
    _require(
        np.array_equal(sample[PLACEBO].to_numpy() == 1, sample[SOURCE].eq("experimental").to_numpy()),
        "Placebo source labels were reversed.",
    )
    _require(not sample.duplicated([SOURCE, SOURCE_ROW]).any(), "Duplicate source-row keys.")
    _groups(sample[PLACEBO].to_numpy())
    return sample, pd.DataFrame(audit)


def apply_common_upper_bound(
    sample: pd.DataFrame, audit: pd.DataFrame, upper: float | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply one fixed eligibility rule, without re-estimating source percentiles."""
    if upper is None:
        return sample, audit
    _require(np.isfinite(upper) and upper >= 0, "Invalid common outcome upper bound.")
    _require(not sample[TREATMENT].any(), "Eligibility restriction expects actual D=0 rows only.")
    outcome = _vector(sample[OUTCOME_CONTINUOUS].to_numpy(), OUTCOME_CONTINUOUS, len(sample))
    eligible = outcome <= upper
    _groups(sample.loc[eligible, PLACEBO].to_numpy())
    selected = sample.loc[eligible].reset_index(drop=True)
    result_audit = audit.copy()
    counts = selected.groupby(SOURCE).size()
    maxima = sample.groupby(SOURCE)[OUTCOME_CONTINUOUS].max()
    result_audit["released_control_maximum"] = result_audit[SOURCE].map(maxima)
    result_audit["common_continuous_upper"] = float(upper)
    result_audit["retained_unexposed_rows"] = result_audit[SOURCE].map(counts).astype(int)
    result_audit["excluded_common_outcome_bound"] = (
        result_audit["unexposed_rows"] - result_audit["retained_unexposed_rows"]
    )
    _require(
        int(result_audit["excluded_common_outcome_bound"].sum()) == int((~eligible).sum()),
        "Common-bound exclusion counts do not reconcile.",
    )
    return selected, result_audit


def validate_folds(
    folds: list[tuple[np.ndarray, np.ndarray]], source: np.ndarray
) -> np.ndarray:
    source = _groups(source)
    n = len(source)
    coverage = np.zeros(n, dtype=int)
    assignment = np.full(n, -1, dtype=np.int16)
    for fold, (train, test) in enumerate(folds):
        for indices in [train, test]:
            _require(
                indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer),
                "Fold indices must be integer vectors.",
            )
            _require(
                len(indices) > 0 and bool(((indices >= 0) & (indices < n)).all()),
                "Empty or out-of-range fold indices.",
            )
            _require(len(np.unique(indices)) == len(indices), "Duplicate indices within a fold.")
            _groups(source[indices])
        _require(len(train) + len(test) == n, "Train and test do not cover the sample.")
        _require(np.intersect1d(train, test).size == 0, "Train/test leakage.")
        coverage[test] += 1
        assignment[test] = fold
    _require(bool((coverage == 1).all()), "Each row must receive exactly one held-out prediction.")
    return assignment


def make_folds(source: np.ndarray, config: Config) -> list[tuple[np.ndarray, np.ndarray]]:
    config.validate()
    source = _groups(source)
    _require(int(np.bincount(source).min()) >= 2 * config.folds, "Insufficient rows per source for CV.")
    splitter = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.seed)
    folds = list(splitter.split(np.zeros(len(source)), source))
    validate_folds(folds, source)
    return folds


def crump_rule(propensity: np.ndarray) -> tuple[float, np.ndarray]:
    """The overlap rule has no access to either outcome."""
    propensity = _vector(propensity, PROPENSITY)
    _require(bool(((propensity > 0) & (propensity < 1)).all()), "Propensities must lie strictly in (0, 1); no clipping is applied.")
    alpha = optimal_trim_threshold(propensity)
    _require(np.isfinite(alpha) and 0 <= alpha < 0.5, "Invalid Crump threshold.")
    keep = (propensity >= alpha) & (propensity <= 1 - alpha)
    _require(bool(keep.any()), "Crump trimming removed the entire sample.")
    return alpha, keep


def _fit_single(
    X: pd.DataFrame,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    prediction_column: str,
    fold: int,
    config: Config,
    output: Path,
) -> tuple[np.ndarray, dict]:
    """Fit one nuisance model, with an explicit origin or actual-treatment role."""
    is_propensity = prediction_column in {PROPENSITY, TREATMENT_PROPENSITY}
    task, metric = ("classification", "log_loss") if is_propensity else ("regression", "rmse")
    _require(np.intersect1d(train, test).size == 0, "Nuisance training/test leakage.")
    _require(len(train) >= config.inner_folds, "Insufficient inner-CV training rows.")
    search_seed = config.seed + 100 * fold + (
        [PROPENSITY, "mu0_continuous", "mu1_continuous", "mu0_binary", "mu1_binary"].index(
            PROPENSITY if is_propensity else prediction_column
        )
    )
    LOGGER.info("Fold %d: tuning %s on %d rows (%s).", fold, prediction_column, len(train), metric)
    started = time.perf_counter()
    automl = AutoML()
    automl.fit(
        X_train=X.iloc[train],
        y_train=np.asarray(y)[train],
        task=task,
        metric=metric,
        estimator_list=["lgbm"],
        ensemble=False,
        eval_method="cv",
        n_splits=config.inner_folds,
        time_budget=config.time_budget,
        max_iter=config.max_iter,
        n_jobs=config.jobs,
        seed=search_seed,
        early_stop=True,
        sample=False,
        auto_augment=False,
        skip_transform=True,
        retrain_full=True,
        mem_thres=1024**3,
        mlflow_logging=False,
        log_file_name=str(output / "flaml_logs" / f"{prediction_column}_fold{fold}.jsonl"),
        verbose=0,
    )
    _require(automl.best_estimator == "lgbm", "FLAML selected an unexpected learner.")
    _require(np.isfinite(automl.best_loss), "FLAML produced no finite tuning loss; increase the tuning budget.")
    _require(automl._state.metric == metric, "The executed FLAML objective differs from the requested objective.")
    estimator = automl.model.estimator
    expected_type = LGBMClassifier if is_propensity else LGBMRegressor
    _require(isinstance(estimator, expected_type), "Expected one fitted LightGBM model, not an ensemble.")
    tree = estimator.booster_.dump_model(num_iteration=1)["tree_info"][0]["tree_structure"]
    fitted_rows = int(tree.get("internal_count", tree.get("leaf_count", -1)))
    _require(fitted_rows == len(train), "Selected model was not refitted on the entire outer-training sample.")
    if is_propensity:
        _require(np.array_equal(automl.classes_, [0, 1]), "Unexpected propensity class order.")
        predictions = np.asarray(automl.predict_proba(X.iloc[test]))[:, 1]
    else:
        predictions = np.asarray(automl.predict(X.iloc[test]))
    predictions = _vector(predictions, prediction_column, len(test))
    native_predictions = estimator.booster_.predict(X.iloc[test[:64]], num_threads=config.jobs)
    np.testing.assert_allclose(predictions[:64], native_predictions, rtol=1e-12, atol=1e-12)
    model_file = Path("models") / f"{prediction_column}_fold{fold}.txt"
    estimator.booster_.save_model(str(output / model_file))
    record = {
        "prediction_column": prediction_column,
        "fold": fold,
        "task": task,
        "metric": metric,
        "best_cv_loss": float(automl.best_loss),
        "best_config": automl.best_config,
        "fitted_lgbm_params": estimator.get_params(),
        "n_train": len(train),
        "n_test": len(test),
        "final_fit_rows_verified": fitted_rows,
        "train_index_sha256": _index_hash(train),
        "test_index_sha256": _index_hash(test),
        "search_seed": search_seed,
        "model_file": str(model_file),
        "seconds": time.perf_counter() - started,
    }
    LOGGER.info("Finished %s fold %d: CV %s %.6g (%.1fs).", prediction_column, fold, metric, automl.best_loss, record["seconds"])
    return predictions, record


def cross_fit_nuisances(
    sample: pd.DataFrame, config: Config, output: Path
) -> tuple[pd.DataFrame, list[dict], float]:
    X = sample[COVARIATES]
    source = _groups(sample[PLACEBO].to_numpy())
    folds = make_folds(source, config)
    predictions = sample[[*PROVENANCE, *OUTCOMES.values()]].copy()
    predictions[FOLD] = validate_folds(folds, source)
    columns = [PROPENSITY, "mu0_continuous", "mu1_continuous", "mu0_binary", "mu1_binary"]
    for column in columns:
        predictions[column] = np.nan
    records = []
    for fold, (train, test) in enumerate(folds):
        values, record = _fit_single(X, source, train, test, PROPENSITY, fold, config, output)
        predictions.loc[test, PROPENSITY] = values
        records.append(record)
        _write_json(output / "model_manifest.json", records)
    alpha, keep = crump_rule(predictions[PROPENSITY].to_numpy())
    _groups(source[keep])
    predictions[KEEP] = keep
    LOGGER.info("New source-propensity Crump alpha=%.8g; keeping %d/%d rows.", alpha, int(keep.sum()), len(keep))
    for fold, (train, test) in enumerate(folds):
        for outcome, column in OUTCOMES.items():
            y = sample[column].to_numpy(dtype=float)
            for group in [0, 1]:
                group_train = train[source[train] == group]
                prediction_column = f"mu{group}_{outcome}"
                values, record = _fit_single(
                    X, y, group_train, test, prediction_column, fold, config, output
                )
                predictions.loc[test, prediction_column] = values
                records.append(record)
                _write_json(output / "model_manifest.json", records)
    for column in columns:
        _vector(predictions[column].to_numpy(), column, len(sample))
    _require(len(records) == 5 * config.folds, "Wrong number of single nuisance models.")
    return predictions, records, alpha


def score_arrays(
    y: np.ndarray, source: np.ndarray, propensity: np.ndarray,
    mu0: np.ndarray, mu1: np.ndarray,
) -> dict[str, np.ndarray]:
    source = _groups(source)
    n = len(source)
    y, propensity, mu0, mu1 = [
        _vector(values, name, n)
        for values, name in [(y, "outcome"), (propensity, PROPENSITY), (mu0, "mu0"), (mu1, "mu1")]
    ]
    _require(bool(((propensity > 0) & (propensity < 1)).all()), "Invalid inverse-probability weights.")
    ipw = source * y / propensity - (1 - source) * y / (1 - propensity)
    dr = mu1 - mu0 + source * (y - mu1) / propensity - (1 - source) * (y - mu0) / (1 - propensity)
    _vector(ipw, "IPW score", n)
    _vector(dr, "DR score", n)
    np.testing.assert_allclose(ipw.mean(), ipw_ate(y, source, propensity), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(dr.mean(), doubly_robust_ate(y, source, propensity, mu0, mu1), rtol=1e-10, atol=1e-12)
    return {"IPW": ipw, "DR": dr}


def inference_row(estimator: str, estimate: float, se: float, method: str) -> dict:
    _require(np.isfinite(estimate) and np.isfinite(se) and se > 0, f"{estimator}: finite estimate and positive SE required.")
    return {
        "Estimator": estimator,
        "Estimate": float(estimate),
        "SE": float(se),
        "CI_Lower": float(estimate - Z_975 * se),
        "CI_Upper": float(estimate + Z_975 * se),
        "P_Value": float(2 * norm.sf(abs(estimate / se))),
        "SE_Method": method,
        "Status": "estimated",
    }


def common_bound_diagnostics(
    released: pd.DataFrame, upper: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Describe the change in means; this is not a test of a causal effect change."""
    eligible = released[OUTCOME_CONTINUOUS].le(upper).to_numpy()
    _groups(released.loc[eligible, PLACEBO].to_numpy())
    excluded = released.loc[~eligible, [*PROVENANCE, *OUTCOMES.values()]].reset_index(drop=True)
    contrasts, decomposition = [], []
    for outcome, column in OUTCOMES.items():
        for population, mask in [
            ("as_released_controls", np.ones(len(released), dtype=bool)),
            ("common_bound_controls", eligible),
        ]:
            group = released.loc[mask, PLACEBO].to_numpy()
            y = released.loc[mask, column].to_numpy(dtype=float)
            estimate, variance = difference_in_means(pd.DataFrame({"y": y, "D": group}))
            row = inference_row("Naive", estimate, np.sqrt(variance), "Unequal-variance difference in means")
            row.update({
                "Outcome": outcome, "Population": population,
                "N_Experimental_Controls": int(group.sum()),
                "N_Observed_Controls": int((group == 0).sum()),
                "Display_Unit": "percentage points" if outcome == "binary" else "Outcome_Continuous units",
            })
            scale = 100.0 if outcome == "binary" else 1.0
            for statistic in ["Estimate", "SE", "CI_Lower", "CI_Upper"]:
                row[f"Display_{statistic}"] = row[statistic] * scale
            contrasts.append(row)
        for source in ["experimental", "observed"]:
            source_mask = released[SOURCE].eq(source).to_numpy()
            y = released.loc[source_mask, column].to_numpy(dtype=float)
            in_bound = eligible[source_mask]
            n_excluded = int((~in_bound).sum())
            retained_mean = float(y[in_bound].mean())
            excluded_mean = float(y[~in_bound].mean()) if n_excluded else None
            mean_shift = (
                n_excluded / len(y) * (excluded_mean - retained_mean)
                if excluded_mean is not None else 0.0
            )
            np.testing.assert_allclose(y.mean(), retained_mean + mean_shift, rtol=1e-12, atol=1e-12)
            decomposition.append({
                "Outcome": outcome, SOURCE: source,
                "N_Released_Controls": len(y), "N_Retained": int(in_bound.sum()),
                "N_Excluded": n_excluded, "Excluded_Fraction": n_excluded / len(y),
                "Released_Mean": float(y.mean()), "Retained_Mean": retained_mean,
                "Excluded_Mean": excluded_mean, "Released_Minus_Retained": mean_shift,
                "Unit": "probability" if outcome == "binary" else "Outcome_Continuous units",
            })
    return pd.DataFrame(contrasts), pd.DataFrame(decomposition), excluded


def _regression_covariates(X: pd.DataFrame) -> np.ndarray:
    values = X.to_numpy(dtype=float, copy=True)
    sd = values.std(axis=0, ddof=1)
    values = values[:, sd > 0]
    if values.shape[1]:
        values -= values.mean(axis=0)
        values /= sd[sd > 0]
    # Constant columns must not suppress the intercept in regression_ate.
    return values


def estimate_population(
    sample: pd.DataFrame, predictions: pd.DataFrame, mask: np.ndarray,
    population: str, psm_reason: str,
) -> pd.DataFrame:
    _require(mask.dtype == bool and mask.shape == (len(sample),), "Invalid population mask.")
    source = _groups(sample.loc[mask, PLACEBO].to_numpy())
    e = predictions.loc[mask, PROPENSITY].to_numpy()
    X = _regression_covariates(sample.loc[mask, COVARIATES])
    n, n1, n0 = len(source), int(source.sum()), int((source == 0).sum())
    rows = []
    for outcome, column in OUTCOMES.items():
        y = sample.loc[mask, column].to_numpy(dtype=float)
        mu0 = predictions.loc[mask, f"mu0_{outcome}"].to_numpy()
        mu1 = predictions.loc[mask, f"mu1_{outcome}"].to_numpy()
        scores = score_arrays(y, source, e, mu0, mu1)
        naive, variance = difference_in_means(pd.DataFrame({"y": y, "D": source}))
        estimates = [inference_row("Naive", naive, np.sqrt(variance), "Difference in means; unequal group variances")]
        reg, reg_se = regression_ate(y, source, X)
        estimates.append(inference_row("Reg", reg, reg_se, "OLS coefficient; HC1 robust covariance"))
        om = outcome_modeling_ate(y, source, mu0, mu1)
        om_se = np.sqrt(
            np.mean((y[source == 1] - mu1[source == 1]) ** 2) / n1
            + np.mean((y[source == 0] - mu0[source == 0]) ** 2) / n0
        )
        estimates.append(inference_row("OM", om, om_se, "Notebook residual-MSE approximation"))
        for method, score in scores.items():
            estimates.append(inference_row(
                method, float(score.mean()), float(score.std(ddof=1) / np.sqrt(n)),
                "Cross-fitted AIPW score SD/sqrt(n)" if method == "DR" else "Notebook plug-in IPW score SD/sqrt(n)",
            ))
        estimates.append({
            "Estimator": "PSM", "Estimate": np.nan, "SE": np.nan,
            "CI_Lower": np.nan, "CI_Upper": np.nan, "P_Value": np.nan,
            "SE_Method": psm_reason, "Status": "not_run",
        })
        scale = 100.0 if outcome == "binary" else 1.0
        for row in estimates:
            row.update({
                "Outcome": outcome, "Population": population, "N": n,
                "N_Experimental_Controls": n1, "N_Observed_Controls": n0,
                "Unit": "probability difference" if outcome == "binary" else "Outcome_Continuous units",
                "Display_Unit": "percentage points" if outcome == "binary" else "Outcome_Continuous units",
            })
            for statistic in ["Estimate", "SE", "CI_Lower", "CI_Upper"]:
                row[f"Display_{statistic}"] = row[statistic] * scale
            rows.append(row)
    return pd.DataFrame(rows)


def estimate_results(sample: pd.DataFrame, predictions: pd.DataFrame, psm_reason: str) -> pd.DataFrame:
    full = estimate_population(sample, predictions, np.ones(len(sample), dtype=bool), "full", psm_reason)
    keep = predictions[KEEP].to_numpy(dtype=bool)
    if keep.all():
        trimmed = full.assign(Population="crump")
    else:
        trimmed = estimate_population(sample, predictions, keep, "crump", psm_reason)
    return pd.concat([full, trimmed], ignore_index=True)


def collect_diagnostics(
    sample: pd.DataFrame, predictions: pd.DataFrame, alpha: float, output: Path
) -> dict:
    source = sample[PLACEBO].to_numpy()
    e = predictions[PROPENSITY].to_numpy()
    keep = predictions[KEEP].to_numpy(dtype=bool)
    counts, quantiles, balance, metrics, fold_scores = [], [], [], [], []
    populations = {"full": np.ones(len(sample), dtype=bool), "crump": keep}
    overlap = {}
    for label, mask in populations.items():
        s, p = source[mask], e[mask]
        _groups(s)
        weights = np.where(s == 1, 1 / p, 1 / (1 - p))
        overlap[label] = {
            "n": int(mask.sum()),
            "auc": float(roc_auc_score(s, p)),
            "log_loss": float(log_loss(s, p, labels=[0, 1])),
            "brier_score": float(brier_score_loss(s, p)),
        }
        for group, name in [(1, "experimental"), (0, "observed")]:
            group_mask = source == group
            pg = p[s == group]
            wg = weights[s == group]
            overlap[label][f"ess_{name}"] = float(wg.sum() ** 2 / (wg @ wg))
            overlap[label][f"max_weight_{name}"] = float(wg.max())
            overlap[label][f"ht_weight_mass_{name}_over_n"] = float(wg.sum() / len(s))
            counts.append({
                "Population": label, SOURCE: name, PLACEBO: group,
                "retained": int((group_mask & mask).sum()),
                "excluded_by_overlap": int((group_mask & ~mask).sum()),
                "excluded_below_alpha": int((group_mask & (e < alpha)).sum()) if label == "crump" else 0,
                "excluded_above_one_minus_alpha": int((group_mask & (e > 1 - alpha)).sum()) if label == "crump" else 0,
            })
            row = {"Population": label, SOURCE: name, "N": len(pg)}
            row.update(dict(zip(
                ["min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"],
                np.quantile(pg, [0, .01, .05, .25, .5, .75, .95, .99, 1]).tolist(),
            )))
            quantiles.append(row)
        for column in COVARIATES:
            values = sample.loc[mask, column].to_numpy(dtype=float)
            x0, x1 = values[s == 0], values[s == 1]
            denominator = np.sqrt((x0.var(ddof=1) + x1.var(ddof=1)) / 2)
            delta = x1.mean() - x0.mean()
            _require(denominator > 0 or delta == 0, f"{column} perfectly separates the two sources.")
            weighted_delta = np.average(x1, weights=weights[s == 1]) - np.average(x0, weights=weights[s == 0])
            balance.append({
                "Population": label, "Covariate": column,
                "Experimental_Mean": float(x1.mean()), "Observed_Mean": float(x0.mean()),
                "SMD": float(delta / denominator) if denominator else 0.0,
                "IPW_SMD": float(weighted_delta / denominator) if denominator else 0.0,
                "Constant": bool(denominator == 0),
            })
        for outcome, column in OUTCOMES.items():
            y = sample.loc[mask, column].to_numpy(dtype=float)
            mu0 = predictions.loc[mask, f"mu0_{outcome}"].to_numpy()
            mu1 = predictions.loc[mask, f"mu1_{outcome}"].to_numpy()
            factual = np.where(s == 1, mu1, mu0)
            for group, name in [(0, "observed"), (1, "experimental")]:
                selected = s == group
                yg, fitted = y[selected], factual[selected]
                metrics.append({
                    "Population": label, "Outcome": outcome, SOURCE: name,
                    "N": int(selected.sum()), "Outcome_Mean": float(yg.mean()),
                    "Factual_RMSE": float(np.sqrt(np.mean((yg - fitted) ** 2))),
                    "Factual_R2": float(r2_score(yg, fitted)) if yg.var() > 0 else None,
                    "Factual_Mean_Residual": float(np.mean(yg - fitted)),
                    "Prediction_Min": float(fitted.min()), "Prediction_Max": float(fitted.max()),
                    "Binary_Predictions_Outside_0_1": int(((fitted < 0) | (fitted > 1)).sum()) if outcome == "binary" else 0,
                })
            score = score_arrays(y, s, p, mu0, mu1)["DR"]
            assignment = predictions.loc[mask, FOLD].to_numpy()
            for fold in sorted(np.unique(assignment)):
                selected = assignment == fold
                fold_scores.append({
                    "Population": label, "Outcome": outcome, FOLD: int(fold),
                    "N": int(selected.sum()), "DR": float(score[selected].mean()),
                    "OM_Component": float((mu1[selected] - mu0[selected]).mean()),
                    "Experimental_Residual_Correction": float((s[selected] * (y[selected] - mu1[selected]) / p[selected]).mean()),
                    "Observed_Residual_Correction": float(((1 - s[selected]) * (y[selected] - mu0[selected]) / (1 - p[selected])).mean()),
                })
    bin_id = np.minimum((10 * e).astype(int), 9)
    calibration = pd.DataFrame({"bin": bin_id, "predicted": e, "source_label": source}).groupby("bin").agg(
        N=("source_label", "size"), Mean_Prediction=("predicted", "mean"),
        Experimental_Fraction=("source_label", "mean"),
    ).reset_index()
    balance_frame = pd.DataFrame(balance)
    for label in populations:
        selected = balance_frame.loc[balance_frame["Population"] == label]
        overlap[label]["max_abs_smd"] = float(selected["SMD"].abs().max())
        overlap[label]["max_abs_ipw_smd"] = float(selected["IPW_SMD"].abs().max())
        overlap[label]["ipw_smd_over_0_1"] = int((selected["IPW_SMD"].abs() > .1).sum())
    for name, rows in [
        ("population_counts", counts), ("propensity_quantiles", quantiles),
        ("covariate_balance", balance), ("outcome_diagnostics", metrics),
        ("fold_dr_decomposition", fold_scores),
    ]:
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    calibration.to_csv(output / "propensity_calibration.csv", index=False)
    diagnostic = {
        "crump_alpha": alpha,
        "same_full_and_crump_population": bool(keep.all()),
        "propensity_clipping": "none",
        "outcome_prediction_clipping": "none; binary outcome models are RMSE regressors",
        "overlap": overlap,
    }
    _write_json(output / "diagnostics.json", diagnostic)
    return diagnostic


def matching_availability() -> dict:
    """Probe infrastructure only; never substitute cached or Python-NN inference."""
    candidates = []
    on_path = shutil.which("Rscript")
    if on_path:
        candidates.append(Path(on_path))
    if os.environ.get("R_HOME"):
        candidates.extend(Path(os.environ["R_HOME"]) / part for part in [r"bin\x64\Rscript.exe", r"bin\Rscript.exe"])
    root = Path(r"C:\Program Files\R")
    if root.is_dir():
        for directory in sorted(root.glob("R-*"), reverse=True):
            candidates.extend(directory / part for part in [r"bin\x64\Rscript.exe", r"bin\Rscript.exe"])
    executable = next((path for path in candidates if path.is_file()), None)
    if executable is None:
        return {"status": "unavailable", "reason": "PSM not run: Rscript is unavailable."}
    code = (
        'cat(R.version.string, "\\n"); '
        'if (!requireNamespace("Matching", quietly=TRUE)) {'
        'cat("R package Matching is unavailable in the active R libraries.\\n"); quit(status=2)}; '
        'cat("Matching", as.character(packageVersion("Matching")), "\\n")'
    )
    try:
        probe = subprocess.run(
            [str(executable), "--vanilla", "-e", code],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "unavailable", "reason": "PSM not run: R/Matching probe exceeded 30 seconds.", "Rscript": str(executable)}
    details = (probe.stdout + probe.stderr).strip()
    if probe.returncode:
        return {"status": "unavailable", "reason": f"PSM not run: {details}", "Rscript": str(executable)}
    return {
        "status": "not_run", "Rscript": str(executable),
        "reason": "PSM not run: this bounded Python work-up implements Reg, OM, IPW and DR; a separate R Matching run is required.",
        "probe": details,
    }


def run_analysis(data_dir: Path, output: Path, config: Config) -> Path:
    config.validate()
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "flaml_logs").mkdir()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log", encoding="utf-8")],
    )
    started = time.perf_counter()
    code_hashes = save_source_snapshot(output)
    paths = {name: data_dir / f"FINAL_PUBLIC_{name}.parquet" for name in ["experimental", "observed"]}
    released, audit = build_placebo_sample(pd.read_parquet(paths["experimental"]), pd.read_parquet(paths["observed"]))
    released_n = len(released)
    released_maxima = released.groupby(SOURCE)[OUTCOME_CONTINUOUS].max().to_dict()
    sample, audit = apply_common_upper_bound(released, audit, config.continuous_upper)
    if config.continuous_upper is not None:
        contrasts, decomposition, excluded = common_bound_diagnostics(released, config.continuous_upper)
        contrasts.to_csv(output / "unadjusted_bound_comparison.csv", index=False)
        decomposition.to_csv(output / "outcome_bound_decomposition.csv", index=False)
        excluded.to_parquet(output / "common_bound_exclusions.parquet", index=False)
        LOGGER.warning(
            "Outcome-dependent common eligibility bound Y<=%g: retaining %d/%d D=0 rows for BOTH outcomes. "
            "This is separate from outcome-independent Crump overlap trimming.",
            config.continuous_upper, len(sample), released_n,
        )
    del released
    audit.to_csv(output / "source_audit.csv", index=False)
    psm = matching_availability()
    LOGGER.warning(psm["reason"])
    manifest = {
        "status": "started_not_yet_validated",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in [
            "numpy", "pandas", "scikit-learn", "lightgbm", "flaml", "statsmodels", "scipy", "pyarrow",
        ]},
        "input_files": {name: {"path": str(path.resolve()), "sha256": _file_hash(path)} for name, path in paths.items()},
        "code_sha256": code_hashes,
        "source_snapshot": "source_snapshot",
        "covariates": COVARIATES,
        "source_sign": "experimental controls (1) minus observational controls (0)",
        "actual_exposure": "New_Feature == False in both input files",
        "sample_rows": len(sample),
        "released_unexposed_rows": released_n,
        "released_control_continuous_maxima": released_maxima,
        "excluded_common_outcome_bound": released_n - len(sample),
        "outcome_eligibility": (
            "As released; no additional outcome restriction"
            if config.continuous_upper is None else
            f"Outcome_Continuous <= {config.continuous_upper:g}, identical in both sources and for both outcomes"
        ),
        "outcome_dependent_eligibility": config.continuous_upper is not None,
        "ensembling": False,
        "analysis_subsampling": False,
        "flaml_search_subsampling": False,
        "propensity_objective": "classification log_loss",
        "outcome_objective_both_outcomes": "regression rmse",
        "outer_split_stratification": "sample origin only, never either outcome",
        "psm": psm,
        "primary_estimator": "DR: direct cross-fitted AIPW, sharing nuisances with OM/IPW; no EconML refit",
        "inference": "Pointwise normal 95% intervals, two-sided nominal p-values; see report for limitations.",
        "target": "Pooled eligible untreated devices; optional common outcome bound precedes new source-propensity Crump trimming.",
    }
    _write_json(output / "manifest.json", manifest)
    LOGGER.info("Using all %d eligible unexposed rows and exactly %d adjustment covariates.", len(sample), len(COVARIATES))
    predictions, records, alpha = cross_fit_nuisances(sample, config, output)
    predictions.to_parquet(output / "placebo_predictions.parquet", index=False)
    results = estimate_results(sample, predictions, psm["reason"])
    if config.continuous_upper is not None:
        results["Common_Continuous_Upper"] = config.continuous_upper
    results.to_csv(output / "placebo_results.csv", index=False)
    collect_diagnostics(sample, predictions, alpha, output)
    manifest.update({
        "status": "analysis_written_pending_validation",
        "crump_alpha": alpha, "models_fitted": len(records),
        "analysis_seconds": time.perf_counter() - started,
    })
    _write_json(output / "manifest.json", manifest)
    return output


def validate_saved_run(output: Path) -> dict:
    """Reload inputs, fitted models and outputs, and recompute the reported inference."""
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    config = Config(**manifest["config"])
    frames = {}
    for source, entry in manifest["input_files"].items():
        path = Path(entry["path"])
        _require(_file_hash(path) == entry["sha256"], f"{source}: input changed since fitting.")
        frames[source] = pd.read_parquet(path)
    for name, expected_hash in manifest["code_sha256"].items():
        _require(
            _file_hash(ROOT / "src" / name) == expected_hash,
            f"{name}: code changed since fitting. Run verification from this run's source_snapshot directory.",
        )
        snapshot_file = output / "source_snapshot" / "src" / name
        _require(snapshot_file.is_file() and _file_hash(snapshot_file) == expected_hash, f"{name}: source snapshot mismatch.")
    released, audit = build_placebo_sample(frames["experimental"], frames["observed"])
    del frames
    _require(
        manifest["released_control_continuous_maxima"] == released.groupby(SOURCE)[OUTCOME_CONTINUOUS].max().to_dict(),
        "Released source maxima do not match the input data.",
    )
    sample, audit = apply_common_upper_bound(released, audit, config.continuous_upper)
    if config.continuous_upper is not None:
        contrasts, decomposition, excluded = common_bound_diagnostics(released, config.continuous_upper)
        pd.testing.assert_frame_equal(
            contrasts, pd.read_csv(output / "unadjusted_bound_comparison.csv"),
            check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12,
        )
        pd.testing.assert_frame_equal(
            decomposition, pd.read_csv(output / "outcome_bound_decomposition.csv"),
            check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12,
        )
        pd.testing.assert_frame_equal(excluded, pd.read_parquet(output / "common_bound_exclusions.parquet"))
        _require(manifest["released_unexposed_rows"] == len(released), "Incorrect pre-restriction sample count.")
        _require(manifest["excluded_common_outcome_bound"] == len(excluded), "Incorrect outcome-bound exclusion count.")
    del released
    predictions = pd.read_parquet(output / "placebo_predictions.parquet")
    pd.testing.assert_frame_equal(
        sample[[*PROVENANCE, *OUTCOMES.values()]], predictions[[*PROVENANCE, *OUTCOMES.values()]],
    )
    pd.testing.assert_frame_equal(audit, pd.read_csv(output / "source_audit.csv"), check_dtype=False)
    folds = make_folds(sample[PLACEBO].to_numpy(), config)
    np.testing.assert_array_equal(predictions[FOLD].to_numpy(), validate_folds(folds, sample[PLACEBO].to_numpy()))
    alpha, keep = crump_rule(predictions[PROPENSITY].to_numpy())
    np.testing.assert_allclose(alpha, manifest["crump_alpha"], atol=1e-14, rtol=0)
    np.testing.assert_array_equal(keep, predictions[KEEP].to_numpy())
    records = json.loads((output / "model_manifest.json").read_text(encoding="utf-8"))
    expected_columns = {PROPENSITY, "mu0_continuous", "mu1_continuous", "mu0_binary", "mu1_binary"}
    expected_keys = {(fold, column) for fold in range(config.folds) for column in expected_columns}
    _require(
        len(records) == 5 * config.folds
        and {(r["fold"], r["prediction_column"]) for r in records} == expected_keys,
        "Missing, duplicate, or unexpected fitted nuisance models.",
    )
    source = sample[PLACEBO].to_numpy()
    native_checks = 0
    for record in records:
        column = record["prediction_column"]
        train, test = folds[record["fold"]]
        if column != PROPENSITY:
            train = train[source[train] == int(column[2])]
        _require(len(train) == record["n_train"] and len(test) == record["n_test"], "Model training/test counts do not match.")
        _require(record["final_fit_rows_verified"] == len(train), "Final fit did not use all training rows.")
        _require(_index_hash(train) == record["train_index_sha256"], "Model training provenance mismatch.")
        _require(_index_hash(test) == record["test_index_sha256"], "Model test provenance mismatch.")
        _require(record["metric"] == ("log_loss" if column == PROPENSITY else "rmse"), "Tuning objective mismatch.")
        _require(np.isfinite(record["best_cv_loss"]), "Nonfinite model selection loss.")
        _vector(predictions[column].to_numpy(), column, len(sample))
        check_rows = test[:64]
        fitted = Booster(model_file=str(output / record["model_file"]))
        replayed = fitted.predict(sample.iloc[check_rows][COVARIATES], num_threads=config.jobs)
        np.testing.assert_allclose(replayed, predictions.loc[check_rows, column], rtol=1e-10, atol=1e-12)
        native_checks += len(check_rows)
    recomputed = estimate_results(sample, predictions, manifest["psm"]["reason"])
    if config.continuous_upper is not None:
        recomputed["Common_Continuous_Upper"] = config.continuous_upper
    saved = pd.read_csv(output / "placebo_results.csv")
    pd.testing.assert_frame_equal(recomputed, saved, check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-12)
    estimated = saved.loc[saved["Status"] == "estimated"]
    numeric = estimated[["Estimate", "SE", "CI_Lower", "CI_Upper", "P_Value"]].to_numpy(dtype=float)
    _require(bool(np.isfinite(numeric).all()), "Nonfinite reported inference.")
    _require(bool((estimated["SE"] > 0).all()), "Nonpositive reported SE.")
    _require(bool(estimated["P_Value"].between(0, 1).all()), "Invalid p-values.")
    _require(
        bool(((estimated["CI_Lower"] <= estimated["Estimate"]) & (estimated["Estimate"] <= estimated["CI_Upper"])).all()),
        "Confidence intervals do not contain their estimates.",
    )
    for row in pd.read_csv(output / "population_counts.csv").to_dict("records"):
        source_mask = sample[SOURCE].eq(row[SOURCE]).to_numpy()
        population_mask = keep if row["Population"] == "crump" else np.ones(len(sample), dtype=bool)
        _require(row["retained"] == int((source_mask & population_mask).sum()), "Incorrect retained count.")
        _require(row["excluded_by_overlap"] == int((source_mask & ~population_mask).sum()), "Incorrect trimming exclusion count.")
        _require(
            row["excluded_below_alpha"] + row["excluded_above_one_minus_alpha"] == row["excluded_by_overlap"],
            "Trimming tail counts do not add up.",
        )
    validation = {
        "status": "passed_computational_validation_not_a_test_of_the_null",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(sample), "actual_exposed_rows": int(sample[TREATMENT].sum()),
        "covariates": len(COVARIATES), "exactly_one_test_fold_per_row": True,
        "training_test_disjoint": True, "input_and_code_hashes_verified": True,
        "source_row_provenance_and_both_outcomes_verified": True,
        "new_crump_threshold_and_masks_recomputed": True,
        "all_point_estimates_se_ci_pvalues_recomputed": True,
        "binary_percentage_point_conversion_verified": True,
        "saved_single_models_checked": len(records),
        "saved_native_model_prediction_checks": native_checks,
        "source_snapshot_verified": True,
        "common_outcome_upper": config.continuous_upper,
        "common_bound_construction_and_decomposition_verified": config.continuous_upper is not None,
    }
    _write_json(output / "validation.json", validation)
    manifest["status"] = "complete_and_validated"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _write_report(output, manifest, saved)
    _write_json(output / "manifest.json", manifest)
    return validation


def _write_report(output: Path, manifest: dict, results: pd.DataFrame) -> None:
    diagnostic = json.loads((output / "diagnostics.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(output / "source_audit.csv")
    counts = pd.read_csv(output / "population_counts.csv")
    balance = pd.read_csv(output / "covariate_balance.csv")
    primary = results.loc[(results["Population"] == "crump") & (results["Estimator"] == "DR")]
    paragraphs = []
    for row in primary.to_dict("records"):
        conclusion = "A nonzero source difference is statistically detected" if row["P_Value"] < .05 else "No source difference is statistically detected; equivalence is not established"
        paragraphs.append(
            f"<p><b>{html.escape(row['Outcome'])}, primary DR:</b> "
            f"{row['Display_Estimate']:+.6f} {html.escape(row['Display_Unit'])}; "
            f"95% CI [{row['Display_CI_Lower']:+.6f}, {row['Display_CI_Upper']:+.6f}], "
            f"p={row['P_Value']:.6g}. {conclusion}.</p>"
        )
    tables = []
    for population in ["full", "crump"]:
        if population == "crump" and diagnostic["same_full_and_crump_population"]:
            tables.append("<p>Crump retains every row; full and Crump results are exactly identical.</p>")
            continue
        display = results.loc[results["Population"] == population, [
            "Outcome", "Estimator", "Display_Estimate", "Display_SE", "Display_CI_Lower",
            "Display_CI_Upper", "P_Value", "Display_Unit", "Status",
        ]]
        tables.append(f"<h3>{population}</h3>" + display.to_html(index=False, float_format=lambda x: f"{x:.6g}", na_rep="Not run"))
    largest = balance.loc[balance["Population"] == "crump"].copy()
    largest = largest.loc[largest["IPW_SMD"].abs().sort_values(ascending=False).index].head(10)
    overlap = pd.DataFrame.from_dict(diagnostic["overlap"], orient="index")
    upper = manifest["config"].get("continuous_upper")
    maxima = manifest["released_control_continuous_maxima"]
    preprocessing = f"""<h2>Input preprocessing</h2><p class="note">
The public files in this workspace already reflect source-specific 97th-percentile continuous-outcome
row deletion in the preparation notebook, calculated separately on each entire source dataset,
including actually exposed devices. In this run's inputs, the actual-D=0 continuous outcome ends at
{maxima['experimental']:g} in the experiment and {maxima['observed']:g} in the observational source.
This is row truncation, not winsorization.
No percentile filter is applied a second time. An as-released source contrast can therefore
reflect unequal outcome-based eligibility, even with excellent measured covariate balance.</p>"""
    if upper is not None:
        bound_comparison = pd.read_csv(output / "unadjusted_bound_comparison.csv")
        mean_decomposition = pd.read_csv(output / "outcome_bound_decomposition.csv")
        preprocessing += f"""<p class="note"><b>Shared eligibility rule:</b>
Outcome_Continuous &lt;= {upper:g} in BOTH actual-D=0 groups, used for BOTH outcomes.
This retains {manifest['sample_rows']:,} of {manifest['released_unexposed_rows']:,} released controls
and excludes {manifest['excluded_common_outcome_bound']:,}.
The numerical bound was fixed before refitting; it was not searched to obtain a null.
All propensity and outcome models below were newly fitted on this eligible population.</p>
<p>This is a deliberately <b>outcome-dependent preprocessing sensitivity</b>, not the
outcome-independent Crump overlap rule. It targets an outcome-restricted untreated population,
does not recover discarded pre-release rows, and must not be interpreted as a clean test
of the entire untruncated population or as an actual-treatment effect.</p>
<h3>Unadjusted means check, before any new ML fits</h3>
{bound_comparison[['Outcome', 'Population', 'N_Experimental_Controls', 'N_Observed_Controls', 'Display_Estimate', 'Display_SE', 'Display_CI_Lower', 'Display_CI_Upper', 'P_Value', 'Display_Unit']].to_html(index=False, float_format=lambda x: f"{x:.8g}")}
<h3>Exact mean-shift decomposition</h3>
<p>For each source: released mean = retained mean + excluded fraction *
(excluded mean - retained mean). Thus the released source contrast equals the shared-bound
contrast plus the experimental mean shift minus the observational mean shift.
This identity describes the arithmetic; it is not a test of a change between correlated estimators.</p>
{mean_decomposition.to_html(index=False, float_format=lambda x: f"{x:.8g}", na_rep="Not applicable")}
<p><a href="unadjusted_bound_comparison.csv">Before/after unadjusted contrasts</a> |
<a href="outcome_bound_decomposition.csv">Mean-shift decomposition</a> |
<a href="common_bound_exclusions.parquet">Excluded-row provenance</a></p>"""
    else:
        preprocessing += "<p>No additional common outcome bound was imposed in this run.</p>"
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Control-sample placebo validation</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1250px;margin:35px auto;padding:0 20px;line-height:1.5}}
table{{border-collapse:collapse;font-size:14px}}td,th{{border:1px solid #bbb;padding:6px;text-align:right}}
th{{background:#eef2f6}}code{{background:#f1f1f1}}.note{{border-left:4px solid #456;padding-left:15px}}</style></head><body>
<h1>Control-sample placebo validation</h1>
<p>Sign: <b>experimental controls minus observational controls</b>. Every included device has
<code>New_Feature == False</code>. Zero is a substantive placebo benchmark under comparability
assumptions, not a known randomized source-assignment effect.</p>
{''.join(paragraphs)}
{preprocessing}
<h2>Design and execution</h2>
<p>All {manifest['sample_rows']:,} eligible rows; 41 specified covariates, with neither outcome,
actual exposure, source label, nor row/fold identifier among adjustment variables.
{manifest['config']['folds']} source-stratified outer folds; {manifest['config']['inner_folds']}-fold
FLAML CV entirely inside each training fold. One selected LightGBM per role/fold, {manifest['models_fitted']}
models total; no model averaging, no analysis or search subsampling, and no old caches.
Budget {manifest['config']['time_budget']:g} seconds/search, at most {manifest['config']['max_iter']} trials;
final refits/overhead are additional. Actual analysis time: {manifest['analysis_seconds'] / 60:.2f} minutes.</p>
<p>Origin objective: <b>classification log loss</b>. Both outcomes: <b>regression RMSE</b>,
matching the current Python nuisance helpers. No claim of probability calibration is made.
Binary outcome regression predictions are not clipped; see outcome diagnostics for excursions outside [0,1].
Propensities are also not clipped. Tuning/search seeds and fitted parameters are saved.</p>
<h3>Original source counts and actual-exposure exclusions</h3>{audit.to_html(index=False)}
<h3>Outcome-independent overlap selection</h3>
<p>New held-out source propensities yield Crump alpha = {manifest['crump_alpha']:.10g}.
The overlap-restricted target is the eligible untreated sample satisfying alpha &lt;= e(source|X) &lt;= 1-alpha.
Here, "full" means the full eligible sample after any explicitly specified common outcome bound.
The same mask is used for both outcomes. Outcome models remain fitted on outer-training controls;
their held-out predictions are reused for the full and restricted targets, without retuning to a result.</p>
{counts.to_html(index=False)}
<h2>Estimates and uncertainty</h2>
<p>Continuous effects are in native <code>Outcome_Continuous</code> units; no transformation or
standardization of the outcome is applied. Binary effects below are <b>percentage points</b>
(100 times the probability difference); CSV raw columns also retain probability units.
Intervals are pointwise 95%; p-values are two-sided and nominal (not multiplicity-adjusted).</p>
{''.join(tables)}
<p><b>PSM:</b> {html.escape(manifest['psm']['reason'])}
No cached results, nearest-neighbor substitute, or fabricated matching standard errors are used.</p>
<h2>Diagnostics</h2>{overlap.to_html(float_format=lambda x: f"{x:.6g}")}
<h3>Largest residual weighted covariate imbalances</h3>
{largest.to_html(index=False, float_format=lambda x: f"{x:.6g}")}
<p>Balance uses experimental-minus-observed standardized mean differences and normalized within-source
IPW means; the denominator is the unweighted pooled within-source SD in each target. These balance
weights are normalized for diagnostics only: the IPW estimator itself is the notebook's unnormalized
Horvitz-Thompson contrast. Held-out AUC, Brier score and calibration bins are descriptive diagnostics.</p>
<h2>Inference and interpretation limits</h2><div class="note"><ul>
<li>Primary DR is the mean cross-fitted AIPW score, with score SD/sqrt(n), using the same newly fitted
nuisances as OM and IPW. It uses the existing AIPW formula directly rather than refitting an EconML learner.</li>
<li>Reg is an additive OLS source coefficient with HC1 covariance, not a nonparametric ATE under arbitrary
heterogeneity. Constant covariates are removed and other covariates are rescaled for numerical stability.</li>
<li>OM uses the notebook residual-MSE SE approximation; IPW uses its plug-in score SE. Neither
non-orthogonal method's interval generally captures all first-order ML nuisance-estimation error.
DR also needs overlap and appropriate nuisance convergence/product-rate assumptions.</li>
<li>Intervals do not incorporate repeated tuning/split variability or uncertainty in the learned
trimming boundary. There is no ensemble, full-pipeline bootstrap, or formal equivalence margin/test.</li>
<li>A common outcome bound is an outcome-dependent, post-hoc diagnostic motivated by the released-data
preprocessing discrepancy. Its intervals are conditional on that fixed target and do not account
for the decision to investigate that target after seeing the original results.</li>
<li>Independent rows are assumed. Public source-row indices are provenance, not device IDs;
cross-file device overlap or clustering cannot be verified from this public schema.</li>
<li>This test excludes observational adopters. Even a precise null cannot establish unconfoundedness
of self-selected actual treatment; it mainly checks comparability of the two untreated samples conditional
on the released covariates. Some released usage covariates are measured during the study period.</li>
<li>Significant source differences are evidence against this comparability check, not estimates of a feature
effect. Nonsignificance is absence of detection, not proof of no bias or equivalence; use the reported CI to assess precision.</li>
</ul></div>
<h2>Reproducibility and validation</h2>
<p><a href="placebo_results.csv">Results CSV</a> |
<a href="manifest.json">Run manifest, versions and input/code hashes</a> |
<a href="validation.json">Executed validation checks</a> |
<a href="model_manifest.json">Per-fold model configurations</a> |
<a href="fold_dr_decomposition.csv">Fold-level DR decomposition</a> |
<a href="outcome_diagnostics.csv">Held-out outcome diagnostics</a> |
<a href="propensity_quantiles.csv">Source-propensity ranges</a> |
<a href="propensity_calibration.csv">Calibration bins</a></p>
<p>The exact Python source is archived under <code>source_snapshot</code>. If the workspace code
later changes, run this version's verification from that directory with
<code>python -m src.placebo_validation --verify-output ..</code>, in the recorded Python environment.</p>
<p>Validation reloads source files and row provenance, verifies exposure/labels, disjoint fold coverage,
trimming counts/masks, finite nuisances and objectives, checks saved native models against held-out predictions,
and recomputes every reported estimate, SE, CI and p-value. Computational validation is not a passed placebo null.</p>
</body></html>
"""
    (output / "placebo_report.html").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--time-budget", type=float, default=30.0)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--continuous-upper", type=float,
        help="Fixed common upper bound on Outcome_Continuous for both D=0 groups and both outcomes; outcome-selected sensitivity.",
    )
    parser.add_argument("--verify-output", type=Path, help="Recheck a saved run without retuning.")
    args = parser.parse_args()
    if args.verify_output is not None:
        output = args.verify_output.resolve()
    else:
        output = args.output_dir or ROOT / "saved_outputs_placebo" / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ")
        output = output.resolve()
        config = Config(
            folds=args.folds, time_budget=args.time_budget, max_iter=args.max_iter,
            seed=args.seed, jobs=args.jobs, continuous_upper=args.continuous_upper,
        )
        run_analysis(args.data_dir.resolve(), output, config)
    validation = validate_saved_run(output)
    print(json.dumps(validation, indent=2))
    results = pd.read_csv(output / "placebo_results.csv")
    print(results.loc[(results["Population"] == "crump") & (results["Status"] == "estimated"), [
        "Outcome", "Estimator", "Display_Estimate", "Display_SE", "Display_CI_Lower",
        "Display_CI_Upper", "P_Value", "Display_Unit",
    ]].to_string(index=False))
    print(f"Validated report: {output / 'placebo_report.html'}")


if __name__ == "__main__":
    main()
