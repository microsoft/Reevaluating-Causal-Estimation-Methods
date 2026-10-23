"""Paired actual-treatment reassessment, with original and shared overlap targets.

No placebo labels or pre-existing fitted models/results are used. The two
regimes use identical single-model FLAML settings. Experimental nuisances can
be reused between regimes only when their entire eligible DataFrames agree.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import shutil
import sys
import time

from lightgbm import Booster
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import brier_score_loss, log_loss, r2_score, roc_auc_score

from .data_loading import COVARIATES, OUTCOME_BINARY, OUTCOME_CONTINUOUS, TREATMENT, difference_in_means
from .estimators import outcome_modeling_ate, regression_ate
from .placebo_validation import (
    Config, FOLD, OUTCOMES, SOURCE_ROW, TREATMENT_PROPENSITY,
    _file_hash, _fit_single, _index_hash, _regression_covariates,
    _require, _vector, _write_json, crump_rule, inference_row, make_folds,
    matching_availability, save_source_snapshot, score_arrays, validate_folds,
)
from .plotting import plot_ate_comparison, plot_propensity_distributions


ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)
D = "D"
ADOPTION_SCORE = "observational_propensity_on_experiment"
DR_MIN_PROPENSITY = 0.01
REGIMES = ("as_released", "common_cutoff")
COHORTS = ("experimental", "observed")
POPULATIONS = ("full", "legacy_overlap", "shared_overlap")
METHODS = ("Naive", "Reg", "OM", "IPW", "DR", "PSM")
PREDICTION_COLUMNS = (
    TREATMENT_PROPENSITY, "mu0_continuous", "mu1_continuous", "mu0_binary", "mu1_binary",
)
IDENTITY_COLUMNS = [SOURCE_ROW, TREATMENT, D, *OUTCOMES.values()]


@dataclass
class PreparedData:
    cohorts: dict[str, dict[str, pd.DataFrame]]
    model_frames: dict[str, pd.DataFrame]
    model_map: dict[str, dict[str, str]]
    audit: pd.DataFrame
    mean_shifts: pd.DataFrame
    paths: dict[str, Path]


def prepare_cohort(raw: pd.DataFrame, upper: float | None) -> pd.DataFrame:
    required = [TREATMENT, *OUTCOMES.values(), *COVARIATES]
    _require(raw.columns.is_unique and set(raw.columns) == set(required), "Expected the 44-column public schema.")
    _require(
        pd.api.types.is_bool_dtype(raw[TREATMENT]) and not raw[TREATMENT].isna().any(),
        "New_Feature must be nonmissing Boolean actual exposure.",
    )
    _require(len(COVARIATES) == len(set(COVARIATES)) == 41, "Expected 41 unique adjustment covariates.")
    _require(not set(COVARIATES).intersection([TREATMENT, D, SOURCE_ROW, *OUTCOMES.values()]), "Adjustment-column leakage.")
    for column in [*COVARIATES, *OUTCOMES.values()]:
        _require(pd.api.types.is_numeric_dtype(raw[column]), f"{column} is not numeric.")
        _vector(raw[column].to_numpy(), column, len(raw))
    _require(raw[OUTCOME_BINARY].isin([0, 1]).all(), "The binary outcome must be 0/1.")
    result = raw.copy()
    result[SOURCE_ROW] = np.arange(len(raw), dtype=np.int64)
    result[D] = result[TREATMENT].astype(np.int8)
    if upper is not None:
        _require(np.isfinite(upper) and upper >= 0, "Invalid common continuous-outcome upper bound.")
        result = result.loc[result[OUTCOME_CONTINUOUS] <= upper].copy()
    result = result.reset_index(drop=True)
    _require(
        bool((np.bincount(result[D].to_numpy(), minlength=2) >= 2).all()),
        "Both actual treatment arms need at least two eligible observations.",
    )
    np.testing.assert_array_equal(result[D], result[TREATMENT].astype(int))
    return result


def prepare_inputs(data_dir: Path, upper: float) -> PreparedData:
    paths = {cohort: data_dir / f"FINAL_PUBLIC_{cohort}.parquet" for cohort in COHORTS}
    cohorts: dict[str, dict[str, pd.DataFrame]] = {regime: {} for regime in REGIMES}
    model_map: dict[str, dict[str, str]] = {regime: {} for regime in REGIMES}
    model_frames, audit, shifts = {}, [], []
    for cohort, path in paths.items():
        raw = pd.read_parquet(path)
        base = prepare_cohort(raw, None)
        common = prepare_cohort(raw, upper)
        for regime, frame in zip(REGIMES, [base, common]):
            key = f"{cohort}_{regime}"
            if regime == "common_cutoff" and common.equals(base):
                key = model_map["as_released"][cohort]
                frame = base
            else:
                model_frames[key] = frame
            cohorts[regime][cohort] = frame
            model_map[regime][cohort] = key
            n1, n0 = int(frame[D].sum()), int((frame[D] == 0).sum())
            audit.append({
                "Regime": regime, "Cohort": cohort, "Model_Set": key,
                "Released_N": len(base), "Eligible_N": len(frame),
                "N_Treated": n1, "N_Control": n0,
                "Dropped_Treated": int(base[D].sum()) - n1,
                "Dropped_Control": int((base[D] == 0).sum()) - n0,
                "Released_Continuous_Max": float(base[OUTCOME_CONTINUOUS].max()),
                "Common_Upper": upper if regime == "common_cutoff" else None,
            })
        for treatment in [0, 1]:
            before = base.loc[base[D] == treatment]
            kept = before[OUTCOME_CONTINUOUS].le(upper).to_numpy()
            fraction = float((~kept).mean())
            for outcome, column in OUTCOMES.items():
                y = before[column].to_numpy(dtype=float)
                selected_mean = float(y[kept].mean())
                excluded_mean = float(y[~kept].mean()) if (~kept).any() else None
                shift = fraction * (excluded_mean - selected_mean) if excluded_mean is not None else 0.0
                np.testing.assert_allclose(y.mean(), selected_mean + shift, rtol=1e-12, atol=1e-12)
                shifts.append({
                    "Cohort": cohort, D: treatment, "Outcome": outcome,
                    "N_Released": len(before), "N_Kept": int(kept.sum()), "N_Excluded": int((~kept).sum()),
                    "Excluded_Fraction": fraction, "Released_Mean": float(y.mean()),
                    "Common_Mean": selected_mean, "Excluded_Mean": excluded_mean,
                    "Released_Minus_Common": shift,
                })
    return PreparedData(cohorts, model_frames, model_map, pd.DataFrame(audit), pd.DataFrame(shifts), paths)


def fit_nuisances(frame: pd.DataFrame, config: Config, folder: Path) -> pd.DataFrame:
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "models").mkdir()
    (folder / "flaml_logs").mkdir()
    actual_d = frame[D].to_numpy()
    np.testing.assert_array_equal(actual_d, frame[TREATMENT].astype(int))
    folds = make_folds(actual_d, config)
    predictions = frame[IDENTITY_COLUMNS].copy()
    predictions[FOLD] = validate_folds(folds, actual_d)
    for column in PREDICTION_COLUMNS:
        predictions[column] = np.nan
    records = []
    X = frame[COVARIATES]
    for fold, (train, test) in enumerate(folds):
        values, record = _fit_single(X, actual_d, train, test, TREATMENT_PROPENSITY, fold, config, folder)
        record["training_target"] = "actual New_Feature exposure; 1=treated"
        predictions.loc[test, TREATMENT_PROPENSITY] = values
        records.append(record)
        _write_json(folder / "model_manifest.json", records)
        for outcome, outcome_column in OUTCOMES.items():
            y = frame[outcome_column].to_numpy(dtype=float)
            for treatment in [0, 1]:
                group_train = train[actual_d[train] == treatment]
                column = f"mu{treatment}_{outcome}"
                values, record = _fit_single(X, y, group_train, test, column, fold, config, folder)
                record["training_target"] = f"{outcome_column} within actual D={treatment}"
                predictions.loc[test, column] = values
                records.append(record)
                _write_json(folder / "model_manifest.json", records)
    validate_predictions(frame, predictions)
    predictions.to_parquet(folder / "predictions.parquet", index=False)
    return predictions


def validate_predictions(frame: pd.DataFrame, predictions: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(frame[IDENTITY_COLUMNS], predictions[IDENTITY_COLUMNS])
    _require("placebo_treated" not in predictions and "placebo_propensity" not in predictions, "Placebo labels in actual-treatment predictions.")
    for column in PREDICTION_COLUMNS:
        _vector(predictions[column].to_numpy(), column, len(frame))
    e = predictions[TREATMENT_PROPENSITY].to_numpy()
    _require(bool(((e > 0) & (e < 1)).all()), "Actual treatment probabilities must lie strictly in (0,1).")


def outcome_diagnostics(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome, column in OUTCOMES.items():
        for treatment in [0, 1]:
            mask = frame[D].eq(treatment).to_numpy()
            y = frame.loc[mask, column].to_numpy(dtype=float)
            all_predictions = predictions[f"mu{treatment}_{outcome}"].to_numpy()
            factual = all_predictions[mask]
            rows.append({
                "Outcome": outcome, D: treatment, "N_Factual": len(y),
                "Factual_Mean": float(y.mean()),
                "Heldout_RMSE": float(np.sqrt(np.mean((y - factual)**2))),
                "Heldout_R2": float(r2_score(y, factual)) if y.var() > 0 else None,
                "Heldout_Mean_Residual": float(np.mean(y - factual)),
                "All_X_Prediction_Min": float(all_predictions.min()),
                "All_X_Prediction_Max": float(all_predictions.max()),
                "Binary_Predictions_Outside_0_1": int(((all_predictions < 0) | (all_predictions > 1)).sum()) if outcome == "binary" else 0,
            })
    return pd.DataFrame(rows)


def project_adoption_propensity(
    experiment: pd.DataFrame, experimental_predictions: pd.DataFrame,
    observed_folder: Path, config: Config,
) -> pd.DataFrame:
    """Use one observational model per experimental fold; never average models."""
    records = json.loads((observed_folder / "model_manifest.json").read_text(encoding="utf-8"))
    by_fold = {r["fold"]: r for r in records if r["prediction_column"] == TREATMENT_PROPENSITY}
    _require(set(by_fold) == set(range(config.folds)), "Missing observational propensity folds.")
    result = experimental_predictions[[SOURCE_ROW, FOLD]].copy()
    result[ADOPTION_SCORE] = np.nan
    for fold, record in by_fold.items():
        rows = np.flatnonzero(result[FOLD].to_numpy() == fold)
        model = Booster(model_file=str(observed_folder / record["model_file"]))
        result.loc[rows, ADOPTION_SCORE] = model.predict(experiment.iloc[rows][COVARIATES], num_threads=config.jobs)
    values = _vector(result[ADOPTION_SCORE].to_numpy(), ADOPTION_SCORE, len(experiment))
    _require(bool(((values > 0) & (values < 1)).all()), "Invalid projected adoption probabilities.")
    return result


def overlap_masks(
    e_observed: np.ndarray, e_experimental: np.ndarray, adoption_on_experiment: np.ndarray, alpha: float,
) -> dict[str, dict[str, np.ndarray]]:
    e_observed = _vector(e_observed, "observed propensity")
    e_experimental = _vector(e_experimental, "experimental propensity")
    adoption_on_experiment = _vector(adoption_on_experiment, ADOPTION_SCORE, len(e_experimental))
    _require(np.isfinite(alpha) and 0 <= alpha < .5, "Invalid overlap threshold.")
    for values in [e_observed, e_experimental, adoption_on_experiment]:
        _require(bool(((values > 0) & (values < 1)).all()), "Invalid overlap probabilities.")
    observed_keep = (e_observed >= alpha) & (e_observed <= 1 - alpha)
    return {
        "observed": {
            "full": np.ones(len(e_observed), dtype=bool),
            "legacy_overlap": observed_keep,
            "shared_overlap": observed_keep.copy(),
        },
        "experimental": {
            "full": np.ones(len(e_experimental), dtype=bool),
            "legacy_overlap": (e_experimental >= alpha) & (e_experimental <= 1 - alpha),
            "shared_overlap": (adoption_on_experiment >= alpha) & (adoption_on_experiment <= 1 - alpha),
        },
    }


def treatment_scores(
    y: np.ndarray, actual_d: np.ndarray, own_propensity: np.ndarray,
    mu0: np.ndarray, mu1: np.ndarray,
) -> dict[str, np.ndarray]:
    raw = score_arrays(y, actual_d, own_propensity, mu0, mu1)
    clipped = np.clip(own_propensity, DR_MIN_PROPENSITY, 1 - DR_MIN_PROPENSITY)
    dr = score_arrays(y, actual_d, clipped, mu0, mu1)["DR"]
    return {"IPW": raw["IPW"], "DR": dr, "DR_Unclipped": raw["DR"]}


def _missing_row(method: str, status: str, reason: str) -> dict:
    return {
        "Estimator": method, "Estimate": np.nan, "SE": np.nan,
        "CI_Lower": np.nan, "CI_Upper": np.nan, "P_Value": np.nan,
        "Log_P_Value": np.nan, "SE_Method": reason, "Status": status,
    }


def _row(method: str, estimate: float, se: float, description: str) -> dict:
    _require(np.isfinite(estimate) and np.isfinite(se) and se >= 0, f"{method}: invalid estimate or SE.")
    if se == 0:
        LOGGER.warning("%s has zero empirical SE; normal inference is not reported.", method)
        result = _missing_row(method, "degenerate_inference", "Zero empirical variance; no normal CI/p-value.")
        result.update({"Estimate": float(estimate), "SE": 0.0})
        return result
    result = inference_row(method, estimate, se, description)
    result["Log_P_Value"] = float(np.log(2) + norm.logsf(abs(estimate / se)))
    return result


def estimate_target(frame: pd.DataFrame, predictions: pd.DataFrame, mask: np.ndarray, psm_reason: str) -> pd.DataFrame:
    _require(mask.dtype == bool and mask.shape == (len(frame),), "Target mask is not aligned.")
    actual_d = frame.loc[mask, D].to_numpy()
    n, n1, n0 = len(actual_d), int(actual_d.sum()), int((actual_d == 0).sum())
    estimable = min(n1, n0) >= 2
    if not estimable:
        LOGGER.warning("Target is not estimable: %d actually treated and %d controls.", n1, n0)
    X = _regression_covariates(frame.loc[mask, COVARIATES]) if estimable else None
    rows = []
    for outcome, column in OUTCOMES.items():
        if not estimable:
            estimates = [_missing_row(method, "not_estimable", "Fewer than two observations in an actual treatment arm.") for method in METHODS[:-1]]
        else:
            y = frame.loc[mask, column].to_numpy(dtype=float)
            e = predictions.loc[mask, TREATMENT_PROPENSITY].to_numpy()
            mu0 = predictions.loc[mask, f"mu0_{outcome}"].to_numpy()
            mu1 = predictions.loc[mask, f"mu1_{outcome}"].to_numpy()
            scores = treatment_scores(y, actual_d, e, mu0, mu1)
            naive, variance = difference_in_means(pd.DataFrame({"y": y, "D": actual_d}))
            estimates = [_row("Naive", naive, np.sqrt(variance), "Unequal-variance difference in means")]
            if X is None or n <= X.shape[1] + 2:
                LOGGER.warning("Regression target has too few rows for the additive covariate design.")
                estimates.append(_missing_row("Reg", "not_estimable", "Too few rows for the additive design."))
            else:
                reg, se = regression_ate(y, actual_d, X)
                estimates.append(_row("Reg", reg, se, "OLS coefficient with HC1 covariance"))
            om = outcome_modeling_ate(y, actual_d, mu0, mu1)
            om_se = np.sqrt(
                np.mean((y[actual_d == 1] - mu1[actual_d == 1])**2) / n1
                + np.mean((y[actual_d == 0] - mu0[actual_d == 0])**2) / n0
            )
            estimates.append(_row("OM", om, om_se, "Notebook residual-MSE approximation"))
            for method in ["IPW", "DR"]:
                score = scores[method]
                description = "Raw HT IPW score SD/sqrt(n)" if method == "IPW" else "Cross-fitted AIPW score SD/sqrt(n); propensity clipped to [0.01,0.99]"
                row = _row(method, float(score.mean()), float(score.std(ddof=1) / np.sqrt(n)), description)
                if method == "DR":
                    row["Unclipped_DR_Estimate"] = float(scores["DR_Unclipped"].mean())
                    row["Unclipped_DR_SE"] = float(scores["DR_Unclipped"].std(ddof=1) / np.sqrt(n))
                estimates.append(row)
        estimates.append(_missing_row("PSM", "not_run", psm_reason))
        for row in estimates:
            row.update({
                "Outcome": outcome, "N": n, "N_Treated": n1, "N_Control": n0,
                "Display_Unit": "percentage points" if outcome == "binary" else "Outcome_Continuous units",
            })
            for statistic in ["Estimate", "SE", "CI_Lower", "CI_Upper"]:
                row[f"Display_{statistic}"] = row[statistic] * (100.0 if outcome == "binary" else 1.0)
            rows.append(row)
    return pd.DataFrame(rows)


def target_diagnostics(frame: pd.DataFrame, predictions: pd.DataFrame, mask: np.ndarray) -> dict:
    actual_d = frame.loc[mask, D].to_numpy()
    e = predictions.loc[mask, TREATMENT_PROPENSITY].to_numpy()
    n1, n0 = int(actual_d.sum()), int((actual_d == 0).sum())
    result = {"N": len(e), "N_Treated": n1, "N_Control": n0, "N_Excluded": int((~mask).sum())}
    if min(n1, n0) < 2:
        result["Status"] = "not_estimable"
        return result
    weights = np.where(actual_d == 1, 1 / e, 1 / (1 - e))
    result.update({
        "Status": "estimable", "Own_Propensity_Min": float(e.min()), "Own_Propensity_Max": float(e.max()),
        "Own_Propensity_AUC": float(roc_auc_score(actual_d, e)),
        "Own_Propensity_LogLoss": float(log_loss(actual_d, e, labels=[0, 1])),
        "Own_Propensity_Brier": float(brier_score_loss(actual_d, e)),
        "DR_Clipped_Rows": int(((e < DR_MIN_PROPENSITY) | (e > 1 - DR_MIN_PROPENSITY)).sum()),
        "Max_IPW_Weight": float(weights.max()),
    })
    for arm, label in [(0, "Control"), (1, "Treated")]:
        w = weights[actual_d == arm]
        result[f"IPW_ESS_{label}"] = float(w.sum()**2 / (w @ w))
    smds, weighted_smds, separated = [], [], []
    for column in COVARIATES:
        values = frame.loc[mask, column].to_numpy(dtype=float)
        x1, x0 = values[actual_d == 1], values[actual_d == 0]
        denominator = np.sqrt((x1.var(ddof=1) + x0.var(ddof=1)) / 2)
        delta = float(x1.mean() - x0.mean())
        if denominator == 0:
            if delta != 0:
                separated.append(column)
            continue
        smds.append(abs(delta / denominator))
        weighted_smds.append(abs(
            (np.average(x1, weights=weights[actual_d == 1]) - np.average(x0, weights=weights[actual_d == 0])) / denominator
        ))
    result["Max_Abs_SMD"] = max(smds) if smds else 0.0
    result["Max_Abs_IPW_SMD"] = max(weighted_smds) if weighted_smds else 0.0
    result["Completely_Separating_Covariates"] = "; ".join(separated) if separated else "none"
    if separated:
        LOGGER.warning("Treatment arms are completely separated on: %s", separated)
    return result


def benchmark_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in REGIMES:
        for population in POPULATIONS:
            for outcome in OUTCOMES:
                selected = results.loc[
                    results["Regime"].eq(regime) & results["Population"].eq(population) & results["Outcome"].eq(outcome)
                ]
                experiment = selected.loc[selected["Cohort"].eq("experimental") & selected["Estimator"].eq("DR")].iloc[0]
                for observed in selected.loc[selected["Cohort"].eq("observed")].to_dict("records"):
                    row = {
                        "Regime": regime, "Population": population, "Outcome": outcome,
                        "Estimator": observed["Estimator"],
                        "Observed_Estimate": observed["Estimate"], "Observed_SE": observed["SE"],
                        "Experimental_DR": experiment["Estimate"], "Experimental_SE": experiment["SE"],
                        "Shared_Region_Rule": population != "legacy_overlap",
                        "Display_Unit": observed["Display_Unit"],
                    }
                    if observed["Status"] == "estimated" and experiment["Status"] == "estimated":
                        difference = observed["Estimate"] - experiment["Estimate"]
                        se = float(np.sqrt(observed["SE"]**2 + experiment["SE"]**2))
                        comparison = inference_row("difference", difference, se, "Approximate independent-cohort SE, conditional on learned region")
                        row.update({
                            "Difference": difference, "Difference_SE": se,
                            "Difference_CI_Lower": comparison["CI_Lower"], "Difference_CI_Upper": comparison["CI_Upper"],
                            "Difference_P": comparison["P_Value"], "Status": "estimated",
                            "Difference_CI_Includes_Zero": comparison["CI_Lower"] <= 0 <= comparison["CI_Upper"],
                            "Observed_Magnitude_Larger": abs(observed["Estimate"]) > abs(experiment["Estimate"]),
                        })
                    else:
                        row.update({
                            "Difference": np.nan, "Difference_SE": np.nan, "Difference_CI_Lower": np.nan,
                            "Difference_CI_Upper": np.nan, "Difference_P": np.nan, "Status": "not_estimable",
                            "Difference_CI_Includes_Zero": None, "Observed_Magnitude_Larger": None,
                        })
                    scale = 100 if outcome == "binary" else 1
                    for statistic in [
                        "Observed_Estimate", "Observed_SE", "Experimental_DR", "Experimental_SE",
                        "Difference", "Difference_SE", "Difference_CI_Lower", "Difference_CI_Upper",
                    ]:
                        row[f"Display_{statistic}"] = row[statistic] * scale
                    rows.append(row)
    return pd.DataFrame(rows)


def derive_tables(
    prepared: PreparedData, predictions: dict[str, pd.DataFrame], output: Path, config: Config, psm_reason: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    estimates, counts, thresholds = [], [], []
    projections, cache = {}, {}
    for regime in REGIMES:
        observed_key = prepared.model_map[regime]["observed"]
        experimental_key = prepared.model_map[regime]["experimental"]
        observed = predictions[observed_key]
        experimental = predictions[experimental_key]
        projection = project_adoption_propensity(
            prepared.cohorts[regime]["experimental"], experimental,
            output / "nuisances" / observed_key, config,
        )
        projections[regime] = projection
        alpha, _ = crump_rule(observed[TREATMENT_PROPENSITY].to_numpy())
        thresholds.append({"Regime": regime, "Alpha": alpha, "Estimated_From": "observational actual-treatment out-of-fold propensity"})
        masks = overlap_masks(
            observed[TREATMENT_PROPENSITY].to_numpy(), experimental[TREATMENT_PROPENSITY].to_numpy(),
            projection[ADOPTION_SCORE].to_numpy(), alpha,
        )
        for cohort in COHORTS:
            key = prepared.model_map[regime][cohort]
            frame, pred = prepared.cohorts[regime][cohort], predictions[key]
            for population, mask in masks[cohort].items():
                cache_key = (key, hashlib.sha256(mask.tobytes()).hexdigest())
                if cache_key not in cache:
                    cache[cache_key] = estimate_target(frame, pred, mask, psm_reason)
                estimates.append(cache[cache_key].assign(Regime=regime, Cohort=cohort, Population=population))
                diagnostic = target_diagnostics(frame, pred, mask)
                counts.append({"Regime": regime, "Cohort": cohort, "Population": population, "Alpha": alpha, **diagnostic})
    return pd.concat(estimates, ignore_index=True), pd.DataFrame(counts), pd.DataFrame(thresholds), projections


def _notebook_source_hash(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = [(cell["cell_type"], "".join(cell.get("source", []))) for cell in notebook["cells"]]
    return hashlib.sha256(json.dumps(source, ensure_ascii=True).encode("utf-8")).hexdigest()


def _save_code(output: Path, notebook: Path | None) -> tuple[dict[str, str], dict]:
    hashes = save_source_snapshot(output)
    for name in ["main_cutoff_validation.py", "plotting.py"]:
        source = ROOT / "src" / name
        destination = output / "source_snapshot" / "src" / name
        _require(not destination.exists(), "Refusing to overwrite archived source.")
        hashes[name] = _file_hash(source)
        shutil.copy2(source, destination)
        _require(_file_hash(destination) == hashes[name], f"{name}: source archiving failed.")
    notebook_record = {}
    if notebook is not None:
        notebook_record["source_sha256"] = _notebook_source_hash(notebook)
        relative = Path("source_snapshot") / "notebooks" / notebook.name
        (output / relative).parent.mkdir()
        content = json.loads(notebook.read_text(encoding="utf-8"))
        for cell in content["cells"]:
            if cell["cell_type"] == "code":
                cell["outputs"], cell["execution_count"] = [], None
        _write_json(output / relative, content)
        notebook_record["snapshot_path"] = str(relative)
        _require(_notebook_source_hash(output / relative) == notebook_record["source_sha256"], "Notebook source changed during archiving.")
    return hashes, notebook_record


def run_reassessment(
    data_dir: Path, output: Path, config: Config, common_upper: float = 44.0,
    notebook: Path | None = None,
) -> Path:
    config.validate()
    _require(config.continuous_upper is None, "Use common_upper for this paired analysis, not the placebo Config bound.")
    _require(np.isfinite(common_upper) and common_upper >= 0, "Invalid common cutoff.")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "nuisances").mkdir()
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    handlers = [logging.StreamHandler(), logging.FileHandler(output / "run.log", encoding="utf-8")]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    started = time.perf_counter()
    try:
        code_hashes, notebook_record = _save_code(output, notebook)
        prepared = prepare_inputs(data_dir.resolve(), common_upper)
        prepared.audit.to_csv(output / "eligibility_audit.csv", index=False)
        prepared.mean_shifts.to_csv(output / "outcome_mean_shifts.csv", index=False)
        psm = matching_availability()
        LOGGER.warning(psm["reason"])
        model_count = len(prepared.model_frames) * 5 * config.folds
        manifest = {
            "status": "started_not_yet_validated", "started_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config), "common_upper": common_upper, "dr_min_propensity": DR_MIN_PROPENSITY,
            "actual_treatment": "D = int(New_Feature); 1=exposed, 0=unexposed, within each cohort",
            "input_files": {name: {"path": str(path.resolve()), "sha256": _file_hash(path)} for name, path in prepared.paths.items()},
            "code_sha256": code_hashes, "notebook": notebook_record,
            "python": sys.version, "python_executable": sys.executable,
            "packages": {name: importlib.metadata.version(name) for name in [
                "numpy", "pandas", "scikit-learn", "flaml", "lightgbm", "statsmodels",
                "scipy", "pyarrow", "matplotlib", "seaborn",
            ]},
            "covariates": COVARIATES, "model_map": prepared.model_map,
            "expected_selected_models": model_count, "ensembling": False, "analysis_subsampling": False,
            "propensity_objective": "classification log_loss", "both_outcome_objective": "regression rmse",
            "legacy_target": "each cohort's own actual-treatment propensity inside the observational alpha interval",
            "shared_target": "observational adoption propensity evaluated on both cohorts, with one model per outer fold and the same alpha",
            "experimental_weighting": "experimental actual-treatment propensity, NEVER the observational adoption score",
            "dr_implementation": "Direct cross-fitted T-learner AIPW; shares OM nuisances; clips propensity to [0.01,0.99] like the notebook's EconML setting",
            "nuisance_reuse": "Full eligible-sample nuisances reused across X-only overlap masks; identical experimental frames reused across regimes",
            "psm": psm,
        }
        _write_json(output / "manifest.json", manifest)
        LOGGER.info("Fresh paired actual-treatment analysis: %d model sets, %d single selected models.", len(prepared.model_frames), model_count)
        predictions = {}
        for key, frame in prepared.model_frames.items():
            LOGGER.info("Fitting %s: N=%d, actual treated=%d.", key, len(frame), int(frame[D].sum()))
            predictions[key] = fit_nuisances(frame, config, output / "nuisances" / key)
        pd.concat([
            outcome_diagnostics(frame, predictions[key]).assign(Model_Set=key)
            for key, frame in prepared.model_frames.items()
        ], ignore_index=True).to_csv(output / "outcome_diagnostics.csv", index=False)
        results, counts, thresholds, projections = derive_tables(prepared, predictions, output, config, psm["reason"])
        results.to_csv(output / "main_results.csv", index=False)
        counts.to_csv(output / "target_diagnostics.csv", index=False)
        thresholds.to_csv(output / "overlap_thresholds.csv", index=False)
        for regime, projection in projections.items():
            projection.to_parquet(output / f"experimental_selection_scores_{regime}.parquet", index=False)
        comparisons = benchmark_comparisons(results)
        comparisons.to_csv(output / "benchmark_comparisons.csv", index=False)
        manifest.update({
            "status": "analysis_written_pending_validation", "selected_models_fitted": model_count,
            "analysis_seconds": time.perf_counter() - started,
        })
        _write_json(output / "manifest.json", manifest)
    finally:
        for handler in handlers:
            root_logger.removeHandler(handler)
            handler.close()
        root_logger.setLevel(previous_level)
    return output


def validate_run(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    config = Config(**manifest["config"])
    config.validate()
    for name, expected in manifest["code_sha256"].items():
        _require(_file_hash(ROOT / "src" / name) == expected, f"{name}: source changed; verify from the archived source_snapshot.")
        _require(_file_hash(output / "source_snapshot" / "src" / name) == expected, f"{name}: archived source mismatch.")
    for entry in manifest["input_files"].values():
        _require(_file_hash(Path(entry["path"])) == entry["sha256"], "Input data changed since fitting.")
    if manifest["notebook"]:
        entry = manifest["notebook"]
        _require(_notebook_source_hash(output / entry["snapshot_path"]) == entry["source_sha256"], "Archived notebook source mismatch.")
    data_dirs = {Path(entry["path"]).parent for entry in manifest["input_files"].values()}
    _require(len(data_dirs) == 1, "Input cohorts must share the recorded data directory.")
    prepared = prepare_inputs(next(iter(data_dirs)), manifest["common_upper"])
    _require(prepared.model_map == manifest["model_map"], "Model reuse does not match exact eligible-frame identity.")
    pd.testing.assert_frame_equal(prepared.audit, pd.read_csv(output / "eligibility_audit.csv"), check_dtype=False)
    pd.testing.assert_frame_equal(
        prepared.mean_shifts, pd.read_csv(output / "outcome_mean_shifts.csv"),
        check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12,
    )
    predictions, n_models, native_checks = {}, 0, 0
    for key, frame in prepared.model_frames.items():
        folder = output / "nuisances" / key
        pred = pd.read_parquet(folder / "predictions.parquet")
        validate_predictions(frame, pred)
        folds = make_folds(frame[D].to_numpy(), config)
        np.testing.assert_array_equal(pred[FOLD], validate_folds(folds, frame[D].to_numpy()))
        records = json.loads((folder / "model_manifest.json").read_text(encoding="utf-8"))
        expected_keys = {(fold, column) for fold in range(config.folds) for column in PREDICTION_COLUMNS}
        _require(
            len(records) == len(expected_keys) and {(r["fold"], r["prediction_column"]) for r in records} == expected_keys,
            "Missing or duplicate actual-treatment nuisance models.",
        )
        for record in records:
            train, test = folds[record["fold"]]
            column = record["prediction_column"]
            if column != TREATMENT_PROPENSITY:
                train = train[frame.iloc[train][D].to_numpy() == int(column[2])]
            expected_metric = "log_loss" if column == TREATMENT_PROPENSITY else "rmse"
            _require(record["metric"] == expected_metric and np.isfinite(record["best_cv_loss"]), "Invalid tuning objective or loss.")
            _require(record["n_train"] == record["final_fit_rows_verified"] == len(train), "Final fit did not use all intended training rows.")
            _require(record["n_test"] == len(test), "Incorrect held-out prediction count.")
            _require(record["train_index_sha256"] == _index_hash(train), "Training-index provenance mismatch.")
            _require(record["test_index_sha256"] == _index_hash(test), "Held-out-index provenance mismatch.")
            check_rows = test[:64]
            model = Booster(model_file=str(folder / record["model_file"]))
            replayed = model.predict(frame.iloc[check_rows][COVARIATES], num_threads=config.jobs)
            np.testing.assert_allclose(replayed, pred.loc[check_rows, column], rtol=1e-10, atol=1e-12)
            n_models += 1
            native_checks += len(check_rows)
        predictions[key] = pred
    _require(n_models == manifest["selected_models_fitted"] == manifest["expected_selected_models"], "Selected model count mismatch.")
    results, counts, thresholds, projections = derive_tables(prepared, predictions, output, config, manifest["psm"]["reason"])
    comparisons = benchmark_comparisons(results)
    outcomes = pd.concat([
        outcome_diagnostics(frame, predictions[key]).assign(Model_Set=key)
        for key, frame in prepared.model_frames.items()
    ], ignore_index=True)
    for name, expected in [
        ("main_results.csv", results), ("target_diagnostics.csv", counts),
        ("overlap_thresholds.csv", thresholds), ("benchmark_comparisons.csv", comparisons),
        ("outcome_diagnostics.csv", outcomes),
    ]:
        pd.testing.assert_frame_equal(
            expected, pd.read_csv(output / name), check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-12,
        )
    for regime, expected in projections.items():
        pd.testing.assert_frame_equal(
            expected, pd.read_parquet(output / f"experimental_selection_scores_{regime}.parquet"),
            check_exact=False, rtol=1e-10, atol=1e-12,
        )
    estimated = results.loc[results["Status"] == "estimated"]
    _require(np.isfinite(estimated[["Estimate", "SE", "CI_Lower", "CI_Upper", "P_Value"]].to_numpy()).all(), "Nonfinite inference.")
    _require(estimated["SE"].gt(0).all() and estimated["P_Value"].between(0, 1).all(), "Invalid SE or p-value.")
    validation = {
        "status": "passed_computational_checks_not_equivalence_or_unconfoundedness",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "actual_treatment_labels_and_both_arms_verified": True,
        "both_outcomes_share_eligibility": True,
        "row_provenance_input_and_source_hashes_verified": True,
        "crossfit_coverage_and_training_separation_verified": True,
        "all_selected_model_fits_verified": n_models,
        "saved_native_prediction_checks": native_checks,
        "experimental_selection_uses_observational_models_without_averaging": True,
        "experimental_weighting_uses_own_treatment_propensity": True,
        "all_targets_estimates_se_ci_pvalues_and_comparisons_recomputed": True,
        "model_reuse_requires_identical_eligible_frames": True,
    }
    _write_json(output / "validation.json", validation)
    write_report(output, manifest, results, counts, comparisons)
    manifest["status"] = "complete_and_validated"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(output / "manifest.json", manifest)
    return validation


def primary_summary(comparisons: pd.DataFrame) -> pd.DataFrame:
    return comparisons.loc[
        comparisons["Estimator"].eq("DR") & comparisons["Population"].eq("shared_overlap"),
        [
            "Regime", "Outcome", "Display_Observed_Estimate", "Display_Experimental_DR",
            "Display_Difference", "Display_Difference_SE", "Display_Difference_CI_Lower",
            "Display_Difference_CI_Upper", "Difference_P", "Display_Unit", "Status",
        ],
    ].reset_index(drop=True)


def write_report(output: Path, manifest: dict, results: pd.DataFrame, counts: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    common_map = manifest["model_map"]["common_cutoff"]
    experimental_predictions = pd.read_parquet(
        output / "nuisances" / common_map["experimental"] / "predictions.parquet",
        columns=[D, TREATMENT_PROPENSITY],
    )
    observed_predictions = pd.read_parquet(
        output / "nuisances" / common_map["observed"] / "predictions.parquet",
        columns=[D, TREATMENT_PROPENSITY],
    )
    projected = pd.read_parquet(output / "experimental_selection_scores_common_cutoff.parquet")
    alpha = float(pd.read_csv(output / "overlap_thresholds.csv").set_index("Regime").loc["common_cutoff", "Alpha"])
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, frame, e, title in [
        (axes[0], experimental_predictions, experimental_predictions[TREATMENT_PROPENSITY].to_numpy(), "Experiment: own treatment propensity"),
        (axes[1], observed_predictions, observed_predictions[TREATMENT_PROPENSITY].to_numpy(), "Observation: adoption propensity"),
        (axes[2], experimental_predictions, projected[ADOPTION_SCORE].to_numpy(), "Observational model on experiment"),
    ]:
        actual_d = frame[D].to_numpy()
        plot_propensity_distributions(
            e[actual_d == 0], e[actual_d == 1], title=title,
            trim_threshold=alpha, xlim=(0, 1), ax=ax,
        )
        ax.axvline(1 - alpha, color="gray", linestyle="--", linewidth=1)
    figure.tight_layout()
    figure.savefig(output / "propensity_common_cutoff.png", dpi=130)
    plt.close(figure)
    for outcome in OUTCOMES:
        figure, axes = plt.subplots(2, 3, figsize=(17, 9))
        for row_index, regime in enumerate(REGIMES):
            for column_index, population in enumerate(POPULATIONS):
                ax = axes[row_index, column_index]
                selected = results.loc[
                    results["Outcome"].eq(outcome) & results["Regime"].eq(regime) & results["Population"].eq(population)
                ]
                benchmark = selected.loc[selected["Cohort"].eq("experimental") & selected["Estimator"].eq("DR")].iloc[0]
                observed = selected.loc[
                    selected["Cohort"].eq("observed") & selected["Status"].eq("estimated") & ~selected["Estimator"].isin(["Naive", "PSM"])
                ].copy()
                if benchmark["Status"] != "estimated" or observed.empty:
                    ax.text(.5, .5, "No estimable benchmark/contrast\nSee target counts and status.", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"{regime}: {population}")
                    continue
                scale = 100 if outcome == "binary" else 1
                for column in ["Estimate", "SE", "CI_Lower", "CI_Upper"]:
                    observed[column] *= scale
                plot_ate_comparison(
                    observed, benchmark["Estimate"] * scale, benchmark["SE"] * scale,
                    title=f"{regime}: {population}", ax=ax,
                )
                ax.set_ylabel("Percentage points" if outcome == "binary" else "Native continuous units")
        figure.suptitle(f"Actual-treatment estimates: {outcome} (PSM not run)")
        figure.tight_layout()
        figure.savefig(output / f"main_comparison_{outcome}.png", dpi=130)
        plt.close(figure)
    primary = primary_summary(comparisons)
    audit = pd.read_csv(output / "eligibility_audit.csv")
    mean_shifts = pd.read_csv(output / "outcome_mean_shifts.csv")
    nuisance_outcomes = pd.read_csv(output / "outcome_diagnostics.csv")
    experimental = results.loc[results["Cohort"].eq("experimental") & results["Estimator"].eq("DR"), [
        "Regime", "Population", "Outcome", "N", "N_Treated", "N_Control",
        "Display_Estimate", "Display_SE", "Display_CI_Lower", "Display_CI_Upper", "Display_Unit", "Status",
    ]]
    render = lambda frame: frame.to_html(index=False, float_format=lambda value: f"{value:.6g}", na_rep="Not estimable")
    text = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Main results: common-cutoff reassessment</title>
<style>body{{font:16px system-ui,sans-serif;margin:30px auto;max-width:1500px;padding:0 20px;line-height:1.5}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;font-size:13px}}th,td{{border:1px solid #bbb;padding:6px}}
th{{background:#eef2f6}}img{{max-width:100%}}.note{{border-left:4px solid #456;padding-left:14px}}</style></head><body>
<h1>Main results with a common outcome cutoff</h1>
<p><b>This is actual-treatment analysis, not the control-source placebo.</b>
D = New_Feature, with 1 = actually exposed and 0 = unexposed, separately within each study.
The fixed common rule Outcome_Continuous &lt;= {manifest['common_upper']:g} applies to both treatment
arms in both studies and to both outcomes. The as-released analysis is refitted under identical settings.</p>
<p class="note">Both released files were already source-specifically outcome-truncated.
No new 97th percentile is estimated. Outcome-based eligibility can itself compromise randomization
and change the estimand. These are robustness checks of selected-data estimates, not proof about
the untruncated causal effect. Pre-truncation data are unavailable here.</p>
<h2>Primary: shared-overlap DR comparison</h2>
<p>Difference = observational ATE minus experimental DR benchmark. Binary display units are percentage
points; continuous units are unchanged. Difference CIs assume independent study samples, conditional
on the fitted region. They are approximate, pointwise, and do not establish equivalence.</p>
<div class="scroll">{render(primary)}</div>
<h2>What was actually rerun</h2>
<ul><li>{manifest['selected_models_fitted']} newly selected LightGBM nuisance models:
{manifest['config']['folds']}-fold outer cross-fitting, {manifest['config']['inner_folds']}-fold inner FLAML CV.
No ensembles, analysis subsampling, old model weights, old hyperparameters or old result caches.</li>
<li>Classification log loss for actual-treatment propensities; regression RMSE for both outcomes.
Budget {manifest['config']['time_budget']:g}s/search, at most {manifest['config']['max_iter']} trials,
plus final refits. Analysis time: {manifest['analysis_seconds']/60:.2f} minutes.</li>
<li>Experimental nuisance predictions are reused between regimes only if the entire eligible
DataFrames are identical. Full eligible-sample nuisances are reused across X-only overlap targets.</li>
<li>Reg uses OLS/HC1; OM uses separate treatment-arm models and the notebook residual-MSE approximation;
IPW is raw unnormalized HT. DR is direct T-learner AIPW using the same nuisances, with propensity
clipping to [0.01,0.99], matching the original notebook's EconML clipping setting.
Unclipped DR estimates are retained in the results CSV.</li>
<li>This is a controlled single-model reassessment, not an exact numerical replication of the
paper's ensembles or the original notebook's separate pooled-outcome EconML refit.</li>
<li>{manifest['psm']['reason']} No stale PSM estimates or invented matching SEs are substituted.</li></ul>
<h2>Why both overlap rules are reported</h2>
<p><b>Legacy:</b> each cohort's own actual-treatment propensity is compared with the observational
Crump alpha. This reproduces the notebook's selection logic but does not define a shared X-region;
its cross-study difference is descriptive, not a clean estimator-bias test.</p>
<p><b>Shared:</b> the observational adoption model is evaluated on both cohorts, using one observational
model per held-out fold without averaging, and the same alpha interval selects both samples.
The experimental AIPW/IPW weights still use the experiment's <b>own actual-treatment propensity</b>,
never the observational adoption score. The experimental benchmark is newly computed, not loaded.</p>
<h2>Actual treatment-arm counts and eligibility</h2><div class="scroll">{render(audit)}</div>
<h3>Outcome-based exclusion contributions</h3>
<p>For each actual treatment arm: released mean = common-bound mean + excluded fraction *
(excluded mean - common-bound mean). Subtracting the D=0 shift from the D=1 shift gives the
mechanical change in that cohort's naive treatment contrast; it is not itself a causal effect.</p>
<div class="scroll">{render(mean_shifts)}</div>
<h2>Fresh experimental benchmarks</h2><div class="scroll">{render(experimental)}</div>
<h2>Common-cutoff propensity distributions</h2>
<p>Gray lines use the observational alpha. The first and third panels illustrate why own experimental
assignment probabilities and transported observational adoption scores are not interchangeable.</p>
<img src="propensity_common_cutoff.png" alt="Own and projected treatment propensity distributions">
<h2>Continuous results</h2><img src="main_comparison_continuous.png" alt="Continuous effect comparisons">
<h2>Binary results</h2><img src="main_comparison_binary.png" alt="Binary effect comparisons">
<h2>Target counts, overlap and weighting diagnostics</h2><div class="scroll">{render(counts)}</div>
<h2>Held-out outcome-model diagnostics</h2><div class="scroll">{render(nuisance_outcomes)}</div>
<h2>Interpretation limits</h2><ul>
<li>Non-rejection of observational/experimental agreement is not equivalence or proof of unconfoundedness.</li>
<li>Intervals do not incorporate repeated tuning/split variability or uncertainty in learned trimming boundaries.
OM/IPW additionally inherit non-orthogonal nuisance-estimation limitations.</li>
<li>The same region rule need not give exactly identical covariate distributions across studies under heterogeneity.
Public row positions do not identify devices; cross-study overlap or clustering cannot be ruled out.</li>
<li>The two cutoff regimes share observations. Their estimates must not be treated as independent when testing
a change between regimes. No such independent-sample change test is reported.</li>
<li>Binary outcome models are RMSE regressors, not calibrated classifiers; outcome predictions are not clipped.
Very small p-values can underflow to zero; log p-values are preserved in the results CSV.</li></ul>
<h2>Reproducibility</h2><p>
<a href="main_results.csv">All estimates and uncertainty</a> |
<a href="benchmark_comparisons.csv">All benchmark comparisons</a> |
<a href="overlap_thresholds.csv">New thresholds</a> |
<a href="outcome_diagnostics.csv">Held-out nuisance performance</a> |
<a href="manifest.json">Manifest and settings</a> |
<a href="validation.json">Executed checks</a></p>
<p>Input hashes, raw source-row positions, actual exposure, both outcomes, fold IDs, full held-out
predictions, selected model configurations and native models are saved. Code and notebook source are
archived under source_snapshot. Verification recomputes selections, projections, estimates, SEs,
CIs and p-values without model refitting.</p></body></html>"""
    (output / "main_cutoff_report.html").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--continuous-upper", type=float, default=44.0)
    parser.add_argument("--time-budget", type=float, default=30.0)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify-output", type=Path)
    args = parser.parse_args()
    if args.verify_output is not None:
        output = args.verify_output.resolve()
    else:
        output = args.output_dir or ROOT / "saved_outputs_main_common_cutoff" / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ")
        config = Config(folds=args.folds, time_budget=args.time_budget, max_iter=args.max_iter, seed=args.seed, jobs=args.jobs)
        output = run_reassessment(args.data_dir, output, config, args.continuous_upper)
    validation = validate_run(output)
    print(json.dumps(validation, indent=2))
    print(primary_summary(pd.read_csv(output / "benchmark_comparisons.csv")).to_string(index=False))
    print(f"Validated report: {output / 'main_cutoff_report.html'}")


if __name__ == "__main__":
    main()
