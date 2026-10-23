"""Actual-treatment, common-cutoff, and shared-overlap regression tests."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from IPython.display import Image, Markdown, display
import numpy as np
import pandas as pd

from src.data_loading import COVARIATES, OUTCOME_BINARY, OUTCOME_CONTINUOUS, TREATMENT
from src.main_cutoff_validation import (
    ADOPTION_SCORE, D, DR_MIN_PROPENSITY, IDENTITY_COLUMNS, TREATMENT_PROPENSITY,
    Config, benchmark_comparisons, estimate_target, overlap_masks, prepare_cohort,
    prepare_inputs, primary_summary, run_reassessment, treatment_scores, validate_run,
)
from src.placebo_validation import SOURCE_ROW


def raw_cohort(n=240, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(n, len(COVARIATES))), columns=COVARIATES)
    treatment = np.arange(n) % 4 == 0
    frame[TREATMENT] = treatment
    frame[OUTCOME_CONTINUOUS] = rng.poisson(np.exp(.4 + .25 * treatment + .15 * frame[COVARIATES[0]]))
    frame[OUTCOME_BINARY] = rng.binomial(1, .8 - .2 * treatment)
    return frame


class ActualTreatmentConstructionTests(unittest.TestCase):
    def test_common_bound_filters_both_actual_arms_and_keeps_boundary(self):
        raw = raw_cohort(12)
        raw.loc[0, OUTCOME_CONTINUOUS] = 45
        raw.loc[1, OUTCOME_CONTINUOUS] = 46
        raw.loc[4, OUTCOME_CONTINUOUS] = 44
        frame = prepare_cohort(raw, 44)
        self.assertEqual(len(frame), 10)
        self.assertNotIn(0, frame[SOURCE_ROW].tolist())
        self.assertNotIn(1, frame[SOURCE_ROW].tolist())
        self.assertIn(4, frame[SOURCE_ROW].tolist())
        self.assertEqual(int(frame[D].sum()), 2)
        np.testing.assert_array_equal(frame[D], frame[TREATMENT].astype(int))
        self.assertNotIn("placebo_treated", frame)
        self.assertEqual(frame.loc[frame[SOURCE_ROW] == 4, OUTCOME_BINARY].iloc[0], raw.loc[4, OUTCOME_BINARY])

    def test_actual_labels_missing_values_and_empty_arms_fail(self):
        invalid = raw_cohort()
        invalid[TREATMENT] = 1
        with self.assertRaises(ValueError):
            prepare_cohort(invalid, 44)
        invalid = raw_cohort()
        invalid.loc[0, COVARIATES[0]] = np.inf
        with self.assertRaises(ValueError):
            prepare_cohort(invalid, 44)
        invalid = raw_cohort()
        invalid.loc[invalid[TREATMENT], OUTCOME_CONTINUOUS] = 45
        with self.assertRaises(ValueError):
            prepare_cohort(invalid, 44)

    def test_identical_experiment_is_reused_but_changed_observation_is_not(self):
        with tempfile.TemporaryDirectory(prefix="main_cutoff_input_test_") as directory:
            folder = Path(directory)
            experiment, observed = raw_cohort(), raw_cohort(seed=2)
            observed.loc[[0, 1], OUTCOME_CONTINUOUS] = [45, 46]
            experiment.to_parquet(folder / "FINAL_PUBLIC_experimental.parquet", index=False)
            observed.to_parquet(folder / "FINAL_PUBLIC_observed.parquet", index=False)
            prepared = prepare_inputs(folder, 44)
            self.assertEqual(len(prepared.model_frames), 3)
            self.assertEqual(prepared.model_map["as_released"]["experimental"], prepared.model_map["common_cutoff"]["experimental"])
            self.assertNotEqual(prepared.model_map["as_released"]["observed"], prepared.model_map["common_cutoff"]["observed"])
            changed = prepared.audit.loc[prepared.audit.Regime.eq("common_cutoff") & prepared.audit.Cohort.eq("observed")].iloc[0]
            self.assertEqual(changed.Dropped_Control, 1)
            self.assertEqual(changed.Dropped_Treated, 1)
            for row in prepared.mean_shifts.to_dict("records"):
                self.assertAlmostEqual(row["Released_Mean"], row["Common_Mean"] + row["Released_Minus_Common"])


class OverlapAndInferenceTests(unittest.TestCase):
    def test_shared_selection_does_not_use_experimental_assignment_propensity(self):
        observed = np.array([.01, .2, .5, .99])
        experimental = np.full(4, .025)
        adoption = np.array([.02, .3, .6, .98])
        masks = overlap_masks(observed, experimental, adoption, .1)
        np.testing.assert_array_equal(masks["observed"]["legacy_overlap"], [False, True, True, False])
        np.testing.assert_array_equal(masks["observed"]["shared_overlap"], masks["observed"]["legacy_overlap"])
        self.assertFalse(masks["experimental"]["legacy_overlap"].any())
        np.testing.assert_array_equal(masks["experimental"]["shared_overlap"], [False, True, True, False])

    def test_experimental_weighting_uses_own_propensity_not_selection_score(self):
        treatment = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        y = 2.0 * treatment
        own_propensity = np.full(8, .25)
        selection_score = np.full(8, .5)
        zeros = np.zeros(8)
        correct = treatment_scores(y, treatment, own_propensity, zeros, zeros)["DR"]
        incorrect = treatment_scores(y, treatment, selection_score, zeros, zeros)["DR"]
        self.assertAlmostEqual(correct.mean(), 2.)
        self.assertAlmostEqual(incorrect.mean(), 1.)

    def test_notebook_propensity_clipping_is_explicit_and_ipw_stays_raw(self):
        treatment = np.array([0, 0, 1, 1])
        y = np.array([2., 3., 4., 5.])
        e = np.array([.001, .999, .001, .999])
        mu0, mu1 = np.ones(4), np.full(4, 2.)
        scores = treatment_scores(y, treatment, e, mu0, mu1)
        clipped = np.clip(e, DR_MIN_PROPENSITY, 1 - DR_MIN_PROPENSITY)
        expected_dr = mu1 - mu0 + treatment * (y - mu1) / clipped - (1 - treatment) * (y - mu0) / (1 - clipped)
        expected_ipw = treatment * y / e - (1 - treatment) * y / (1 - e)
        np.testing.assert_allclose(scores["DR"], expected_dr)
        np.testing.assert_allclose(scores["IPW"], expected_ipw)
        self.assertFalse(np.allclose(scores["DR"], scores["DR_Unclipped"]))

    def test_empty_legacy_target_is_explicitly_not_estimable(self):
        frame = prepare_cohort(raw_cohort(), None)
        predictions = frame[IDENTITY_COLUMNS].copy()
        for column in [TREATMENT_PROPENSITY, "mu0_continuous", "mu1_continuous", "mu0_binary", "mu1_binary"]:
            predictions[column] = .5
        with self.assertLogs("src.main_cutoff_validation", level="WARNING"):
            results = estimate_target(frame, predictions, np.zeros(len(frame), dtype=bool), "PSM unavailable")
        self.assertTrue(results.loc[results.Estimator.ne("PSM"), "Status"].eq("not_estimable").all())
        self.assertTrue(results["Estimate"].isna().all())

    def test_known_benchmark_difference_and_units(self):
        rows = []
        for regime in ["as_released", "common_cutoff"]:
            for population in ["full", "legacy_overlap", "shared_overlap"]:
                for outcome in ["continuous", "binary"]:
                    for cohort, estimate, se in [("experimental", .2, .03), ("observed", .25, .04)]:
                        rows.append({
                            "Regime": regime, "Population": population, "Outcome": outcome,
                            "Cohort": cohort, "Estimator": "DR", "Estimate": estimate,
                            "SE": se, "Status": "estimated",
                            "Display_Unit": "percentage points" if outcome == "binary" else "native",
                        })
        comparisons = benchmark_comparisons(pd.DataFrame(rows))
        np.testing.assert_allclose(comparisons["Difference"], .05)
        np.testing.assert_allclose(comparisons["Difference_SE"], .05)
        np.testing.assert_allclose(comparisons.loc[comparisons.Outcome.eq("binary"), "Display_Difference"], 5.)
        self.assertFalse(comparisons.loc[comparisons.Population.eq("legacy_overlap"), "Shared_Region_Rule"].any())


class NotebookBackendIntegrationTests(unittest.TestCase):
    def test_real_models_full_roundtrip_and_no_old_cache_inputs(self):
        with tempfile.TemporaryDirectory(prefix="main_cutoff_integration_") as directory:
            folder = Path(directory)
            data = folder / "Data"
            data.mkdir()
            experiment, observed = raw_cohort(240, 1), raw_cohort(280, 3)
            observed.loc[[0, 1], OUTCOME_CONTINUOUS] = [45, 46]
            experiment.to_parquet(data / "FINAL_PUBLIC_experimental.parquet", index=False)
            observed.to_parquet(data / "FINAL_PUBLIC_observed.parquet", index=False)
            output = folder / "run"
            notebook = folder / "fixture.ipynb"
            notebook.write_text(json.dumps({
                "cells": [{
                    "cell_type": "code", "source": ["print('fixture')\n"], "execution_count": 7,
                    "outputs": [{"output_type": "stream", "name": "stdout", "text": ["stale result\n"]}],
                    "metadata": {},
                }],
                "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
            }), encoding="utf-8")
            config = Config(folds=2, inner_folds=2, time_budget=10, max_iter=2, jobs=2)
            with patch("src.main_cutoff_validation.matching_availability", return_value={"status": "unavailable", "reason": "PSM unavailable in integration fixture"}):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    run_reassessment(data, output, config, 44, notebook=notebook)
                    validation = validate_run(output)
            self.assertEqual(validation["all_selected_model_fits_verified"], 30)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete_and_validated")
            self.assertFalse(manifest["ensembling"])
            archived_notebook = json.loads((output / manifest["notebook"]["snapshot_path"]).read_text())
            self.assertEqual(archived_notebook["cells"][0]["outputs"], [])
            self.assertIsNone(archived_notebook["cells"][0]["execution_count"])
            self.assertTrue((output / "main_cutoff_report.html").is_file())
            self.assertTrue((output / "main_comparison_continuous.png").is_file())
            self.assertTrue((output / "propensity_common_cutoff.png").is_file())
            results = pd.read_csv(output / "main_results.csv")
            self.assertEqual(len(results), 144)
            self.assertEqual(set(results.Outcome), {"continuous", "binary"})
            self.assertFalse(results.duplicated(["Regime", "Cohort", "Population", "Outcome", "Estimator"]).any())
            actual = results.loc[
                results.Regime.eq("common_cutoff") & results.Cohort.eq("observed") & results.Population.eq("full")
            ]
            self.assertTrue(actual.N.eq(278).all())
            projection = pd.read_parquet(output / "experimental_selection_scores_common_cutoff.parquet")
            self.assertEqual(len(projection), 240)
            self.assertIn(ADOPTION_SCORE, projection)
            notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "01_main_results.ipynb"
            content = json.loads(notebook_path.read_text(encoding="utf-8"))
            namespace = {
                "OUT_DIR": output, "pd": pd, "display": display, "Image": Image,
                "Markdown": Markdown, "primary_summary": primary_summary,
            }
            post_fit = {
                "read-results", "all-benchmarks", "all-observational-methods",
                "overlap-diagnostics", "propensity-figure", "continuous-figure", "binary-figure", "output-links",
            }
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                for cell in content["cells"]:
                    if cell["cell_type"] == "code":
                        code = compile("".join(cell["source"]), f"notebook:{cell['id']}", "exec")
                        if cell["id"] in post_fit:
                            exec(code, namespace)


if __name__ == "__main__":
    unittest.main()
