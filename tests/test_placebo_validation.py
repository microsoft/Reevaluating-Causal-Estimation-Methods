"""Focused construction, cross-fitting, estimator, and FLAML integration checks."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data_loading import COVARIATES, OUTCOME_BINARY, OUTCOME_CONTINUOUS, TREATMENT
from src.estimators import regression_ate
from src.placebo_validation import (
    Config, FOLD, KEEP, OUTCOMES, PLACEBO, PROPENSITY, SOURCE, SOURCE_ROW, TREATMENT_PROPENSITY,
    SOURCE_FILES, _file_hash, _fit_single, _regression_covariates,
    apply_common_upper_bound, build_placebo_sample, common_bound_diagnostics, crump_rule,
    cross_fit_nuisances, estimate_results, inference_row, make_folds, score_arrays,
    save_source_snapshot, validate_folds,
)


def source_frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(n, len(COVARIATES))), columns=COVARIATES)
    frame[TREATMENT] = False
    frame.loc[n - 1, TREATMENT] = True
    frame[OUTCOME_CONTINUOUS] = rng.normal(size=n) + 3
    frame[OUTCOME_BINARY] = np.arange(n) % 2
    return frame


class ConstructionTests(unittest.TestCase):
    def test_only_unexposed_and_provenance_and_sign(self):
        exp, obs = source_frame(7), source_frame(9, 1)
        exp.index = np.arange(100, 107)
        obs.index = np.arange(200, 209)
        exp.loc[106, OUTCOME_CONTINUOUS] = np.nan
        sample, audit = build_placebo_sample(exp, obs)
        self.assertEqual(len(sample), 14)
        self.assertFalse(sample[TREATMENT].any())
        self.assertEqual(audit["excluded_actual_exposure"].tolist(), [1, 1])
        self.assertEqual(audit["unexposed_rows"].tolist(), [6, 8])
        np.testing.assert_array_equal(sample[PLACEBO], [1] * 6 + [0] * 8)
        np.testing.assert_array_equal(sample[SOURCE_ROW], [*range(6), *range(8)])
        self.assertEqual(sample.loc[sample[PLACEBO] == 1, SOURCE].unique().tolist(), ["experimental"])
        self.assertEqual(len(sample[COVARIATES].columns), 41)
        self.assertFalse(set(COVARIATES) & {TREATMENT, PLACEBO, SOURCE, SOURCE_ROW, FOLD, KEEP, PROPENSITY, *OUTCOMES.values()})

    def test_invalid_control_values_fail(self):
        for column, value in [(COVARIATES[0], np.inf), (OUTCOME_CONTINUOUS, np.nan), (OUTCOME_BINARY, 2)]:
            with self.subTest(column=column):
                exp = source_frame()
                exp.loc[0, column] = value
                with self.assertRaises(ValueError):
                    build_placebo_sample(exp, source_frame(seed=1))

    def test_missing_exposure_or_wrong_schema_fail(self):
        exp = source_frame()
        exp[TREATMENT] = exp[TREATMENT].astype("boolean")
        exp.loc[0, TREATMENT] = pd.NA
        with self.assertRaises(ValueError):
            build_placebo_sample(exp, source_frame())
        with self.assertRaises(ValueError):
            build_placebo_sample(source_frame().assign(source_indicator=1), source_frame())

    def test_empty_source_controls_fail(self):
        exp = source_frame()
        exp[TREATMENT] = True
        with self.assertRaises(ValueError):
            build_placebo_sample(exp, source_frame())


class FoldAndTrimmingTests(unittest.TestCase):
    def setUp(self):
        self.source = np.array([1] * 30 + [0] * 36)
        self.config = Config()

    def test_complete_disjoint_reproducible_source_only_folds(self):
        folds = make_folds(self.source, self.config)
        assignment = validate_folds(folds, self.source)
        self.assertEqual(set(assignment), {0, 1, 2})
        for (train, test), (_, repeat_test) in zip(folds, make_folds(self.source, self.config)):
            self.assertEqual(np.intersect1d(train, test).size, 0)
            np.testing.assert_array_equal(test, repeat_test)
            self.assertEqual(int(self.source[test].sum()), 10)

    def test_leakage_duplicate_coverage_and_bad_indices_fail(self):
        folds = make_folds(self.source, self.config)
        train, test = folds[0]
        leaking = [(np.concatenate([train[1:], test[:1]]), test), *folds[1:]]
        duplicated = [folds[0], folds[0], folds[2]]
        outside = [(train, np.concatenate([test[:-1], [len(self.source)]])), *folds[1:]]
        for invalid in [leaking, duplicated, outside]:
            with self.assertRaises(ValueError):
                validate_folds(invalid, self.source)

    def test_crump_known_no_trim_and_tail_exclusion(self):
        alpha, keep = crump_rule(np.full(10, .5))
        self.assertEqual(alpha, 0)
        self.assertTrue(keep.all())
        alpha, keep = crump_rule(np.array([.01, .2, .5, .8, .99]))
        self.assertAlmostEqual(alpha, .5 - np.sqrt(.25 - 1 / 11), places=10)
        np.testing.assert_array_equal(keep, [False, True, True, True, False])

    def test_nonfinite_and_boundary_propensities_fail(self):
        for e in [np.array([0, .5]), np.array([1, .5]), np.array([np.nan, .5]), np.array([np.inf, .5])]:
            with self.assertRaises(ValueError):
                crump_rule(e)

    def test_crossfit_dispatch_has_no_heldout_or_other_source_training(self):
        sample, _ = build_placebo_sample(source_frame(40), source_frame(44, 1))
        config = Config(folds=2)
        calls = []

        def fit(X, y, train, test, column, fold, cfg, output):
            self.assertEqual(list(X.columns), COVARIATES)
            self.assertEqual(np.intersect1d(train, test).size, 0)
            if column != PROPENSITY:
                expected_group = int(column[2])
                self.assertTrue((sample.iloc[train][PLACEBO] == expected_group).all())
            values = np.full(len(test), np.mean(y[train]))
            calls.append((column, fold))
            return values, {"column": column, "fold": fold}

        with tempfile.TemporaryDirectory(prefix="placebo_test_") as directory:
            with patch("src.placebo_validation._fit_single", side_effect=fit):
                predictions, records, alpha = cross_fit_nuisances(sample, config, Path(directory))
        self.assertEqual(len(records), 10)
        self.assertEqual(len(set(calls)), 10)
        self.assertEqual(alpha, 0)
        self.assertTrue(predictions[KEEP].all())
        self.assertFalse(predictions.isna().any().any())


class CommonBoundTests(unittest.TestCase):
    def setUp(self):
        exp, obs = source_frame(7), source_frame(7, 2)
        exp[OUTCOME_CONTINUOUS] = [0., 1., 2., 3., 43., 44., 1000.]
        obs[OUTCOME_CONTINUOUS] = [0., 1., 2., 44., 45., 46., 2000.]
        self.sample, self.audit = build_placebo_sample(exp, obs)

    def test_same_fixed_rule_keeps_boundary_and_original_source_positions(self):
        selected, audit = apply_common_upper_bound(self.sample, self.audit, 44)
        self.assertEqual(len(selected), 10)
        self.assertFalse(selected[TREATMENT].any())
        self.assertTrue(selected[OUTCOME_CONTINUOUS].le(44).all())
        self.assertEqual(int(selected[OUTCOME_CONTINUOUS].eq(44).sum()), 2)
        self.assertEqual(audit["excluded_actual_exposure"].tolist(), [1, 1])
        self.assertEqual(audit["unexposed_rows"].tolist(), [6, 6])
        self.assertEqual(audit["retained_unexposed_rows"].tolist(), [6, 4])
        self.assertEqual(audit["excluded_common_outcome_bound"].tolist(), [0, 2])
        np.testing.assert_array_equal(selected.loc[selected[PLACEBO] == 0, SOURCE_ROW], [0, 1, 2, 3])
        pd.testing.assert_frame_equal(
            selected[[SOURCE, SOURCE_ROW, OUTCOME_BINARY]],
            self.sample.loc[self.sample[OUTCOME_CONTINUOUS] <= 44, [SOURCE, SOURCE_ROW, OUTCOME_BINARY]].reset_index(drop=True),
        )
        self.assertEqual(self.audit.columns.tolist(), [
            SOURCE, PLACEBO, "original_rows", "excluded_actual_exposure",
            "unexposed_rows", "excluded_missing_or_nonfinite",
        ])

    def test_none_preserves_original_run_and_binary_cannot_select_rows(self):
        selected, audit = apply_common_upper_bound(self.sample, self.audit, None)
        pd.testing.assert_frame_equal(selected, self.sample)
        pd.testing.assert_frame_equal(audit, self.audit)
        changed = self.sample.copy()
        changed[OUTCOME_BINARY] = 1 - changed[OUTCOME_BINARY]
        original, _ = apply_common_upper_bound(self.sample, self.audit, 44)
        flipped, _ = apply_common_upper_bound(changed, self.audit, 44)
        pd.testing.assert_frame_equal(original[[SOURCE, SOURCE_ROW]], flipped[[SOURCE, SOURCE_ROW]])

    def test_invalid_bounds_and_loss_of_source_fail(self):
        for upper in [-1., np.nan, np.inf]:
            with self.assertRaises(ValueError):
                Config(continuous_upper=upper).validate()
            with self.assertRaises(ValueError):
                apply_common_upper_bound(self.sample, self.audit, upper)
        with self.assertRaises(ValueError):
            apply_common_upper_bound(self.sample, self.audit, 0)

    def test_exact_mean_decomposition_and_diagnostic_counts(self):
        contrasts, decomposition, excluded = common_bound_diagnostics(self.sample, 44)
        self.assertEqual(len(contrasts), 4)
        self.assertEqual(len(decomposition), 4)
        self.assertEqual(len(excluded), 2)
        self.assertEqual(excluded[SOURCE].unique().tolist(), ["observed"])
        np.testing.assert_array_equal(excluded[OUTCOME_CONTINUOUS], [45, 46])
        for row in decomposition.to_dict("records"):
            self.assertAlmostEqual(row["Released_Mean"], row["Retained_Mean"] + row["Released_Minus_Retained"])
            self.assertEqual(row["N_Released_Controls"], row["N_Retained"] + row["N_Excluded"])
        cont = contrasts.loc[contrasts["Outcome"] == "continuous"].set_index("Population")
        parts = decomposition.loc[decomposition["Outcome"] == "continuous"].set_index(SOURCE)
        self.assertAlmostEqual(
            cont.loc["as_released_controls", "Estimate"] - cont.loc["common_bound_controls", "Estimate"],
            parts.loc["experimental", "Released_Minus_Retained"] - parts.loc["observed", "Released_Minus_Retained"],
        )

    def test_snapshot_is_exact_and_cannot_overwrite_a_run(self):
        with tempfile.TemporaryDirectory(prefix="placebo_snapshot_test_") as directory:
            output = Path(directory)
            hashes = save_source_snapshot(output)
            self.assertEqual(set(hashes), set(SOURCE_FILES))
            for name, expected in hashes.items():
                self.assertEqual(_file_hash(output / "source_snapshot" / "src" / name), expected)
            with self.assertRaises(FileExistsError):
                save_source_snapshot(output)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([2., 4., 10., 12.])
        self.source = np.array([0, 0, 1, 1])
        self.e = np.full(4, .5)
        self.mu0 = np.ones(4)
        self.mu1 = np.full(4, 3.)

    def test_known_scores_and_source_reversal(self):
        scores = score_arrays(self.y, self.source, self.e, self.mu0, self.mu1)
        np.testing.assert_array_equal(scores["IPW"], [-4, -8, 20, 24])
        np.testing.assert_array_equal(scores["DR"], [0, -4, 16, 20])
        reversed_scores = score_arrays(self.y, 1 - self.source, 1 - self.e, self.mu1, self.mu0)
        for name in scores:
            self.assertEqual(scores[name].mean(), 8)
            np.testing.assert_array_equal(reversed_scores[name], -scores[name])

    def test_normal_interval_se_and_two_sided_p(self):
        row = inference_row("known", 1., .5, "known SE")
        self.assertAlmostEqual(row["P_Value"], .0455002638963584)
        self.assertAlmostEqual(row["CI_Lower"], 1 - norm.ppf(.975) * .5)
        self.assertAlmostEqual(row["CI_Upper"], 1 + norm.ppf(.975) * .5)
        score = score_arrays(self.y, self.source, self.e, self.mu0, self.mu1)["DR"]
        row = inference_row("DR", score.mean(), score.std(ddof=1) / 2, "score")
        self.assertAlmostEqual(row["SE"], np.sqrt(416 / 3) / 2)

    def test_invalid_estimates_vectors_and_labels_fail(self):
        for estimate, se in [(np.nan, 1), (0, 0), (1, -1), (1, np.inf)]:
            with self.assertRaises(ValueError):
                inference_row("bad", estimate, se, "bad")
        for source, e in [(np.array([0, 0, 1, 2]), self.e), (self.source, self.e[:-1]), (self.source, np.zeros(4))]:
            with self.assertRaises(ValueError):
                score_arrays(self.y, source, e, self.mu0, self.mu1)

    def test_synthetic_null_nonnull_and_double_robustness_both_outcomes(self):
        rng = np.random.default_rng(812)
        x = rng.normal(size=30000)
        e = 1 / (1 + np.exp(-.7 * x))
        source = rng.binomial(1, e)
        for outcome, effect in [("continuous", 0.), ("continuous", 1.), ("binary", 0.), ("binary", .04)]:
            mu0 = 2 + x if outcome == "continuous" else .3 + .1 * np.tanh(x)
            mu1 = mu0 + effect
            mean = np.where(source == 1, mu1, mu0)
            y = mean + rng.normal(size=len(x)) if outcome == "continuous" else rng.binomial(1, mean)
            for ps, m0, m1 in [
                (e, mu0, mu1),
                (np.full(len(x), .5), mu0, mu1),
                (e, np.zeros(len(x)), np.zeros(len(x))),
            ]:
                score = score_arrays(y, source, ps, m0, m1)["DR"]
                se = score.std(ddof=1) / np.sqrt(len(x))
                with self.subTest(outcome=outcome, effect=effect):
                    self.assertLess(abs(score.mean() - effect), 4 * se)

    def test_regression_constant_covariate_cannot_shift_source_coefficient(self):
        rng = np.random.default_rng(31)
        source = np.tile([0, 1], 100)
        x = rng.normal(size=len(source))
        noise = rng.normal(size=len(source))
        y = 3 * source + 2 * x + noise
        covariates = pd.DataFrame({"constant_one": np.ones(len(x)), "varying": x})
        actual = regression_ate(y, source, _regression_covariates(covariates))
        expected = regression_ate(y, source, x[:, None])
        np.testing.assert_allclose(actual, expected, rtol=1e-11)

    def test_full_crump_identity_and_binary_percentage_point_units(self):
        sample, _ = build_placebo_sample(source_frame(70), source_frame(80, 2))
        predictions = sample.copy()
        predictions[PROPENSITY] = sample[PLACEBO].mean()
        predictions[KEEP] = True
        for name, column in OUTCOMES.items():
            for group in [0, 1]:
                predictions[f"mu{group}_{name}"] = sample.loc[sample[PLACEBO] == group, column].mean()
        results = estimate_results(sample, predictions, "PSM infrastructure unavailable in this test")
        full = results.loc[results["Population"] == "full"].reset_index(drop=True)
        trimmed = results.loc[results["Population"] == "crump"].reset_index(drop=True)
        pd.testing.assert_frame_equal(full.drop(columns="Population"), trimmed.drop(columns="Population"))
        selected = results.loc[(results["Outcome"] == "binary") & (results["Status"] == "estimated")]
        for statistic in ["Estimate", "SE", "CI_Lower", "CI_Upper"]:
            np.testing.assert_allclose(selected[f"Display_{statistic}"], 100 * selected[statistic])
        self.assertTrue(results.loc[results["Estimator"] == "PSM", "SE"].isna().all())


class FLAMLIntegrationTests(unittest.TestCase):
    def test_single_fitted_native_models_and_actual_objectives(self):
        rng = np.random.default_rng(918)
        X = pd.DataFrame(rng.normal(size=(240, 4)), columns=["a", "b", "c", "d"])
        source = np.tile([0, 1], 120)
        y = 2 * X["a"].to_numpy() + rng.normal(size=len(X))
        train, test = make_folds(source, Config())[0]
        config = Config(time_budget=10, max_iter=2, jobs=2)
        with tempfile.TemporaryDirectory(prefix="placebo_flaml_test_") as directory:
            output = Path(directory)
            (output / "models").mkdir()
            (output / "flaml_logs").mkdir()
            for column, target in [(PROPENSITY, source), (TREATMENT_PROPENSITY, source), ("mu0_continuous", y), ("mu0_binary", (y > 0).astype(int))]:
                prediction, record = _fit_single(X, target, train, test, column, 0, config, output)
                self.assertEqual(prediction.shape, test.shape)
                self.assertTrue(np.isfinite(prediction).all())
                self.assertTrue(np.isfinite(record["best_cv_loss"]))
                self.assertEqual(record["metric"], "log_loss" if column in {PROPENSITY, TREATMENT_PROPENSITY} else "rmse")
                self.assertEqual(record["final_fit_rows_verified"], len(train))
                self.assertTrue((output / record["model_file"]).is_file())
                if column in {PROPENSITY, TREATMENT_PROPENSITY}:
                    self.assertTrue(((prediction > 0) & (prediction < 1)).all())


if __name__ == "__main__":
    unittest.main()
