"""
ATE estimators: Regression, Outcome Modeling, IPW, PSM, and Doubly Robust (AIPW).

All estimators operate on a trimmed sample with pre-computed nuisance functions
(propensity scores and/or outcome predictions).  Bootstrap standard errors
are provided via :func:`bootstrap_ate`.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from flaml import AutoML
from lightgbm import LGBMRegressor
import statsmodels.api as sm
from typing import Dict, Optional, Tuple


# ======================================================================
# Averaging regressor ensemble (Breiman 1996)
# ======================================================================
class AveragingRegressor(BaseEstimator, RegressorMixin):
    """Ensemble that averages predictions from multiple base regressors."""

    def __init__(self, estimators=None):
        self.estimators = estimators if estimators is not None else []

    def fit(self, X, y, sample_weight=None):
        self.fitted_estimators_ = [clone(est).fit(X, y) for est in self.estimators]
        return self

    def predict(self, X):
        if not hasattr(self, "fitted_estimators_"):
            raise NotFittedError("AveragingRegressor is not fitted yet.")
        preds = np.column_stack([est.predict(X) for est in self.fitted_estimators_])
        return preds.mean(axis=1)


# ======================================================================
# FLAML-tuned outcome model
# ======================================================================
class FLAMLOutcomeModel(BaseEstimator, RegressorMixin):
    """AutoML-tuned LGBM regressor for outcome modelling (μ_d)."""

    def __init__(self, time_budget: int = 100, seed: int = 0, n_splits: int = 3):
        self.time_budget = time_budget
        self.seed = seed
        self.n_splits = n_splits

    def fit(self, X, y):
        automl = AutoML()
        automl.fit(
            X_train=X,
            y_train=y,
            task="regression",
            metric="rmse",
            estimator_list=["lgbm"],
            time_budget=self.time_budget,
            early_stop=True,
            eval_method="cv",
            n_splits=self.n_splits,
            seed=self.seed,
            verbose=0,
        )
        self.best_config_ = automl.best_config
        self.model_ = LGBMRegressor(**self.best_config_, verbosity=-1)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


# ======================================================================
# Individual ATE estimators
# ======================================================================
def regression_ate(y, D, W) -> Tuple[float, float]:
    """OLS regression adjustment: Y ~ D + W.

    Returns (estimate, analytic_se).
    """
    X = sm.add_constant(np.column_stack([D, W]))
    model = sm.OLS(y, X).fit(cov_type="HC1")
    return float(model.params[1]), float(model.bse[1])


def outcome_modeling_ate(
    y: np.ndarray,
    D: np.ndarray,
    mu0_hat: np.ndarray,
    mu1_hat: np.ndarray,
) -> float:
    """Oaxaca-Blinder / T-learner ATE (Equation 3.7).

    Parameters
    ----------
    y : outcome
    D : treatment indicator
    mu0_hat, mu1_hat : predicted E[Y|D=d, X] for d=0,1 evaluated on all X.
    """
    return float(np.mean(mu1_hat - mu0_hat))


def ipw_ate(
    y: np.ndarray,
    D: np.ndarray,
    e_hat: np.ndarray,
) -> float:
    """Inverse probability weighting ATE (Equation 3.11).

    Parameters
    ----------
    y : outcome
    D : treatment indicator (0/1)
    e_hat : estimated propensity scores
    """
    w1 = D / e_hat
    w0 = (1 - D) / (1 - e_hat)
    return float(np.mean(w1 * y) - np.mean(w0 * y))


def psm_ate(
    y: np.ndarray,
    D: np.ndarray,
    e_hat: np.ndarray,
    n_neighbors: int = 1,
) -> float:
    """Propensity score matching ATE (Equation 3.9).

    Uses nearest-neighbor matching on the estimated propensity score.
    """
    treated_idx = np.where(D == 1)[0]
    control_idx = np.where(D == 0)[0]

    # Match treated → control
    nn_tc = NearestNeighbors(n_neighbors=n_neighbors)
    nn_tc.fit(e_hat[control_idx].reshape(-1, 1))
    _, idx_tc = nn_tc.kneighbors(e_hat[treated_idx].reshape(-1, 1))
    y0_hat_treated = np.mean(y[control_idx[idx_tc]], axis=1)

    # Match control → treated
    nn_ct = NearestNeighbors(n_neighbors=n_neighbors)
    nn_ct.fit(e_hat[treated_idx].reshape(-1, 1))
    _, idx_ct = nn_ct.kneighbors(e_hat[control_idx].reshape(-1, 1))
    y1_hat_control = np.mean(y[treated_idx[idx_ct]], axis=1)

    ate = np.mean(
        np.concatenate([
            y[treated_idx] - y0_hat_treated,
            y1_hat_control - y[control_idx],
        ])
    )
    return float(ate)


def doubly_robust_ate(
    y: np.ndarray,
    D: np.ndarray,
    e_hat: np.ndarray,
    mu0_hat: np.ndarray,
    mu1_hat: np.ndarray,
) -> float:
    """Augmented IPW / Doubly Robust ATE (Equation 3.13).

    Parameters
    ----------
    y : outcome
    D : treatment indicator (0/1)
    e_hat : propensity scores
    mu0_hat, mu1_hat : outcome model predictions for d=0, d=1
    """
    mu_D = np.where(D == 1, mu1_hat, mu0_hat)
    score = (
        (D / e_hat - (1 - D) / (1 - e_hat)) * (y - mu_D)
        + mu1_hat
        - mu0_hat
    )
    return float(np.mean(score))


# ======================================================================
# Cross-fitted DR estimation (Chernozhukov et al. 2018)
# ======================================================================
def cross_fitted_dr_ate(
    y: np.ndarray,
    D: np.ndarray,
    W: np.ndarray,
    n_folds: int = 3,
    time_budget: int = 100,
    seed: int = 0,
) -> Tuple[float, np.ndarray]:
    """Cross-fitted doubly robust ATE with FLAML-tuned nuisance models.

    Returns
    -------
    ate : float
        Point estimate.
    scores : ndarray
        Per-observation DR scores (for variance estimation).
    """
    from .propensity import FLAMLPropensityEstimator

    y = np.asarray(y, dtype=float)
    D = np.asarray(D, dtype=int)
    W = np.asarray(W, dtype=float)

    scores = np.zeros(len(y))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_idx, test_idx in kf.split(W):
        # Propensity
        ps_model = FLAMLPropensityEstimator(time_budget=time_budget, seed=seed)
        ps_model.fit(W[train_idx], D[train_idx])
        e_test = ps_model.predict_proba(W[test_idx])[:, 1]

        # Outcome models
        mu0_model = FLAMLOutcomeModel(time_budget=time_budget, seed=seed)
        mu1_model = FLAMLOutcomeModel(time_budget=time_budget, seed=seed)
        mu0_model.fit(W[train_idx][D[train_idx] == 0], y[train_idx][D[train_idx] == 0])
        mu1_model.fit(W[train_idx][D[train_idx] == 1], y[train_idx][D[train_idx] == 1])

        mu0_test = mu0_model.predict(W[test_idx])
        mu1_test = mu1_model.predict(W[test_idx])

        D_test = D[test_idx]
        y_test = y[test_idx]
        mu_D_test = np.where(D_test == 1, mu1_test, mu0_test)

        scores[test_idx] = (
            (D_test / e_test - (1 - D_test) / (1 - e_test)) * (y_test - mu_D_test)
            + mu1_test
            - mu0_test
        )

    ate = float(np.mean(scores))
    return ate, scores


# ======================================================================
# Bootstrap standard errors
# ======================================================================
def bootstrap_ate(
    estimator_fn,
    y: np.ndarray,
    D: np.ndarray,
    W: np.ndarray,
    e_hat: np.ndarray,
    mu0_hat: np.ndarray,
    mu1_hat: np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Non-parametric bootstrap for ATE standard errors.

    Parameters
    ----------
    estimator_fn : callable
        One of :func:`doubly_robust_ate`, :func:`ipw_ate`, etc.
        Must accept (y, D, e_hat, mu0_hat, mu1_hat) or a subset—will be
        called with keyword arguments matching its signature.
    n_bootstrap : int
        Number of bootstrap replications.

    Returns
    -------
    point_estimate, se, ci_lower, ci_upper : floats
    """
    import inspect

    rng = np.random.RandomState(seed)
    n = len(y)

    sig = inspect.signature(estimator_fn)
    param_names = list(sig.parameters.keys())

    kwargs_full = dict(y=y, D=D, W=W, e_hat=e_hat, mu0_hat=mu0_hat, mu1_hat=mu1_hat)
    kwargs = {k: v for k, v in kwargs_full.items() if k in param_names}

    point = estimator_fn(**kwargs)

    boot_estimates = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        kwargs_b = {k: np.asarray(v)[idx] for k, v in kwargs.items()}
        boot_estimates[b] = estimator_fn(**kwargs_b)

    se = float(np.std(boot_estimates, ddof=1))
    ci_lower = float(np.percentile(boot_estimates, 2.5))
    ci_upper = float(np.percentile(boot_estimates, 97.5))
    return point, se, ci_lower, ci_upper


# ======================================================================
# Run all estimators at once
# ======================================================================
def run_all_estimators(
    y: np.ndarray,
    D: np.ndarray,
    W: np.ndarray,
    e_hat: np.ndarray,
    mu0_hat: np.ndarray,
    mu1_hat: np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Run all five ATE estimators and return a summary table.

    Returns
    -------
    DataFrame with columns: Estimator, Estimate, SE, CI_Lower, CI_Upper.
    """
    results = []

    # 1. Regression
    reg_est, reg_se = regression_ate(y, D, W)
    results.append(("Reg", reg_est, reg_se, reg_est - 1.96 * reg_se, reg_est + 1.96 * reg_se))

    # 2-5: bootstrap the rest
    for name, fn in [
        ("OM", outcome_modeling_ate),
        ("IPW", ipw_ate),
        ("PSM", psm_ate),
        ("DR", doubly_robust_ate),
    ]:
        est, se, ci_lo, ci_hi = bootstrap_ate(
            fn, y, D, W, e_hat, mu0_hat, mu1_hat,
            n_bootstrap=n_bootstrap, seed=seed,
        )
        results.append((name, est, se, ci_lo, ci_hi))

    return pd.DataFrame(
        results, columns=["Estimator", "Estimate", "SE", "CI_Lower", "CI_Upper"]
    )
