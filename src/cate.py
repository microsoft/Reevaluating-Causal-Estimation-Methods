"""
Conditional Average Treatment Effect (CATE) estimation.

Implements meta-learners (DR-, R-, X-, S-, T-learner), DR-scores for
heterogeneity detection (Chernozhukov et al. 2024), and Q-aggregation
(Lan & Syrgkanis 2024) for model combination.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from typing import Dict, List, Optional, Tuple


# ======================================================================
# DR pseudo-outcome (Equation 3.15 / 6.1)
# ======================================================================
def dr_pseudo_outcome(
    y: np.ndarray,
    D: np.ndarray,
    e_hat: np.ndarray,
    mu0_hat: np.ndarray,
    mu1_hat: np.ndarray,
) -> np.ndarray:
    """Compute doubly robust pseudo-outcomes Y^DR (Equation 3.15).

    Y^DR_i = (mu1 - mu0) + (D_i - e_i) / [e_i(1-e_i)] * (Y_i - mu_{D_i})
    """
    mu_D = np.where(D == 1, mu1_hat, mu0_hat)
    return (mu1_hat - mu0_hat) + (D - e_hat) / (e_hat * (1 - e_hat)) * (y - mu_D)


# ======================================================================
# Meta-learners
# ======================================================================
def dr_learner(
    W: np.ndarray,
    y_dr: np.ndarray,
    model=None,
) -> BaseEstimator:
    """Fit a DR-learner: regress Y^DR on X (Kennedy 2023)."""
    if model is None:
        model = LGBMRegressor(n_estimators=200, verbosity=-1)
    model = clone(model)
    model.fit(W, y_dr)
    return model


def t_learner(
    W: np.ndarray,
    y: np.ndarray,
    D: np.ndarray,
    model=None,
) -> Tuple[BaseEstimator, BaseEstimator]:
    """T-learner: fit separate models μ0(X) and μ1(X)."""
    if model is None:
        model = LGBMRegressor(n_estimators=200, verbosity=-1)
    m0 = clone(model).fit(W[D == 0], y[D == 0])
    m1 = clone(model).fit(W[D == 1], y[D == 1])
    return m0, m1


def s_learner(
    W: np.ndarray,
    y: np.ndarray,
    D: np.ndarray,
    model=None,
) -> BaseEstimator:
    """S-learner: fit one model μ(X, D)."""
    if model is None:
        model = LGBMRegressor(n_estimators=200, verbosity=-1)
    X_aug = np.column_stack([W, D])
    model = clone(model).fit(X_aug, y)
    return model


def x_learner_cate(
    W: np.ndarray,
    y: np.ndarray,
    D: np.ndarray,
    e_hat: np.ndarray,
    model=None,
) -> np.ndarray:
    """X-learner CATE estimates (Künzel et al. 2019)."""
    if model is None:
        model = LGBMRegressor(n_estimators=200, verbosity=-1)

    # Stage 1: T-learner
    m0, m1 = t_learner(W, y, D, model)

    # Stage 2: imputed treatment effects
    d1 = y[D == 1] - m0.predict(W[D == 1])
    d0 = m1.predict(W[D == 0]) - y[D == 0]

    tau1 = clone(model).fit(W[D == 1], d1)
    tau0 = clone(model).fit(W[D == 0], d0)

    # Stage 3: weighted combination
    cate = e_hat * tau0.predict(W) + (1 - e_hat) * tau1.predict(W)
    return cate


def r_learner_cate(
    W: np.ndarray,
    y: np.ndarray,
    D: np.ndarray,
    e_hat: np.ndarray,
    mu_hat: np.ndarray,
    model=None,
) -> BaseEstimator:
    """R-learner CATE (Nie & Wager 2020).

    Parameters
    ----------
    mu_hat : predicted E[Y|X] (marginal outcome model).
    """
    if model is None:
        model = LGBMRegressor(n_estimators=200, verbosity=-1)

    y_resid = y - mu_hat
    d_resid = D - e_hat
    pseudo_y = y_resid / d_resid
    weights = d_resid ** 2

    model = clone(model)
    model.fit(W, pseudo_y, sample_weight=weights)
    return model


# ======================================================================
# DR-score for heterogeneity detection (Equation 6.2)
# ======================================================================
def dr_score(
    y_dr: np.ndarray,
    tau_hat: np.ndarray,
    tau_const: Optional[float] = None,
) -> float:
    """Normalised DR-score (Equation 6.2).

    A score of 0 means the CATE model is no better than a constant;
    positive values indicate captured heterogeneity.
    """
    if tau_const is None:
        tau_const = np.mean(y_dr)
    mse_const = np.mean((y_dr - tau_const) ** 2)
    mse_model = np.mean((y_dr - tau_hat) ** 2)
    if mse_const == 0:
        return 0.0
    return float((mse_const - mse_model) / mse_const)


def cross_validated_dr_score(
    W: np.ndarray,
    y_dr: np.ndarray,
    model=None,
    n_folds: int = 3,
    seed: int = 0,
) -> float:
    """Out-of-sample DR-score via cross-validation."""
    if model is None:
        model = LGBMRegressor(n_estimators=200, verbosity=-1)

    tau_const = np.mean(y_dr)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    residuals_const = 0.0
    residuals_model = 0.0

    for train_idx, test_idx in kf.split(W):
        m = clone(model).fit(W[train_idx], y_dr[train_idx])
        tau_test = m.predict(W[test_idx])
        residuals_const += np.sum((y_dr[test_idx] - tau_const) ** 2)
        residuals_model += np.sum((y_dr[test_idx] - tau_test) ** 2)

    if residuals_const == 0:
        return 0.0
    return float((residuals_const - residuals_model) / residuals_const)


# ======================================================================
# Q-aggregation (Lan & Syrgkanis 2024, Equation 6.3)
# ======================================================================
def q_aggregate(
    y_dr: np.ndarray,
    cate_predictions: Dict[str, np.ndarray],
    nu: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Q-aggregation of multiple CATE models.

    Parameters
    ----------
    y_dr : DR pseudo-outcomes.
    cate_predictions : dict mapping model name → predicted CATE array.
    nu : interpolation between convex regression (0) and best model (1).

    Returns
    -------
    aggregated_cate : ndarray
    weights : dict mapping model name → weight
    """
    from scipy.optimize import minimize

    names = list(cate_predictions.keys())
    preds = np.column_stack([cate_predictions[n] for n in names])
    tau_bar = np.mean(y_dr)

    # Center predictions
    preds_c = preds - tau_bar
    y_c = y_dr - tau_bar
    M = len(names)

    def objective(w):
        w = np.array(w)
        tau_w = preds_c @ w
        convex_loss = np.mean((y_c - tau_w) ** 2)
        individual_losses = np.array([np.mean((y_c - preds_c[:, m]) ** 2) for m in range(M)])
        return (1 - nu) * convex_loss + nu * np.dot(w, individual_losses)

    # Simplex constraint
    from scipy.optimize import LinearConstraint

    constraints = LinearConstraint(np.ones(M), lb=1.0, ub=1.0)
    bounds = [(0, 1)] * M
    x0 = np.ones(M) / M

    result = minimize(objective, x0, bounds=bounds, constraints=constraints, method="SLSQP")
    w_opt = result.x

    aggregated = tau_bar + preds_c @ w_opt
    weights = {n: float(w_opt[i]) for i, n in enumerate(names)}
    return aggregated, weights
