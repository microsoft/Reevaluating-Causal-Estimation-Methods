"""
Sensitivity analysis following Chernozhukov et al. (2022).

Quantifies how strongly an omitted confounder U would need to relate to both
the outcome Y and treatment D (via C²_Y, C²_D, and ρ) in order to shift the
estimated ATE to zero or to move the CI to include zero.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from typing import Dict, Tuple


def _partial_r2_outcome(
    y: np.ndarray,
    D: np.ndarray,
    mu_hat: np.ndarray,
    mu_hat_minus_j: np.ndarray,
) -> float:
    """Non-parametric partial R² of confounder j with outcome Y.

    C²_Y = [Var(E[Y|D,X,U]) − Var(E[Y|D,X])] / [Var(Y) − Var(E[Y|D,X])]

    Approximated by comparing outcome-model R² with and without feature j.
    """
    ss_full = np.var(mu_hat)
    ss_reduced = np.var(mu_hat_minus_j)
    ss_total = np.var(y)
    denom = ss_total - ss_full
    if denom <= 0:
        return 0.0
    return float(max(0, (ss_full - ss_reduced) / denom))


def _partial_r2_treatment(
    D: np.ndarray,
    e_hat: np.ndarray,
    e_hat_minus_j: np.ndarray,
) -> float:
    """Relative gain in treatment prediction precision when adding feature j.

    C²_D = [E[1/(e(1−e))|X,U] − E[1/(e(1−e))|X]] / E[1/(e(1−e))|X,U]

    Approximated using propensity model with and without feature j.
    """
    # Precision with full model
    e_full = np.clip(e_hat, 1e-6, 1 - 1e-6)
    prec_full = np.mean(1.0 / (e_full * (1 - e_full)))

    # Precision with reduced model
    e_red = np.clip(e_hat_minus_j, 1e-6, 1 - 1e-6)
    prec_red = np.mean(1.0 / (e_red * (1 - e_red)))

    if prec_full <= 0:
        return 0.0
    return float(max(0, (prec_full - prec_red) / prec_full))


def sensitivity_bound(
    c_y: float,
    c_d: float,
    rho: float,
    S: float,
) -> float:
    """Upper bound on ATE bias from an unobserved confounder.

    |τ_S − τ| ≤ |ρ| × C_Y × C_D × S

    Parameters
    ----------
    c_y : C²_Y (partial R² with outcome)
    c_d : C²_D (relative gain in treatment prediction)
    rho : correlation between confounder effects on Y and D
    S : sensitivity scaling factor

    Returns
    -------
    bias_bound : float
    """
    return abs(rho) * np.sqrt(c_y) * np.sqrt(c_d) * S


def robustness_value(
    ate: float,
    se: float,
    S: float,
    rho: float = 1.0,
) -> Tuple[float, float]:
    """Compute robustness values RV(ρ) and RVα(ρ).

    Parameters
    ----------
    ate : ATE point estimate
    se : standard error
    S : sensitivity scaling factor
    rho : assumed correlation

    Returns
    -------
    rv : float
        Minimum C_Y = C_D to shift point estimate to zero.
    rv_alpha : float
        Minimum C_Y = C_D to shift CI to include zero.
    """
    if S == 0 or rho == 0:
        return np.inf, np.inf

    # RV: |ate| = |rho| * rv * rv * S  →  rv = sqrt(|ate| / (|rho| * S))
    rv = abs(ate) / (abs(rho) * S)
    # For rv_alpha: |ate| - 1.96*se = |rho| * rv_alpha^2 * S
    ci_edge = abs(ate) - 1.96 * se
    if ci_edge <= 0:
        rv_alpha = 0.0
    else:
        rv_alpha = ci_edge / (abs(rho) * S)

    return float(rv), float(rv_alpha)


def compute_sensitivity_scaling(
    y: np.ndarray,
    D: np.ndarray,
    e_hat: np.ndarray,
    mu_hat: np.ndarray,
) -> float:
    """Compute the sensitivity scaling factor S.

    S = sqrt(Var(Y - μ_D(X)) × E[1/(e(X)(1-e(X)))])
    """
    mu_D = mu_hat  # prediction under observed treatment
    resid_var = np.var(y - mu_D)
    e_clip = np.clip(e_hat, 1e-6, 1 - 1e-6)
    precision = np.mean(1.0 / (e_clip * (1 - e_clip)))
    return float(np.sqrt(resid_var * precision))


def benchmark_confounders(
    y: np.ndarray,
    D: np.ndarray,
    W: np.ndarray,
    feature_names: list,
    e_hat: np.ndarray,
    mu0_hat: np.ndarray,
    mu1_hat: np.ndarray,
    fit_propensity_fn=None,
    fit_outcome_fn=None,
) -> pd.DataFrame:
    """Benchmark each observed covariate by leave-one-out refitting.

    For each covariate j, refit propensity and outcome models without j,
    then compute (C²_Y, C²_D) to benchmark against unobserved confounders.

    Parameters
    ----------
    fit_propensity_fn : callable(W, D) -> e_hat
        Function that re-estimates propensity scores on reduced W.
    fit_outcome_fn : callable(W, y, D) -> (mu0_hat, mu1_hat)
        Function that re-estimates outcome models on reduced W.

    Returns
    -------
    DataFrame with columns: feature, C2_Y, C2_D
    """
    mu_hat = np.where(D == 1, mu1_hat, mu0_hat)
    rows = []

    for j, fname in enumerate(feature_names):
        W_minus_j = np.delete(W, j, axis=1)

        if fit_propensity_fn is not None:
            e_minus_j = fit_propensity_fn(W_minus_j, D)
        else:
            e_minus_j = e_hat  # fallback: no refit

        if fit_outcome_fn is not None:
            mu0_j, mu1_j = fit_outcome_fn(W_minus_j, y, D)
            mu_j = np.where(D == 1, mu1_j, mu0_j)
        else:
            mu_j = mu_hat

        c2_y = _partial_r2_outcome(y, D, mu_hat, mu_j)
        c2_d = _partial_r2_treatment(D, e_hat, e_minus_j)
        rows.append({"feature": fname, "C2_Y": c2_y, "C2_D": c2_d})

    return pd.DataFrame(rows)
