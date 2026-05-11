"""
Shared utility functions: cross-fitting, outcome model ensembling,
standardised mean differences, and nuisance model performance tables.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, roc_auc_score
from typing import List, Tuple


def cross_fit_predict(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 3,
    seed: int = 0,
    method: str = "predict",
) -> np.ndarray:
    """Return out-of-fold predictions from cross-fitting.

    Parameters
    ----------
    model : sklearn-compatible estimator
    X, y : training data
    n_folds : number of folds
    seed : random seed
    method : "predict" or "predict_proba"
    """
    X = np.asarray(X)
    y = np.asarray(y)
    predictions = np.zeros(len(y)) if method == "predict" else np.zeros((len(y), 2))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_idx, test_idx in kf.split(X):
        m = clone(model).fit(X[train_idx], y[train_idx])
        if method == "predict_proba":
            predictions[test_idx] = m.predict_proba(X[test_idx])
        else:
            predictions[test_idx] = m.predict(X[test_idx])
    return predictions


def standardized_mean_differences(
    df: pd.DataFrame,
    group_col: str = "D",
    feature_cols: List[str] = None,
) -> pd.DataFrame:
    """Compute standardized mean differences between groups (Appendix Table A.1/A.2).

    Returns a DataFrame with columns: Variable, Group0_Mean, Group1_Mean, SMD.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in [group_col, "y"]]

    g0 = df[df[group_col] == 0]
    g1 = df[df[group_col] == 1]

    rows = []
    for col in feature_cols:
        m0, m1 = g0[col].mean(), g1[col].mean()
        s0, s1 = g0[col].std(), g1[col].std()
        pooled_sd = np.sqrt((s0 ** 2 + s1 ** 2) / 2)
        smd = (m1 - m0) / pooled_sd if pooled_sd > 0 else 0.0
        rows.append({
            "Variable": col,
            "Control Mean": f"{m0:.4f} ({s0:.4f})",
            "Treated Mean": f"{m1:.4f} ({s1:.4f})",
            "SMD": round(smd, 3),
        })
    return pd.DataFrame(rows)


def nuisance_performance_table(
    y: np.ndarray,
    D: np.ndarray,
    W: np.ndarray,
    mu0_hat: np.ndarray,
    mu1_hat: np.ndarray,
    e_hat: np.ndarray,
    model_name: str = "Tuned LGBM",
) -> dict:
    """Compute performance metrics for nuisance models (Table 3).

    Returns dict with mu0_r2, mu1_r2, e_auc.
    """
    # R² for outcome models on their respective subsets
    ctrl_mask = D == 0
    treat_mask = D == 1

    mu0_r2 = r2_score(y[ctrl_mask], mu0_hat[ctrl_mask])
    mu1_r2 = r2_score(y[treat_mask], mu1_hat[treat_mask])
    e_auc = roc_auc_score(D, e_hat)

    return {
        "Model": model_name,
        "mu0 R²": round(mu0_r2, 3),
        "mu1 R²": round(mu1_r2, 3),
        "e AUC": round(e_auc, 3),
    }


def ensemble_outcome_models(
    W: np.ndarray,
    y: np.ndarray,
    D: np.ndarray,
    n_models: int = 50,
    time_budget: int = 100,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit ensembled outcome models μ0(X) and μ1(X).

    Returns
    -------
    mu0_hat, mu1_hat : arrays of shape (n,)
        Averaged predictions from ``n_models`` FLAML-tuned LGBM regressors.
    """
    from .estimators import FLAMLOutcomeModel

    W = np.asarray(W)
    y = np.asarray(y)
    D = np.asarray(D)
    n = len(y)

    mu0_all = np.zeros((n, n_models))
    mu1_all = np.zeros((n, n_models))

    ctrl_mask = D == 0
    treat_mask = D == 1

    for i in range(n_models):
        if verbose:
            print(f"  Fitting outcome models {i + 1}/{n_models} ...")
        m0 = FLAMLOutcomeModel(time_budget=time_budget, seed=i)
        m1 = FLAMLOutcomeModel(time_budget=time_budget, seed=i)
        m0.fit(W[ctrl_mask], y[ctrl_mask])
        m1.fit(W[treat_mask], y[treat_mask])
        mu0_all[:, i] = m0.predict(W)
        mu1_all[:, i] = m1.predict(W)

    return mu0_all.mean(axis=1), mu1_all.mean(axis=1)
