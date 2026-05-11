"""
Propensity score estimation with AutoML (FLAML) tuning and model ensembling.

Implements the propensity estimation pipeline described in Section 4.2 of the
paper: fit multiple tuned LGBM classifiers with different seeds, then ensemble
via averaging to reduce model uncertainty (Breiman, 1996).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from flaml import AutoML
from lightgbm import LGBMClassifier
from typing import List, Optional


# ---------------------------------------------------------------------------
# Averaging ensembles
# ---------------------------------------------------------------------------
class AveragingClassifier(BaseEstimator, ClassifierMixin):
    """Ensemble that averages predicted probabilities from base classifiers."""

    def __init__(self, estimators: Optional[List] = None):
        self.estimators = estimators if estimators is not None else []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.fitted_estimators_ = [clone(est).fit(X, y) for est in self.estimators]
        return self

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        if not hasattr(self, "fitted_estimators_"):
            raise NotFittedError("AveragingClassifier is not fitted yet.")
        all_probs = np.stack(
            [est.predict_proba(X) for est in self.fitted_estimators_], axis=0
        )
        return all_probs.mean(axis=0)


# ---------------------------------------------------------------------------
# FLAML-based propensity estimator
# ---------------------------------------------------------------------------
class FLAMLPropensityEstimator(BaseEstimator, ClassifierMixin):
    """AutoML-tuned LGBM propensity score model using FLAML.

    Parameters
    ----------
    time_budget : int
        Seconds allocated to FLAML hyperparameter search per seed.
    seed : int
        Random seed for reproducibility.
    n_splits : int
        Number of CV folds used during FLAML tuning.
    """

    def __init__(self, time_budget: int = 100, seed: int = 0, n_splits: int = 3):
        self.time_budget = time_budget
        self.seed = seed
        self.n_splits = n_splits
        self.model_ = None
        self.best_config_ = None

    def fit(self, X, y):
        automl = AutoML()
        automl.fit(
            X_train=X,
            y_train=y,
            task="classification",
            metric="log_loss",
            estimator_list=["lgbm"],
            time_budget=self.time_budget,
            early_stop=True,
            eval_method="cv",
            n_splits=self.n_splits,
            seed=self.seed,
            verbose=0,
        )
        self.best_config_ = automl.best_config
        self.model_ = LGBMClassifier(**self.best_config_, verbosity=-1)
        self.model_.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


# ---------------------------------------------------------------------------
# Main pipeline: ensemble propensity estimation
# ---------------------------------------------------------------------------
def estimate_propensity_ensemble(
    W: np.ndarray,
    D: np.ndarray,
    n_models: int = 50,
    time_budget: int = 100,
    n_splits: int = 3,
    verbose: bool = True,
) -> np.ndarray:
    """Estimate propensity scores as the ensemble average of *n_models*
    FLAML-tuned LGBM classifiers, each trained with a different random seed.

    Parameters
    ----------
    W : array-like of shape (n, p)
        Covariates.
    D : array-like of shape (n,)
        Binary treatment.
    n_models : int
        Number of models to ensemble.
    time_budget : int
        FLAML time budget per model (seconds).
    n_splits : int
        CV folds during tuning.
    verbose : bool
        Print progress.

    Returns
    -------
    propensity_scores : ndarray of shape (n,)
        Averaged predicted P(D=1 | W).
    """
    W = np.asarray(W)
    D = np.asarray(D)
    all_probs = np.zeros((len(D), n_models))

    for i in range(n_models):
        if verbose:
            print(f"  Fitting propensity model {i + 1}/{n_models} ...")
        model = FLAMLPropensityEstimator(
            time_budget=time_budget, seed=i, n_splits=n_splits
        )
        model.fit(W, D)
        all_probs[:, i] = model.predict_proba(W)[:, 1]

    avg_probs = all_probs.mean(axis=1)
    if verbose:
        auc = roc_auc_score(D, avg_probs)
        print(f"  Ensemble propensity AUC: {auc:.4f}")
    return avg_probs


def cross_fit_propensity(
    W: np.ndarray,
    D: np.ndarray,
    n_folds: int = 3,
    time_budget: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Cross-fitted propensity scores (single model per fold).

    Parameters
    ----------
    W, D : arrays
        Covariates and treatment.
    n_folds : int
        Number of cross-fitting folds.
    time_budget : int
        FLAML time budget per fold.
    seed : int
        Random seed.

    Returns
    -------
    pscores : ndarray of shape (n,)
    """
    W = np.asarray(W)
    D = np.asarray(D)
    pscores = np.zeros(len(D))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_idx, test_idx in kf.split(W):
        model = FLAMLPropensityEstimator(
            time_budget=time_budget, seed=seed, n_splits=n_folds
        )
        model.fit(W[train_idx], D[train_idx])
        pscores[test_idx] = model.predict_proba(W[test_idx])[:, 1]

    return pscores
