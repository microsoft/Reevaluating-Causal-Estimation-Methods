"""Sklearn-compatible averaging ensemble wrappers.

These classes serve two purposes:

1. **Loading saved models** — the original Colab notebooks pickled
   ``AveragingClassifier`` and ``AveragingRegressor`` objects on ``__main__``.
   Importing this module and registering the classes (see :func:`register`)
   lets us unpickle those files in any environment.

2. **EconML integration** — ``clone()`` works correctly on these because all
   constructor arguments are stored as attributes, following the sklearn
   estimator protocol.  An ``AveragingRegressor`` with 10 pre-fitted LGBMs
   can be passed to ``econml.dr.LinearDRLearner`` as ``model_regression``
   (wrapped in ``econml.utilities.SeparateModel`` for y0/y1).  EconML's
   internal cross-fitting will ``clone()`` then ``fit()`` each fold.
"""
from __future__ import annotations

import sys
from typing import List, Optional

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    clone,
)
from sklearn.exceptions import NotFittedError


class AveragingClassifier(BaseEstimator, ClassifierMixin):
    """Average predicted probabilities from multiple classifiers."""

    def __init__(self, estimators: Optional[List] = None):
        self.estimators = estimators if estimators is not None else []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.fitted_estimators_ = [clone(est).fit(X, y) for est in self.estimators]
        return self

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def predict_proba(self, X):
        if not hasattr(self, "fitted_estimators_"):
            raise NotFittedError(
                "This AveragingClassifier instance is not fitted yet."
            )
        all_probs = np.column_stack(
            [est.predict_proba(X) for est in self.fitted_estimators_]
        )
        return np.mean(
            all_probs.reshape(len(X), len(self.fitted_estimators_), -1), axis=1
        )


class AveragingRegressor(BaseEstimator, RegressorMixin):
    """Average predictions from multiple regressors."""

    def __init__(self, estimators: Optional[List] = None):
        self.estimators = estimators if estimators is not None else []

    def fit(self, X, y):
        self.fitted_estimators_ = [clone(est).fit(X, y) for est in self.estimators]
        return self

    def predict(self, X):
        if not hasattr(self, "fitted_estimators_"):
            raise NotFittedError(
                "This AveragingRegressor instance is not fitted yet."
            )
        preds = np.column_stack(
            [est.predict(X) for est in self.fitted_estimators_]
        )
        return np.mean(preds, axis=1)


# ---------------------------------------------------------------------------
# ``FLAMLRegressor`` stub — only needed for unpickling the saved y0/y1 model
# files.  The real payload is the ``.model_`` attribute which is a standard
# ``LGBMRegressor``.  After loading, callers should do::
#
#     lgbm = joblib.load("...seed0.pkl").model_
#
class FLAMLRegressor(BaseEstimator, RegressorMixin):
    """Shim for unpickling ``FLAMLRegressor`` objects saved on __main__."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def fit(self, X, y):  # pragma: no cover
        raise NotImplementedError(
            "FLAMLRegressor is a loading shim only. "
            "Extract .model_ (LGBMRegressor) and use that instead."
        )

    def predict(self, X):
        if hasattr(self, "model_"):
            return self.model_.predict(X)
        raise NotFittedError("No .model_ attribute found.")


# ---------------------------------------------------------------------------
def register():
    """Register wrapper classes on ``__main__`` so ``pickle.load`` can find them.

    Call this once before loading any saved model pkl files::

        from src.ensemble_wrappers import register
        register()
    """
    main = sys.modules.get("__main__")
    if main is not None:
        main.AveragingClassifier = AveragingClassifier
        main.AveragingRegressor = AveragingRegressor
        main.FLAMLRegressor = FLAMLRegressor
