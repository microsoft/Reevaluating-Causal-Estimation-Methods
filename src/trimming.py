"""
Sample trimming for overlap (Crump et al., 2009).

Implements the optimal trimming rule that minimises the asymptotic variance
of ATE estimates by selecting the propensity-score threshold α such that
A* = {x : α ≤ e(x) ≤ 1 − α}.
"""

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
from typing import Tuple


def optimal_trim_threshold(propensity_scores: np.ndarray) -> float:
    """Compute the Crump et al. (2009) optimal trimming threshold α.

    Parameters
    ----------
    propensity_scores : array of shape (n,)
        Estimated propensity scores, each in (0, 1).

    Returns
    -------
    alpha : float
        Optimal cutoff.  Keep observations with α ≤ e(x) ≤ 1 − α.
    """
    ps = np.asarray(propensity_scores, dtype=float)
    sum_wt = 1.0 / (ps * (1.0 - ps))
    k = 2.0 * np.mean(sum_wt) - np.max(sum_wt)

    if k >= 0:
        return 0.0

    def _trim_fun(x):
        return x - 2.0 * np.mean(sum_wt[sum_wt <= x])

    bracket = [float(np.min(sum_wt)), float(np.max(sum_wt))]
    result = root_scalar(_trim_fun, bracket=bracket, method="brentq")
    lambda_opt = result.root
    alpha = 0.5 - np.sqrt(0.25 - 1.0 / lambda_opt)
    return float(alpha)


def trim_sample(
    df: pd.DataFrame,
    propensity_scores: np.ndarray,
    alpha: float,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Trim a dataframe and its propensity scores using threshold α.

    Parameters
    ----------
    df : DataFrame
        The sample to trim.
    propensity_scores : array of shape (n,)
        Propensity scores aligned with *df*.
    alpha : float
        Trimming threshold from :func:`optimal_trim_threshold`.

    Returns
    -------
    df_trimmed, ps_trimmed : tuple
    """
    ps = np.asarray(propensity_scores)
    mask = (ps >= alpha) & (ps <= 1.0 - alpha)
    return df.loc[mask].reset_index(drop=True), ps[mask]


def trim_summary(
    n_original: int,
    n_trimmed: int,
    n_treated_original: int,
    n_treated_trimmed: int,
) -> pd.DataFrame:
    """Return a summary table like Table 2 in the paper.

    Parameters
    ----------
    n_original, n_trimmed : int
        Total observations before / after trimming.
    n_treated_original, n_treated_trimmed : int
        Treated observations before / after trimming.

    Returns
    -------
    DataFrame
    """
    n_ctrl_orig = n_original - n_treated_original
    n_ctrl_trim = n_trimmed - n_treated_trimmed
    rows = {
        "Number of Observations": [n_original, n_trimmed, 1 - n_trimmed / n_original],
        "Number of Treated Units": [
            n_treated_original,
            n_treated_trimmed,
            1 - n_treated_trimmed / n_treated_original,
        ],
        "Number of Control Units": [
            n_ctrl_orig,
            n_ctrl_trim,
            1 - n_ctrl_trim / n_ctrl_orig,
        ],
    }
    return pd.DataFrame(
        rows, index=["Untrimmed", "Trimmed", "% Dropped"]
    ).T
