"""
Plotting utilities for paper figures.

Generates the key figures from the paper:
- Propensity score distributions (Figure 1)
- ATE comparison charts (Figures 2, 3, 6, 7)
- Nuisance model performance (Figure 4)
- Model uncertainty (Figure 5)
- Sensitivity contour plots (Figure 8)
- DR-score bar charts (Figure 9)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple

# Paper-quality defaults
plt.rcParams.update({
    "figure.figsize": (8, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def plot_propensity_distributions(
    ps_control: np.ndarray,
    ps_treated: np.ndarray,
    title: str = "Estimated Propensity Score Distributions",
    trim_threshold: Optional[float] = None,
    xlim: Tuple[float, float] = (0, 0.20),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot propensity score distributions for treated/control (Figure 1)."""
    if ax is None:
        _, ax = plt.subplots()

    ax.hist(ps_control, bins=80, density=True, alpha=0.5, label="Control", color="tab:blue")
    ax.hist(ps_treated, bins=80, density=True, alpha=0.5, label="Treated", color="tab:orange")

    if trim_threshold is not None:
        ax.axvline(trim_threshold, color="gray", linestyle="--", linewidth=1,
                   label=f"Trim threshold: {trim_threshold:.4f}")

    ax.set_xlabel("Propensity Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.legend()
    return ax


def plot_ate_comparison(
    results: pd.DataFrame,
    benchmark_est: float,
    benchmark_se: float,
    naive_est: Optional[float] = None,
    naive_se: Optional[float] = None,
    title: str = "ATE Estimates",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot ATE estimates with CIs against the experimental benchmark (Figure 2/3).

    Parameters
    ----------
    results : DataFrame
        Must have columns: Estimator, Estimate, CI_Lower, CI_Upper.
    benchmark_est, benchmark_se : float
        Experimental ground truth point estimate and SE.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    # Benchmark band
    ax.axhspan(
        benchmark_est - 1.96 * benchmark_se,
        benchmark_est + 1.96 * benchmark_se,
        alpha=0.15, color="tab:blue", label="Experimental 95% CI",
    )
    ax.axhline(benchmark_est, color="tab:blue", linestyle="--", linewidth=1.5,
               label=f"Experimental benchmark: {benchmark_est:.4f}")

    # Naive DM
    if naive_est is not None and naive_se is not None:
        ax.axhline(naive_est, color="black", linestyle="--", linewidth=1, alpha=0.7,
                   label=f"Naive diff-in-means: {naive_est:.4f}")
        ax.axhspan(naive_est - 1.96 * naive_se, naive_est + 1.96 * naive_se,
                   alpha=0.08, color="black")

    # Estimator points
    x_pos = np.arange(len(results))
    ax.errorbar(
        x_pos,
        results["Estimate"],
        yerr=[
            results["Estimate"] - results["CI_Lower"],
            results["CI_Upper"] - results["Estimate"],
        ],
        fmt="o",
        capsize=5,
        color="tab:red",
        markersize=8,
        linewidth=2,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results["Estimator"])
    ax.set_ylabel("ATE Estimate")
    ax.set_title(title)
    ax.legend(loc="best")
    return ax


def plot_nuisance_model_comparison(
    results: pd.DataFrame,
    benchmark_est: float,
    benchmark_se: float,
    title: str = "DR Estimates Across First-Stage Models",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot DR ATE estimates for different nuisance models (Figure 4).

    Parameters
    ----------
    results : DataFrame
        Must have columns: Model, Estimate, CI_Lower, CI_Upper.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.axhspan(
        benchmark_est - 1.96 * benchmark_se,
        benchmark_est + 1.96 * benchmark_se,
        alpha=0.15, color="tab:blue",
    )
    ax.axhline(benchmark_est, color="tab:blue", linestyle="--", linewidth=1.5)

    x_pos = np.arange(len(results))
    ax.errorbar(
        x_pos,
        results["Estimate"],
        yerr=[
            results["Estimate"] - results["CI_Lower"],
            results["CI_Upper"] - results["Estimate"],
        ],
        fmt="s",
        capsize=4,
        color="tab:red",
        markersize=7,
        linewidth=1.5,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results["Model"], rotation=30, ha="right")
    ax.set_ylabel("ATE Estimate")
    ax.set_title(title)
    return ax


def plot_model_uncertainty(
    ensemble_ate: float,
    single_run_ates: Dict[str, np.ndarray],
    benchmark_est: float,
    benchmark_se: float,
    title: str = "Model Uncertainty: ATE Estimates",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot model instability across seeds (Figure 5).

    Parameters
    ----------
    ensemble_ate : float
        Fully ensembled ATE.
    single_run_ates : dict
        Mapping scenario label → array of ATE estimates from individual seeds.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.axhspan(
        benchmark_est - 1.96 * benchmark_se,
        benchmark_est + 1.96 * benchmark_se,
        alpha=0.15, color="tab:blue",
    )
    ax.axhline(ensemble_ate, color="red", linestyle=":", linewidth=2,
               label=f"Ensembled: {ensemble_ate:.4f}")

    for i, (label, ates) in enumerate(single_run_ates.items()):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(ates))
        ax.scatter(np.full_like(ates, i) + jitter, ates, alpha=0.5, s=30, label=label)

    ax.set_xticks(range(len(single_run_ates)))
    ax.set_xticklabels(list(single_run_ates.keys()), rotation=15, ha="right")
    ax.set_ylabel("ATE Estimate")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_sensitivity_contour(
    ate: float,
    S: float,
    rho: float = 0.5,
    benchmark_cy: Optional[float] = None,
    benchmark_cd: Optional[float] = None,
    title: str = "Sensitivity Analysis",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot sensitivity contour (Figure 8).

    Isoquants show adjusted ATE for varying (C_D, C_Y) strength of an
    unobserved confounder.
    """
    if ax is None:
        _, ax = plt.subplots()

    cd_grid = np.linspace(0, 0.10, 200)
    cy_grid = np.linspace(0, 0.10, 200)
    CD, CY = np.meshgrid(cd_grid, cy_grid)

    bias = abs(rho) * np.sqrt(CY) * np.sqrt(CD) * S
    adjusted = ate - np.sign(ate) * bias

    contour = ax.contourf(CD, CY, adjusted, levels=20, cmap="RdYlBu")
    plt.colorbar(contour, ax=ax, label="Adjusted ATE")

    # Zero line
    ax.contour(CD, CY, adjusted, levels=[0], colors="red", linestyles="--", linewidths=2)

    # Unadjusted point
    ax.plot(0, 0, "ko", markersize=8, label=f"Unadjusted: {ate:.4f}")

    # Benchmark dots
    if benchmark_cy is not None and benchmark_cd is not None:
        ax.plot(benchmark_cd, benchmark_cy, "bo", markersize=8, label="1x benchmark")
        ax.plot(2 * benchmark_cd, 2 * benchmark_cy, "bs", markersize=8, label="2x benchmark")

    ax.set_xlabel("$C^2_D$ (confounder impact on treatment)")
    ax.set_ylabel("$C^2_Y$ (confounder impact on outcome)")
    ax.set_title(title)
    ax.legend(loc="best")
    return ax


def plot_dr_scores(
    scores: Dict[str, float],
    title: str = "DR Scores by Learner",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Bar chart of DR-scores for each CATE learner (Figure 9)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    names = list(scores.keys())
    vals = list(scores.values())
    colors = ["tab:green" if v > 0 else "tab:red" for v in vals]

    ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("DR Score")
    ax.set_title(title)
    return ax
