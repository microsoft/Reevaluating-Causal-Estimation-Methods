"""
Data loading utilities for the causal validation benchmark datasets.

Loads the public parquet files and defines standard column groups used
throughout the analysis (outcome, treatment, covariates).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Column definitions (public data schema)
# ---------------------------------------------------------------------------
TREATMENT = "New_Feature"
OUTCOME_CONTINUOUS = "Outcome_Continuous"
OUTCOME_BINARY = "Outcome_Binary"

COVARIATES = [
    "Device_Spec_A",
    "Total_Device_Usage",
    "Browser_1_Usage",
    "Browser_2_Usage",
    "Browser_3_Usage",
    "Browser_4_Usage",
    "Other_Browser_Usage",
    # Engagement segment (one-hot, first category dropped)
    "EngagementSegment_General",
    "EngagementSegment_Highly Engaged",
    "EngagementSegment_Inactive",
    "EngagementSegment_Low Engaged",
    # Region (one-hot, scrambled, first category dropped)
    "A14Region_APAC",
    "A14Region_CEE",
    "A14Region_Canada",
    "A14Region_France",
    "A14Region_Germany",
    "A14Region_Greater China",
    "A14Region_India",
    "A14Region_Japan",
    "A14Region_Korea",
    "A14Region_Latam",
    "A14Region_MEA",
    "A14Region_UK",
    "A14Region_United States",
    "A14Region_Western Europe",
    # App-usage cohort (one-hot, first category dropped)
    "AppCategoryCohort_Developer",
    "AppCategoryCohort_Gamer",
    "AppCategoryCohort_General",
    "AppCategoryCohort_Media",
    "AppCategoryCohort_Productivity",
    # Manufacturer (one-hot, anonymised, first category dropped)
    "Manufacturer_1",
    "Manufacturer_2",
    "Manufacturer_3",
    "Manufacturer_4",
    "Manufacturer_5",
    "Manufacturer_6",
    # Device specs (one-hot, first category dropped)
    "Device_Spec_B_1",
    "Device_Spec_B_2",
    "Device_Spec_C_1",
    "Device_Spec_C_2",
    "Device_Spec_C_3",
]


def load_data(
    data_dir: str = "Data",
    outcome: str = "continuous",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and return the experimental and observational DataFrames.

    Parameters
    ----------
    data_dir : str
        Path to the folder containing the parquet files.
    outcome : str
        ``"continuous"`` or ``"binary"`` — selects which outcome column to
        keep as ``y``.

    Returns
    -------
    observed, experiment : tuple of DataFrames
        Each DataFrame has columns ``y``, ``D``, and all covariates in
        ``COVARIATES``.
    """
    data_dir = Path(data_dir)
    obs = pd.read_parquet(data_dir / "FINAL_PUBLIC_observed.parquet")
    exp = pd.read_parquet(data_dir / "FINAL_PUBLIC_experimental.parquet")

    outcome_col = OUTCOME_CONTINUOUS if outcome == "continuous" else OUTCOME_BINARY

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        out = df[[outcome_col, TREATMENT] + COVARIATES].copy()
        out = out.rename(columns={outcome_col: "y", TREATMENT: "D"})
        out["D"] = out["D"].astype(int)
        out = out.reset_index(drop=True)
        return out

    return _prep(obs), _prep(exp)


def difference_in_means(
    df: pd.DataFrame,
) -> Tuple[float, float]:
    """Compute the simple difference-in-means estimator and its variance.

    Parameters
    ----------
    df : DataFrame
        Must contain columns ``y`` and ``D``.

    Returns
    -------
    estimate, variance : tuple of float
    """
    y1 = df.loc[df["D"] == 1, "y"]
    y0 = df.loc[df["D"] == 0, "y"]
    est = y1.mean() - y0.mean()
    var = y1.var() / len(y1) + y0.var() / len(y0)
    return est, var
