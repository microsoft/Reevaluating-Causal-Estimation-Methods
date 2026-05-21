# Reevaluating Causal Estimation Methods with Data from a Product Release
**Justin Young and Eleanor W. Dillon (2026)**

This repository contains the benchmark datasets, replication code, and pre-computed artifacts for the paper
[*Reevaluating Causal Estimation Methods with Data from a Product Release*](https://drive.google.com/file/d/1Yj2IUO5r_JsIArNiAQBmkIKFSxCUHc2N/view?usp=sharing). 

We release a paired dataset: a randomized experiment and a parallel observational study on the
same population. We also attach here a reproducible notebook (`notebooks/01_main_results.ipynb`) demonstrating our
recommended best practices for observational causal estimation.

## Repository Structure

```
├── Data/
│   ├── README.md                            # Dataset documentation (44 columns)
│   ├── FINAL_PUBLIC_experimental.parquet    # Randomized A/B sample (435,170 obs)
│   └── FINAL_PUBLIC_observed.parquet        # Observational sample (445,286 obs)
├── notebooks/
│   └── 01_main_results.ipynb                # §4 main results: Figures 1–3, Table 2
├── src/                                     # Reusable Python modules
│   ├── data_loading.py                      # Data loading + covariate list
│   ├── propensity.py                        # FLAML-tuned LGBM propensity ensembles
│   ├── trimming.py                          # Crump et al. (2009) optimal trimming
│   ├── estimators.py                        # ATE estimators (Reg, OM, IPW, PSM, DR)
│   ├── ensemble_wrappers.py                 # AveragingRegressor / AveragingClassifier
│   ├── plotting.py                          # Figure generation helpers
│   ├── cache.py                             # Pickle load/compute helpers
│   ├── utils.py                             # Cross-fitting + ensembling utilities
│   ├── cate.py                              # CATE meta-learners (additional)
│   └── sensitivity.py                       # Sensitivity analysis (additional)
├── saved_outputs/                           # Pre-computed artifacts for fast replication
│   ├── prop_averaged_FLAML_FINAL_LGBM.pkl   #   observational propensity scores
│   ├── exp_dr_ate_FLAML_LGBM_Continuous.pkl #   experimental DR benchmark
│   ├── *_hyperparams.pkl                    #   FLAML-tuned LGBM hyperparameter dicts
│   └── *_ate_psm_pass_noreplace.pkl         #   cached PSM results (R Matching pkg)
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/microsoft/Reevaluating-Causal-Estimation-Methods.git
cd Reevaluating-Causal-Estimation-Methods
```

### 2. Install Python dependencies

```bash
python -m venv .venv
# Windows:    .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the main notebook

```bash
jupyter notebook notebooks/01_main_results.ipynb
```

With the default flags (`RERUN_ALL=False`, `RUN_PSM=False`), the notebook loads
pre-computed FLAML hyperparameters and cached PSM results from `saved_outputs/`,
cross-fits fresh LGBM nuisance models on the public data, and reproduces
**Figures 1–3** and **Table 2** in a few minutes on a laptop.

### 4. (Optional) Re-run from scratch

```python
RERUN_ALL = True      # full FLAML AutoML search (slow; minutes–hours)
RUN_PSM   = True      # recompute PSM via R (~30 min/call); see step 5
```

### 5. (Optional) Install R for propensity score matching

PSM uses R's [`Matching`](https://cran.r-project.org/package=Matching) package
(Sekhon 2011) via `rpy2`. PSM with `replace=FALSE, ties=FALSE` on the trimmed
sample (~300k rows) takes ~30 minutes per call. If R is not installed or
`RUN_PSM=False`, the notebook falls back to cached PSM results.

```bash
# 1. Install R: https://cran.r-project.org/
# 2. Install rpy2 + Matching:
pip install rpy2==3.6.7
Rscript -e 'install.packages("Matching", repos="https://cran.r-project.org")'
```

## What the Notebook Does

| Step | Description | Paper reference |
|------|-------------|-----------------|
| 1 | Load public experimental & observational datasets | §2 |
| 2 | Compute naive difference-in-means | §4 |
| 3 | Estimate propensity scores (ensembled, tuned LGBM) | §4 |
| 4 | Propensity score distributions | Figure 1 |
| 5 | Apply Crump et al. (2009) optimal trimming | §4, Table 2 |
| 6 | Establish experimental ground-truth benchmark | §4 |
| 7 | Fit tuned, ensembled nuisance models (cross-fit) | §4 |
| 8 | Estimate ATE with five methods | §4, Figure 2 |
| 9 | Compare trimmed vs. untrimmed results | §4, Figure 3 |

### Estimators

| Estimator | Method |
|---|---|
| **Reg** | OLS on `y ~ D + W` | 
| **OM** | Outcome modeling (cross-fit) |
| **IPW** | Inverse probability weighting (cross-fit) | 
| **PSM** | 1-NN propensity-score matching (R `Matching`) | 
| **DR** | Cross-fit AIPW via EconML `LinearDRLearner` | 

## Citation

```bibtex
@article{young2026reevaluating,
  title={Reevaluating Causal Estimation Methods with Data from a Product Release},
  author={Young, Justin and Dillon, Eleanor W.},
  year={2026},
  url={https://arxiv.org/abs/2601.11845}
}
```

## License

MIT. See [LICENSE](LICENSE).

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the
rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
