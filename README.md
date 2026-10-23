# Reevaluating Causal Estimation Methods with Data from a Product Release
**Justin Young and Eleanor W. Dillon (2026)**

This repository contains the benchmark datasets, replication code, and pre-computed artifacts for the paper
[*Reevaluating Causal Estimation Methods with Data from a Product Release*](https://drive.google.com/file/d/1Yj2IUO5r_JsIArNiAQBmkIKFSxCUHc2N/view?usp=sharing). 

We release a paired dataset: a randomized experiment and a parallel observational study on the
same population. We also attach here a reproducible notebook (`notebooks/01_main_results.ipynb`) demonstrating our
recommended best practices for observational causal estimation. The main analysis uses
the same eligibility rule in both sources: `Outcome_Continuous <= 44`, applied to both
actual treatment arms and to both outcomes.

## Repository Structure

```
├── Data/
│   ├── README.md                            # Dataset documentation (44 columns)
│   ├── FINAL_PUBLIC_experimental.parquet    # Randomized A/B sample (435,170 obs)
│   └── FINAL_PUBLIC_observed.parquet        # Observational sample (445,286 obs)
├── notebooks/
│   └── 01_main_results.ipynb                # Common-sample estimates and benchmark comparisons
├── src/                                     # Reusable Python modules
│   ├── data_loading.py                      # Data loading + covariate list
│   ├── main_cutoff_validation.py            # Actual-treatment analysis and saved-run validation
│   ├── placebo_validation.py                # Untreated-source placebo and shared nuisance helpers
│   ├── propensity.py                        # FLAML-tuned LGBM propensity ensembles
│   ├── trimming.py                          # Crump et al. (2009) optimal trimming
│   ├── estimators.py                        # ATE estimators (Reg, OM, IPW, PSM, DR)
│   ├── ensemble_wrappers.py                 # AveragingRegressor / AveragingClassifier
│   ├── plotting.py                          # Figure generation helpers
│   ├── cache.py                             # Pickle load/compute helpers
│   ├── utils.py                             # Cross-fitting + ensembling utilities
│   ├── cate.py                              # CATE meta-learners (additional)
│   └── sensitivity.py                       # Sensitivity analysis (additional)
├── saved_outputs/                           # Published artifacts; not inputs to common-sample runs
│   ├── prop_averaged_FLAML_FINAL_LGBM.pkl   #   observational propensity scores
│   ├── exp_dr_ate_FLAML_LGBM_Continuous.pkl #   experimental DR benchmark
│   ├── *_hyperparams.pkl                    #   FLAML-tuned LGBM hyperparameter dicts
│   └── *_ate_psm_pass_noreplace.pkl         #   cached PSM results (R Matching pkg)
├── requirements.txt
├── tests/                                   # Construction, inference, and real-model checks
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

Use Python 3.10 or newer and a Jupyter frontend with the environment above
selected as its kernel. If needed, install the notebook frontend separately:

```bash
pip install notebook
jupyter notebook notebooks/01_main_results.ipynb
```

The notebook fits fresh nuisance models and experimental benchmarks for both
outcomes. Its default bounded configuration uses three outer cross-fitting
folds, three-fold FLAML tuning within each training fold, 30 seconds per search,
at most 30 search iterations, and four worker threads. It uses **single selected
LightGBM models, not ensembles**, and takes approximately 20-30 minutes on the
full public data. Search budgets exclude final refits and overhead.

The main tables use the common analysis sample and a shared covariate-overlap
rule. The as-released sample is also fitted under the same model settings as
a sample-definition sensitivity. This is not an exact numerical replication
of the paper's ensembles or separate pooled-outcome EconML fit.

Every execution writes a new timestamped directory under
`saved_outputs_main_common_cutoff`. Existing results are not overwritten.
The notebook does **not** load the published propensity arrays, nuisance
hyperparameters, experimental benchmarks, or matching results from
`saved_outputs`, because fitted quantities must correspond to the eligible sample.

### 4. Command-line execution and validation

From the repository root:

```bash
python -m unittest discover -s tests -p "test_*validation.py" -v
python -m src.main_cutoff_validation --continuous-upper 44 --time-budget 30 --jobs 4
```

The analysis checks source data, actual treatment labels, sample eligibility,
fold coverage and train/test separation, saved-model predictions, overlap
masks, and reported estimates, standard errors, confidence intervals and
p-values. It archives the exact source and records input/code hashes and
package versions. Recheck a completed run without refitting:

```bash
python -m src.main_cutoff_validation --verify-output <run-directory>
```

If the working code has changed, run this command from the completed run's
`source_snapshot` directory with `--verify-output ..` instead. The source-hash
check is not bypassed.

### 5. Appendix A control-sample placebo

```bash
python -m src.placebo_validation --continuous-upper 44 --time-budget 30 --jobs 4
```

This separate check uses only actually untreated devices in both sources,
assigns a new pseudo-treatment indicator to experimental controls, and estimates
experimental-minus-observational outcome contrasts. It fits new source
propensity models and applies the same eligibility rule to both outcomes.
Results are saved separately under `saved_outputs_placebo`. A null contrast
does not establish equivalence or unconfoundedness of actual adoption.

## Sample definition and overlap

The public file counts in the repository structure describe the released files,
not the sample after analysis eligibility. Both released files already
incorporate within-source continuous-outcome truncation. The common rule
retains `Outcome_Continuous <= 44` and drops rows above it; it does not
winsorize values or recompute a percentile. On the current files, it retains
all experimental rows and removes observational rows from both actual
treatment arms. The notebook reports the exact counts.

This outcome-based eligibility is separate from Crump propensity-overlap
selection. For the primary shared-region comparison, the observational
adoption-propensity model is evaluated on both sources, and the same interval
is applied to both. **Experimental IPW/AIPW weighting uses experimental
treatment probabilities**, not observational adoption probabilities.
Full-sample and cohort-specific propensity-region results are also reported;
the latter need not identify the same covariate region across sources.

Outcome-dependent selection can change the estimand and compromise
randomization. Harmonized eligibility does not recover excluded observations
or identify an untruncated population effect. Confidence intervals are
pointwise, assume independent rows and appropriate nuisance/overlap
conditions, and do not include repeated model-selection or learned-boundary
uncertainty. A shared region is not exact covariate-distribution reweighting.

### Estimators

| Estimator | Method and inference |
|---|---|
| **Reg** | OLS on `y ~ D + W`; HC1 robust covariance |
| **OM** | Cross-fitted treatment-arm models; residual-MSE SE approximation |
| **IPW** | Unnormalized inverse-probability score; plug-in score SE |
| **DR** | Direct cross-fitted AIPW sharing OM nuisances; score SE |

Propensity tuning minimizes classification log loss; both outcome models
minimize regression RMSE. Actual-treatment DR explicitly clips propensity
to `[0.01, 0.99]` and retains an unclipped diagnostic. No probability-calibration
claim is made for the RMSE binary regressors. OM and IPW intervals have additional
non-orthogonal nuisance-estimation limitations.

PSM requires a separate analysis with R's `Matching` package. The bounded
Python workflow checks its availability and reports PSM as **not run**;
it never substitutes an unrelated cached estimate or fabricated matching SE.

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
