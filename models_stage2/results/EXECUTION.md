# Execution record — 2026-10-23 (Asia/Calcutta)

All source and artifact changes for this task are inside `models_stage2/`.
The pre-existing empty `models_stage 2/model_stage2_1.py`, Stage 1 models and
frontend source were left untouched. No frontend integration was performed.

## Completed against real CDC data

- Read the seven official August 2021–August 2023 codebooks; recorded coding,
  eligibility, unit and temporal-leakage decisions in [CODEBOOK.md](../CODEBOOK.md).
- Downloaded all seven official XPTs and seven HTML codebooks using the Python
  downloader. Validated transport format, schema and participant uniqueness.
- Prepared 7,745 complete-label adults from 7,809 eligible adults. Excluded
  64 adults with at least one invalid/missing diagnosis response. Excluded
  4,124 nonadult/missing-age demographic records before label selection.
- Positives: CHD 391, CHF 324, previous heart attack 316. Multiple positive
  diagnoses: 275 participants. These are unweighted sample counts.
- Missing inputs: BMI 1,817; systolic BP 1,923; diastolic BP 1,923;
  cholesterol 2,283; diabetes 2; smoking 27. Age and sex were complete.
  Retained 1,725 interview-only participants; imputation does not remove
  exam-participation or missingness bias.
- Iterative stratification with seed 42 produced 5,421 training, 1,162
  validation and 1,162 test participants, with no participant overlap.
- Ran all six logistic settings and all four neural settings with default
  maximum 100 epochs, batch 32 and early-stopping patience 15 on CPU.
- Neural selected by validation macro AP: **0.210745**, versus baseline
  **0.198771**. Selected MLP: learning rate 0.001, unweighted loss, epoch 21.
- Saved both family finalists and the exact fitted preprocessing pipeline.
- Evaluated both family finalists once on the untouched test partition.
  Later calls to `evaluate.py` used cached results without test rescoring.
- Saved 20 PNG plots across EDA, learning curves, test evaluation and XAI.
- Generated exact grouped Shapley explanations for 20 validation participants,
  global importance, summary plots, five-repeat validation permutation
  importance and machine-readable explanation artifacts.
- Ran example-patient inference with explanations and without explanations;
  both script and package imports work. Predictions came from saved artifacts.
- Revalidated all seven cached XPT modules using downloader `--offline`.

## Genuine test results

Run ID: `20261023T181032Z-cc7796d8`.

| Selected neural model | Validation threshold | Test ROC-AUC | Test average precision | Test F1 | Test Brier |
|---|---:|---:|---:|---:|---:|
| Reported CHD history | 0.148300 | 0.863861 | 0.210263 | 0.331551 | 0.043172 |
| Reported CHF history | 0.133731 | 0.848892 | 0.185715 | 0.256757 | 0.037009 |
| Reported previous MI | 0.154149 | 0.870556 | 0.224292 | 0.294574 | 0.034501 |

Test positives are only 59, 49 and 47, respectively, among 1,162 participants.
Test precision at the selected neural thresholds is approximately 0.242,
0.192 and 0.232, so positive classifications have many false positives.
These results do not establish clinical validity. The baseline has higher
test average precision for CHD and CHF; the recommended family remains the
one selected using validation, without changing it after seeing test results.

The complete [comparison report](runs/20261023T181032Z-cc7796d8/report.md),
[metrics and subgroup JSON](runs/20261023T181032Z-cc7796d8/test_evaluation.json),
[comparison CSV](runs/20261023T181032Z-cc7796d8/model_comparison.csv),
[plots](plots/20261023T181032Z-cc7796d8/), and
[example prediction](example_prediction.json) contain the actual saved outputs.

## Software verification

- Stage 2 suite: **21 passed**, including target missingness, adult eligibility,
  BP averaging, categorical encoding, training-only imputation, reproducible
  disjoint splits and fallback, threshold selection, specificity, strict inputs,
  HTML-download rejection, baseline/neural artifact loading, corruption detection,
  exact SHAP additivity, cached evaluation and neural backpropagation.
- Existing `frontend_corveritas2/backend/tests` suite: **35 passed**, including
  actual Stage 1 saved-artifact compatibility and predictions. Tests were run
  with bytecode disabled for existing source and temporary/cache locations
  redirected into Stage 2.
- Python compilation and `pip check` passed. Third-party warnings concern
  deprecated matplotlib/pyparsing, sklearn/SciPy and TensorFlow behavior;
  the intentional fallback test emits its documented splitter warning.
- Artificial fixtures were used only for software tests. They were never
  added to the real cohort or used for the reported model metrics.

## Environment and execution limits

Python 3.11, PyTorch 2.11.0, sklearn 1.6.1, NumPy 2.0.2, pandas 2.3.3,
matplotlib 3.10.6, iterative-stratification 0.1.9. See
[exact dependency versions](../requirements-executed.txt). The Stage 2 virtual
environment is inside this folder; existing package files were copied read-only
and a cached compatible PyTorch wheel was installed to avoid slow large network
downloads. The original environments were not modified.

Initial network attempts failed under the sandbox. Approved network retries
successfully downloaded the official CDC files and the small missing dependency.
Interrupted dependency downloads were replaced with compatible cached packages.

CUDA and MPS were unavailable in the execution context, so accelerator paths
were not exercised. The optional third-party `--shap` path was not executed;
the built-in exact grouped Shapley path was executed and tested. No probability
calibration was fitted. No prospective/external clinical validation, survey
population estimation, frontend tests or frontend integration was performed.

The models describe associations with self-reported **previous diagnoses**.
Their uncalibrated scores are not future disease forecasts or medical diagnoses.
