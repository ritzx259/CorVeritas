# CorVeritas Stage 2

Independent multi-label classification of **self-reported previous diagnoses**
using CDC NHANES August 2021–August 2023. Each participant receives three scores:
reported coronary heart disease, congestive heart failure and previous myocardial
infarction. Outputs are independent sigmoids, may overlap and need not sum to one.

This research prototype estimates cross-sectional associations. It does not
predict future onset, supply a validated diagnosis or replace medical assessment.
All code, downloaded files and generated artifacts for this stage live here.
Stage 1 and the existing frontends remain separate; Stage 1 is never a gate.

## Setup and commands

Python **3.11**, macOS arm64/Intel, Linux or Windows. CPU works without CUDA;
device selection also supports CUDA and Apple MPS when available. Commands below
start in the CorVeritas workspace root. No API key or paid service is required.

```bash
python3.11 -m venv models_stage2/.venv
source models_stage2/.venv/bin/activate
python -m pip install -r models_stage2/requirements.txt

python models_stage2/download_data.py
python models_stage2/preprocess.py
python models_stage2/train.py
python models_stage2/evaluate.py
python models_stage2/explain.py --samples 20
python models_stage2/predict.py --input models_stage2/example_patient.json

python -m pytest -c models_stage2/pytest.ini models_stage2/tests --basetemp=models_stage2/tests/.tmp
```

Windows activation: `models_stage2\.venv\Scripts\activate`. Alternatively, use
package invocation from the workspace root, e.g. `python -m models_stage2.train`.
Paths are resolved relative to the source directory, not the working directory.

`train.py` performs test evaluation and explanations automatically after saving
the validation-selected models. Subsequent `evaluate.py` reads the cached report
and does not rescore the test set. `--skip-test` defers the first evaluation;
`--skip-explain` defers explanation generation. Separate evaluation and explanation
commands are useful for these deferred steps.

```bash
# Enforce CPU, or shorten training for a software smoke test (not a final experiment).
python models_stage2/train.py --device cpu --epochs 5 --patience 3 --model-dir models_stage2/models/smoke

# New experiments must use a fresh directory. Existing artifacts are never overwritten.
python models_stage2/train.py --model-dir models_stage2/models/experiment2
python models_stage2/evaluate.py --model-dir models_stage2/models/experiment2
python models_stage2/predict.py --model-dir models_stage2/models/experiment2 --input models_stage2/example_patient.json

# Use previously downloaded official local files; no network requests.
python models_stage2/download_data.py --offline
# An alternate raw directory can be read during preparation.
python models_stage2/preprocess.py --raw-dir models_stage2/data/local_cdc_copies

# Optional third-party SHAP summaries; built-in grouped SHAP needs no extra package.
python -m pip install 'shap>=0.46,<0.50'
python models_stage2/explain.py --samples 20 --shap
```

The executed Python 3.11 environment is isolated inside `models_stage2/.venv`.
To avoid slow downloads, cached PyTorch wheels and read-only copies of existing
scientific packages were used during setup; those dependencies were materialized
inside Stage 2 and no existing environment was modified. The setup above creates
a conventional standalone environment. `requirements-executed.txt` records the
exact dependency closure of the executed macOS arm64 run. Keep these versions
with artifacts; sklearn/joblib serialization is version-sensitive. The actual
session status is recorded in [results/EXECUTION.md](results/EXECUTION.md).

## Directory layout

```text
models_stage2/
  config.py                 Fixed schema, ranges, labels, defaults
  download_data.py          Official XPT downloader, cache and schema/codebook audit
  preprocess.py             Adult cohort, recoding, splitting and sklearn pipeline
  model.py                  PyTorch network and device/seeding helpers
  train.py                  Baseline/MLP search, early stopping, artifact saving
  evaluate.py               EDA, once-only test metrics and report plots
  explain.py                Exact grouped SHAP, permutation importance, optional SHAP
  predict.py                Frontend-friendly inference API and CLI
  artifacts.py              Loading, checksums, JSON utilities
  CODEBOOK.md               Variable eligibility, coding and leakage review
  data/raw/                 Seven XPTs, codebook HTML and download_manifest.json
  data/processed/cohort.csv  Complete-label adults, raw clinical features, audit metadata
  models/                   Fitted artifacts and participant partitions
  results/                  EDA, audit reports and execution record
  results/plots/            Descriptive figures and <run_id>/ model/explanation plots
  results/runs/<run_id>/     Validation search, test report and explanation data
  tests/                    Artificial fixtures for software integrity checks only
```

## Dataset and target construction

The [official CDC directory](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/)
provides `MCQ_L`, `DEMO_L`, `BMX_L`, `BPXO_L`, `TCHOL_L`, `DIQ_L`, `SMQ_L`.
`download_data.py` uses HTTPS with retries/timeouts, validates SAS transport
headers and required columns, rejects duplicate/missing `SEQN`, caches codebooks
and records SHA-256 hashes. A corrupt cached file raises a descriptive error;
replace it with a fresh official copy. Downloads are written to `.part` files
and renamed only after validation. Offline mode accepts the same XPT filenames;
codebook snapshots are optional in that mode, with this documented verification
as the reference. See [CODEBOOK.md](CODEBOOK.md) for the seven official links.

All modules are left-joined one-to-one onto demographics using `SEQN`. The cohort
keeps age 20+ with three valid diagnosis answers. Code 1 becomes positive and
code 2 negative; 7, 9, missing and any unexpected code remain missing. Participants
with any missing target are excluded and counted in `cohort_audit.json`.
Diagnosis combinations remain intact. This is complete-label analysis, not a
masked-label method. Missing clinical inputs remain available for imputation.
Interview-only adults are retained and counted; the resulting measurement
missingness and selection bias require explicit discussion in a report.

| Label | Official variable |
|---|---|
| `coronary_heart_disease` | `MCQ160c` |
| `heart_failure` | `MCQ160b` |
| `heart_attack_history` | `MCQ160e` |

## Clinical input contract

| Input key | Unit / accepted values | Source |
|---|---|---|
| `age` | Years, required, 20–120; encoded at maximum 80 | `RIDAGEYR` |
| `bmi` | kg/m², 10–100 | `BMXBMI` |
| `systolic_bp` | mmHg, 50–300 | Mean of valid `BPXOSY1..3` |
| `diastolic_bp` | mmHg, 20–200, below systolic when both supplied | Mean of valid `BPXODI1..3` |
| `total_cholesterol` | mg/dL, 40–1000 | `LBXTC` |
| `sex` | Male, Female, Unknown | `RIAGENDR` |
| `diabetes_history` | Yes, No, Borderline, Unknown | `DIQ010` |
| `smoking_history` | Yes, No, Unknown, or documented smoker aliases | `SMQ020` |

All fields except age may be omitted or null. Bounds are broad engineering
guards, not clinical normal ranges. Invalid observed data becomes missing with
audit counts; invalid supplied values raise errors. Valid numeric strings are
accepted. Unsupported categories, unexpected keys, booleans, nonfinite values,
inconsistent BP and ineligible ages are rejected. Missing age is rejected to
confirm adult eligibility. All-missing input columns are kept during fitting.

Smoking means **100+ lifetime cigarettes**, not smoking now. Former/current
smoker aliases map to Yes with a warning to verify the lifetime threshold.
No means fewer than 100 cigarettes; borderline diabetes remains distinct from
No. Pregnancy-only diabetes is not a Yes under the source question definition.
NHANES has no UCI `ca`, `thal`, `oldpeak` or `cp` predictors in this pipeline.

## EDA, splitting and preprocessing

Before training, preparation saves cohort shape/dtypes, source/merge audit,
feature missingness, all-feature summary statistics, positive/negative support,
label combinations, number with multiple diagnoses, distributions, numerical
Pearson correlations and imbalance plots. These are unweighted descriptive
figures. Fixed feature choices and cleaning rules are codebook-driven, not
selected from test outcomes.

Participants are split approximately 70%/15%/15%, seed 42, using iterative
multi-label stratification. Positive label pairs augment the stratification
targets to encourage co-occurrence preservation. Integer rounding can slightly
change fractions. Without that dependency, the fallback stratifies the eight
label combinations when counts permit and otherwise uses a seeded random split.
Every partition's prevalence, counts and combinations are saved and printed.
Missing classes cause a clear error instead of an unsupported metric.
IDs cannot overlap; no resampling or synthetic training data is used.

A sklearn `ColumnTransformer` fits exclusively on training inputs: median
numeric imputation, standard scaling, categorical Unknown imputation and fixed
one-hot categories. Explicit categories come from the codebooks; they are not
learned from held-out data. Raw and transformed ordering are saved. Missingness
indicator features are deliberately omitted to reduce shortcuts based on exam
participation. The exact saved transformer is used for both models and inference.

## Models and selection

The baseline consists of three independent logistic regressions, searching
`C ∈ {0.1, 1, 10}` and unweighted/balanced classes. The MLP has transformed input,
128-unit dense/ReLU/LayerNorm/dropout 0.3, 64-unit dense/ReLU/dropout 0.2 and three
output logits. LayerNorm is used instead of BatchNorm so small or singleton
batches work reliably. Sigmoid is applied only for inference/scoring.

Neural training uses `BCEWithLogitsLoss`, AdamW (weight decay 0.01), gradient
norm clipping at 1, seeded data shuffling, ReduceLROnPlateau and early stopping
on **validation macro average precision** (patience 15, min improvement 1e-5).
Defaults: batch 32, maximum 100 epochs. Search compares learning rates 0.001 and
0.0003, each with/without per-label positive weights. Weights equal
`training_negative / training_positive`; validation and test counts never enter
them. Weighted scores may have poor probability calibration.

Each model family's best validation AP candidate is saved. The higher
validation macro AP selects the recommended model (ties favor the baseline).
For each family and condition, the threshold maximizes validation F1; ties favor
the higher threshold. This rule is a research tradeoff and does not represent a
clinical sensitivity requirement or a cost-sensitive clinical operating point.
Thresholds, best epochs, hyperparameters and all candidate results are recorded.
Validation reuse for search, early stopping and thresholds can overfit it; an
independent external validation dataset is needed before deployment.

Random seeds and deterministic algorithms are requested, but numerical results
can vary with hardware, library versions and unsupported MPS/CUDA deterministic
operations. CPU with a pinned environment is preferred for strict reproduction.

## Evaluation and report artifacts

Both selected family representatives are scored once on the untouched test
partition after model choice and thresholds are fixed. `test_evaluation.json`
is cached with the run ID; rerunning evaluation returns that report. Predictions
are saved so figures and tables can be reused without rescoring or retuning.
Do not remove the cache and repeatedly compare experiments on the same test
set; new research iterations require a justified held-out evaluation design.

For each condition and model the report contains precision, sensitivity,
specificity, F1, ROC-AUC, **average precision** and **trapezoidal PR-AUC** (different
summaries of the PR curve), positive/negative support, confusion matrix,
Brier score, binary log loss and 10-bin expected calibration error. Probability
reliability plots use quantile bins. Zero-division precision/recall/F1 is 0;
unavailable AUCs are null. Brier/log loss/ECE evaluate uncalibrated scores; no
post-hoc calibration is fitted and outputs are never called calibrated
probabilities.

Age groups 20–39/40–59/60+ and recorded Male/Female sex are evaluated only for
conditions with at least 30 subgroup participants and 5 examples of each class.
Insufficient cells are explicitly flagged with their support. Sparse subgroups,
few positive cases and unstable calibration bins prevent strong fairness or
clinical conclusions. No population confidence intervals are claimed.

`results/plots/` contains EDA figures. `results/plots/<run_id>/` contains
loss/AP curves for each neural candidate, per-model ROC and PR curves,
confusion matrices and calibration figures. Comparable train/validation curves
use unweighted BCE; weighted optimization loss is separately recorded in CSV.
`model_comparison.csv`, `candidate_comparison.csv`, split reports and per-label
JSON metrics support academic reporting. Actual results depend on executing
the pipeline; no fixed accuracy is promised.

## Explainability

For eight raw inputs, exact grouped interventional Shapley values require only
256 coalitions. Each is averaged over 16 sampled **training** background rows.
One-hot columns remain grouped under their clinical input. Contributions are
in model-score units and add to the prediction when added to the baseline;
reconstruction error is returned. “Exact” refers to this finite-background
game, not to a population/conditional or causal explanation.

The default uses 20 random validation participants for global mean absolute
SHAP, a summary scatter plot and saved individual contributions. Sampling makes
global summaries approximate. Reusable artifacts include JSON, CSV, NPZ, bar
plots and human-readable descriptions. Raw grouped permutation importance
reports validation AP decrease over five repeats, independent of SHAP. Negative
permutation importance can occur. Correlated variables and artificial feature
coalitions can produce implausible combinations; importance is not causal.
To avoid changing the game, internal artificial coalitions are not revalidated
as real clinical patients.

`--shap` additionally uses the optional SHAP library's sampled permutation
explainer on transformed inputs and generates beeswarm plots per condition.
These encoded-feature plots can distribute importance across one-hot columns;
the built-in grouped explanation is preferable for the frontend. Explanations
do not train models, use test labels or change model selection.

## Frontend inference

```python
from models_stage2.predict import Stage2Predictor

# Load once at application startup. No training, downloads or test evaluation.
stage2 = Stage2Predictor()
result = stage2.predict({
    "age": 55, "sex": "Male", "bmi": 28.4,
    "systolic_bp": 145, "diastolic_bp": 90,
    "total_cholesterol": 240, "diabetes_history": "Yes",
    "smoking_history": "Former smoker",
})
# result is JSON-compatible. A FastAPI/Flask route can return it directly.
# stage2.predict(patient, explain=False) avoids coalition work for low latency.
```

`predict_patient(...)` offers the same interface with one-shot artifact loading.
The result returns model version/run ID, three condition names, actual saved
model scores, validation-selected thresholds, binary classifications,
feature contributions, explanation baselines and limitations. No scores are
hardcoded. `model_name="baseline"` or `"neural"` can override the saved family
selection for research comparisons. The default model remains the one chosen
using validation data. Inference always runs on CPU for portability.

Saved artifacts: `neural.pt`, `baseline.joblib`, `preprocessor.joblib`,
`explanation_background.joblib`, `splits.csv`, `metadata.json`. Metadata includes
raw/transformed names, labels, thresholds, neural architecture, hyperparameters,
validation metrics, software versions, data/partition hashes and artifact
checksums. Test metrics and explanations are referenced by run result directory.
Only load trusted local joblib artifacts; their format can execute Python.
Checksums detect accidental mismatch, not malicious tampering.

## Limitations and academic interpretation

NHANES is cross-sectional and reports previous diagnoses, with recall,
healthcare-access, survivor and ascertainment biases. Absence of a reported
diagnosis does not establish absence of disease. Measurements and lifestyle
may follow diagnosis and treatment; low cholesterol after treatment cannot be
interpreted as a protective cause. Restricting to complete labels can introduce
selection bias; many retained adults lack exam/lab inputs. Imputation does not
recover their unobserved measurements or remove missingness bias.

The pipeline is an **unweighted sample prediction experiment**, not an official
NHANES survey analysis or an estimate of US population disease prevalence.
CDC's phlebotomy weight `WTPH2YR` is relevant for analyte population analyses;
the MEC weight alone would not settle this issue. Survey weights, strata and
PSUs are retained for audit, excluded from predictors and not used to produce
population estimates or design-based uncertainty. Participant random splits
also do not guarantee independence at household or survey-cluster level.

Few positives and overlapping labels limit precision, calibration and subgroup
stability. Age is top-coded at 80, sex is limited to the categories in the source,
smoking omits other tobacco exposures, and diabetes chronology is unknown.
US survey performance cannot be assumed to transfer to Indian clinical patients
or another frontend population. There is no external, prospective or clinical
validation, no future time horizon and no clinically established threshold.
Stage 1 and Stage 2 raw records are never combined. Their scores describe
different datasets and target definitions and should be presented separately.

Software tests use small artificial fixtures solely to check recoding,
participant separation, preprocessing leakage, thresholding, save/load,
input validation, neural backpropagation and SHAP additivity. Fixture metrics
are not research results and fixtures never enter the NHANES training cohort.
