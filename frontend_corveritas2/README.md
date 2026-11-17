# CorVeritas

**An Explainable AI-Powered Cardiovascular Disease Risk Assessment and Clinical Decision-Support System**

CorVeritas is a locally runnable B.Tech AI & ML research application with a React/TypeScript dashboard and a FastAPI inference service. It connects to the real fitted MLP, 1D CNN and Random Forest artifacts already present in this repository. All new application files live inside `frontend_corveritas2/`; original source scripts, CSV and existing models are read without modification.

Every current model predicts **binary heart-disease presence versus absence as recorded in the training dataset**. `stage1`, `stage2` and `stage3` are historical artifact filenames, not disease stages or disease-subtype classifiers. No subtype diagnosis is implemented by Stage 1. The separate Stage 2 extension reports overlapping previous-diagnosis history scores, as described below.

> Research and educational use only. These models are not clinically validated medical devices. A prediction neither establishes nor excludes a diagnosis. A model score is an ML estimate and may not be a calibrated clinical probability. Qualified healthcare professionals must interpret measurements and clinical findings. Acute concerning symptoms require immediate medical attention regardless of the model output.

## Local setup in VS Code on macOS

Open the existing **CorVeritas repository** in VS Code. The commands below are project-relative. Use Python **3.11** and Node.js **22**. The repository's older parent `.venv` uses Python 3.9; do not replace it or assume it has compatible packages. This application has its own environment.

In the integrated terminal, starting at the repository root:

```bash
cd frontend_corveritas2
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
cd frontend
npm ci
cd ..
```

If `python3.11` is unavailable, install Python 3.11 using your normal macOS package manager or Python.org installer first. Dependency installation requires internet access. TensorFlow is a substantial download. The requirements pin TensorFlow, Keras, scikit-learn, NumPy, Pandas and API/test dependencies; the frontend manifest and lockfile pin its dependency tree. Do not arbitrarily upgrade scikit-learn when using existing serialized artifacts.

For the exact tested Python 3.11 macOS arm64 environment, [backend/requirements-lock.txt](backend/requirements-lock.txt) also records the full installed dependency set. On the same platform it can be installed using `python -m pip install -r backend/requirements-lock.txt` instead of the shorter requirements file.

Start the backend in **terminal 1**, from `frontend_corveritas2/`:

```bash
source .venv/bin/activate
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Start the frontend in **terminal 2**, from the repository root:

```bash
cd frontend_corveritas2/frontend
npm run dev
```

Open:

- Application: <http://127.0.0.1:5173>
- Interactive API documentation: <http://127.0.0.1:8000/docs>
- Backend readiness: <http://127.0.0.1:8000/api/health>
- Model availability and load diagnostics: <http://127.0.0.1:8000/api/models>

The Vite server uses a strict port of 5173 and proxies `/api` to FastAPI on 127.0.0.1:8000. Wait for API startup/model loading to complete before submitting an assessment. Port conflicts produce an error instead of silently selecting a different port. Stop each development server using Ctrl+C.

After installing both dependency sets, a convenience launcher is available from `frontend_corveritas2/`:

```bash
python gui2.py
```

It starts both processes and stops them together on Ctrl+C. It uses this folder's `.venv` when present, verifies required imports and frontend dependencies, and does not install dependencies automatically. The manual two-terminal commands remain the clearest way to investigate startup errors.

## Optional configuration

Defaults require no `.env` file. To change them, from `frontend_corveritas2/`:

```bash
cp .env.example .env
# Edit .env in VS Code, then load it into this terminal:
set -a
source .env
set +a
```

Start the backend or launcher from that same terminal. This is a shell environment file; the application does not depend on automatic `.env` discovery.

| Variable | Default | Purpose |
| --- | --- | --- |
| `CORVERITAS_DEFAULT_MODEL` | `random_forest` | Preferred available model in the dashboard |
| `CORVERITAS_MODEL_DIR` | Unset | Restrict loading to one trusted directory; relative paths resolve from `frontend_corveritas2/` |
| `CORVERITAS_THRESHOLD_MLP` | `0.5` | MLP decision threshold |
| `CORVERITAS_THRESHOLD_CNN` | `0.5` | CNN decision threshold |
| `CORVERITAS_THRESHOLD_RANDOM_FOREST` | `0.5` | Random Forest decision threshold |
| `CORVERITAS_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated allowed development origins |

Thresholds are configuration values, not recommendations or clinically validated operating points. Restart the backend after changing configuration or artifacts. A request specifying an unavailable model returns an error; the API never substitutes another model for that request.

## Existing files and model discovery

The original workspace contains:

```text
data/heart_disease_uci.csv
models/training_models.py
models/training_model2.py
models/training_model3.py
models/corveritas_stage1.keras
models/preprocessor.pkl
models/corveritas_stage2_cnn.keras
models/preprocessor_stage2_cnn.pkl
models/corveritas_stage3_random_forest.joblib
```

The application first checks complete artifact sets in its own `frontend_corveritas2/models/`, then uses the existing repository `models/` directory read-only. `CORVERITAS_MODEL_DIR` restricts discovery to the specified directory. Only trusted local serialized files should be loaded. Pickle/Joblib files can execute code during deserialization.

| API model ID | Required files | Inference |
| --- | --- | --- |
| `mlp` | `corveritas_stage1.keras` + `preprocessor.pkl` | Matching preprocessor → floating-point feature matrix → dense network |
| `cnn` | `corveritas_stage2_cnn.keras` + `preprocessor_stage2_cnn.pkl` | Matching preprocessor → `(samples, processed_features, 1)` → Conv1D network |
| `random_forest` | `corveritas_stage3_random_forest.joblib` | Raw DataFrame → complete fitted pipeline's `predict_proba` |

The Random Forest artifact already contains preprocessing. Its standalone `preprocessor_stage3_random_forest.pkl` is not needed for inference. Positive-class probability is selected using the fitted class ordering, not an assumed column index. Availability requires successful loading and compatibility checks, not merely a filename match. The API exposes the effective paths and any load failure; see the model page before relying on a demonstration.

If a model is missing, inspect the original script and run the matching portable launcher command from `frontend_corveritas2/`:

```bash
source .venv/bin/activate
python training/train.py --model mlp --show-plan
python training/train.py --model mlp
python training/train.py --model cnn
python training/train.py --model random_forest
```

Run only the model you need. Training can be slow, particularly RandomizedSearchCV. The wrapper modifies only `ROOT_DIR`, `DATA_PATH` and `MODEL_DIR` assignments in an in-memory copy of the appropriate original source. It leaves its algorithms, splits and tuning logic unchanged. It sets noninteractive plotting, redirects caches and stages outputs under this application, then publishes successful artifacts and plots to this folder's `models/`. Originals remain untouched. It refuses to overwrite a previously trained local set unless `--overwrite` is supplied. Stop/restart the API when replacing artifacts. Existing artifacts do **not** need retraining to use the dashboard.

## Disease Type Classification — Stage 2

The unchanged four-step assessment still runs the selected Stage 1 binary model.
The results page now adds a separate **Stage 2 — Disease Type Classification**
card. After a positive Stage 1 result, choose **Classify Disease Type** to use the
trained artifacts in `../models_stage2/models/`. Compatible measurements are
reused; only missing Stage 2 inputs are offered. Unknown optional values use
Stage 2's own fitted preprocessing. Ages 18–19 remain supported by Stage 1 but
cannot run the 20+ Stage 2 model.

The actual targets are **reported coronary heart disease history**, **reported
congestive heart failure history**, and **reported previous myocardial
infarction**. They are independent, overlapping diagnosis-history scores, not
one exclusive disease subtype or an acute diagnosis. All three scores, saved
thresholds and supported grouped Shapley explanations are displayed separately.
A negative Stage 1 result does not run Stage 2 or guarantee absence of disease.

Stage 2 uses its existing `models_stage2/.venv` in an isolated CPU worker. The
existing launch command and Stage 1 environment/dependencies are unchanged.
Missing Stage 2 files or failures leave Stage 1 results intact. See
[the inspected schema, API, configuration and verification contract](docs/STAGE2_INTEGRATION.md).

## Assessment and inference workflow

1. The dashboard obtains `/api/health`, `/api/models` and `/api/schema` from the real backend. Availability controls model selection.
2. React Hook Form and Zod validate a four-step assessment: demographics; symptoms and reported results; clinical measurements; review and model choice. Age, sex and chest-pain category are required. Unknown optional values remain `null`.
3. The UI submits `POST /api/predict` with `model_id` and the original 13 feature names. Yes/No selections become JSON Booleans. “Reversible defect” is transmitted as the encoder's exact spelling `reversable defect`.
4. Pydantic independently rejects invalid categories, invalid types, unexpected fields and out-of-range inputs. The service creates a single-row DataFrame in verified training column order, preserving categorical/Boolean types and converting missing inputs to NumPy `NaN`.
5. The already-loaded selected model's fitted pipeline transforms the row. No fitting, tuning, CSV evaluation or training occurs during requests. A missing value uses the fitted training median or most-frequent category; arbitrary defaults are not inserted.
6. The model's actual output score is compared with its configured threshold. The response includes the binary class, model score, threshold, missing fields, supported explanation status and disclaimer. The result page renders this response and the submitted input summary.

Input columns, in order:

```text
age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang,
oldpeak, slope, ca, thal
```

`ca` is an integer 0–3 or null; `oldpeak` may be negative, as in the actual CSV. Fasting blood sugar is a **test threshold result**, not a symptom. ECG, slope, vessel imaging and thal categories require clinical tests; users should select Unknown rather than guess. No names, contacts or patient identifiers are requested.

## Verify a real prediction

With the backend running, inspect model readiness:

```bash
curl -s http://127.0.0.1:8000/api/health
curl -s http://127.0.0.1:8000/api/models
```

Then send this complete example directly to the API:

```bash
curl -s -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"random_forest","features":{"age":63,"sex":"Male","cp":"typical angina","trestbps":145,"chol":233,"fbs":true,"restecg":"lv hypertrophy","thalch":150,"exang":false,"oldpeak":2.3,"slope":"downsloping","ca":0,"thal":"fixed defect"}}'
```

The score must come from the loaded pipeline; no expected prediction is hardcoded here. Change `model_id` to `mlp` or `cnn` to exercise their respective saved artifacts. Set an optional value such as `"chol":null` to verify missing-value reporting. An invalid category such as `"cp":"other"` must return HTTP 422. An unavailable explicitly requested model must fail clearly.

In the browser, open Developer Tools → Network, submit the review step and inspect the `/api/predict` request and JSON response. The displayed model, score and threshold must match that response. The automated artifact-compatibility checks independently compare application inference with direct calls to the original fitted artifacts. They verify integration rather than choosing a model from repeated test-set evaluation.

For an independent end-to-end check, keep FastAPI running and execute from `frontend_corveritas2/` in another activated terminal:

```bash
source .venv/bin/activate
python scripts/verify_prediction.py --output verification/inference-check.json
```

This script loads the trusted artifacts separately and compares their direct output against actual HTTP `/api/predict` responses, for both a complete synthetic record and a record with every optional feature missing. The optional JSON report records integration-check metadata, not patient inputs. It does not evaluate model accuracy on the held-out dataset or train/tune any model. Without `--output` it prints the check results without creating a report. The `--model-dir` option follows the configured model-directory override.

The delivered [inference-check report](verification/inference-check.json) confirms that **MLP, CNN and Random Forest all loaded successfully** from the existing trusted artifacts. Each passed both complete-input and optional-values-missing HTTP comparisons with **zero observed difference** from direct artifact inference. This verifies the application integration on those synthetic cases; it is not an accuracy estimate or clinical validation.

The independent [frontend proxy report](verification/frontend-proxy-check.json) repeats those six checks through the actual Vite server on port 5173, again with zero observed difference. This confirms that the frontend development server's `/api` proxy reaches the fitted models.

## Explainability and model comparison

Random Forest local explanations use the **actual trees' probability changes along the patient's decision paths**. Each tree starts at its root positive-class probability; each traversed split contributes its child's probability minus its parent's. Contributions are averaged over the forest and summed back to the original clinical fields, including one-hot categories. The implementation checks that baseline plus all contributions reconstructs the actual positive-class score within numerical tolerance.

This is a tree-path contribution decomposition, **not SHAP**. It depends on learned split ordering and is descriptive, not causal. It is not a treatment recommendation or a decomposition of clinical risk. For missing inputs it explains the fitted imputed values, not an observed measurement. Global Random Forest impurity importance is shown separately, grouped by original fields; it describes the fitted forest generally and has known correlation/category-count biases. It must not be confused with a patient's local explanation.

Trustworthy neural-network local explanations are not implemented; those results clearly state **“Explanation currently unavailable.”** No medical narratives or numerical contributions are fabricated from input rules.

The comparison page reports real availability, architectures, input requirements and explanation support. It displays the existing training-history and evaluation PNGs and reads numeric scores from saved evaluation records. The records in `verification/evaluations/` were calculated from the frozen artifacts and `../data/heart_disease_uci.csv`, using the stratified 20% split with `random_state=42` and threshold 0.50 specified in each source script. They contain all six comparison metrics, confusion matrices, sample counts, timestamps and SHA-256 fingerprints for the dataset, source and selected model/preprocessor files. The Random Forest values agree with the older saved summary in `../frontend_corveritas/backend/app/config.py`.

**The current MLP source uses that 184-row partition as validation during training.** Its scores are labeled validation, not independent test performance. The CNN and Random Forest source scripts reserve their corresponding 184-row partition for test evaluation. These columns should not be ranked as equivalent independent test results. The reconstruction follows the current source scripts; historical split metadata was not saved with these artifacts.

To create missing records from existing artifacts without retraining, run from `frontend_corveritas2/`:

```bash
python scripts/record_evaluation_metrics.py
```

Matching reports are reused without rescoring the evaluation partition. Stale reports are not overwritten automatically. The API rejects application-local reports whose dataset or required artifact fingerprints no longer match. The website only reads saved results: no training, tuning or evaluation occurs when loading or refreshing the comparison page. Original model/data files remain unchanged. Records saved alongside the selected artifacts by the training wrapper remain supported; missing values still appear as **“Not available”**.

## Dataset and methodology

The inspected CSV has **920 rows and 16 columns** from Cleveland, Hungary, VA Long Beach and Switzerland. The three original scripts drop identifiers/source metadata (`id`, `dataset`) and the original `num` target from the input. They create `target = (num > 0)`, yielding 509 positive and 411 negative rows. This combined CSV is larger than UCI's familiar 303-record Cleveland subset. See [data/README.md](data/README.md) for verified categories, missing counts and caveats.

All scripts use a stratified 80%/20% split with `random_state=42`: 736 training and 184 evaluation records. The current MLP source uses the 184 rows for validation during training; CNN and Random Forest use them for test evaluation. Numerical missing values use training medians; categorical values use the most-frequent training category plus one-hot encoding. Neural preprocessors standardize numerical features. The MLP uses dense 128/64/32 ReLU layers with batch normalization/dropout and a sigmoid output. The CNN uses Conv1D blocks with 32/64 filters, pooling, dropout, a dense 64 layer and a sigmoid output. Both use balanced training class weights. The current MLP uses the separate 184-row validation partition; the CNN uses an internal 20% training-validation split and early stopping.

The Random Forest uses a complete scikit-learn pipeline with imputation and one-hot encoding. `RandomizedSearchCV` examines 15 parameter combinations, scored using average precision and stratified five-fold CV **inside the outer training set**. Preprocessing is fitted within each CV fold. The selected training configuration is refitted on outer training data and evaluated on the held-out test partition by the original script.

The current MLP uses its validation partition to monitor training and adjust learning rate; no separate untouched MLP test set is defined by that source. The CNN fits preprocessing on the outer training set before Keras selects an internal validation subset, so that internal subset contributes to preprocessing statistics. Its outer test set remains separate. These limitations matter when interpreting or comparing scores; the original scripts and models are preserved.

Accuracy measures overall agreement; precision measures the fraction of positive predictions with positive labels; recall measures the fraction of positive labels detected; F1 combines precision/recall. ROC-AUC assesses ranking across thresholds; average precision summarizes the precision-recall relationship. None establishes clinical utility, calibration or external validity. Do not repeatedly tune/select on the held-out test partition.

Further limitations include small historical data, substantial missingness, site and demographic distribution differences, no prospective/external clinical validation, no dedicated calibration evaluation and training-dependent random variation. Neural training is not fully seed-controlled in the original scripts. A negative prediction cannot guarantee absence of disease. The prototype does not infer myocardial infarction, heart failure, arrhythmia, valve disease, severity or subtype.

The intended future architecture would train and independently validate a **separate disease-subtype classifier** using an appropriately labeled dataset. It is a research plan, not a feature of these binary artifacts.

References: [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart%2Bdisease), [scikit-learn model persistence and version compatibility](https://scikit-learn.org/stable/model_persistence.html), [Random Forest probability prediction](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

## Privacy and local operation

Assessment inputs and results remain in browser memory for the current application session. The app does not write them to a database, browser storage or input log, and sends requests only to the locally configured backend. Reloading the browser clears the assessment. API access logs contain route/status information, not the request body. Model artifacts and source CSV are not patient-submission storage. Bind the development services to loopback as shown; this is not a production deployment or a public clinical service.

## Automated checks

From `frontend_corveritas2/` with the environment activated:

```bash
python -m pytest backend/tests training/tests -q
python training/train.py --model mlp --show-plan
python training/train.py --model cnn --show-plan
python training/train.py --model random_forest --show-plan
cd frontend
npm test
npm run build
```

Backend checks cover schema/category/type rejection, missing values, feature ordering, correct pipeline selection, CNN dimensions, unavailable-model errors, API responses and available stored-artifact inference. Explanation checks verify score reconstruction rather than inventing an expected score. Frontend tests cover validation and rendering actual-shaped API results. The production build validates TypeScript and bundles the application. Artifact checks require real trusted model files; synthetic service doubles isolate error/shape behavior and are not exposed as available clinical models.

The delivered frontend passed **8 tests across three files** and its production build completed without warnings. Its form interaction tests exercise all four steps, required-field validation, Back navigation, reset, and exact Boolean/null/category JSON serialization through the real API client using a transport fixture. Actual inference is verified separately against the trusted artifacts over HTTP. See [verification/README.md](verification/README.md) for the complete check record and visual-QA limitation.

Training-wrapper checks verify that the three real scripts' non-path ASTs remain unchanged, that output paths stay inside this application, that the dataset path points to the original read-only CSV and that source-file contents remain intact. They compile transformed programs without executing any training.

Recorded delivery checks: the combined backend/wrapper suite passed **38 tests** in 6.03 seconds, with 19 dependency deprecation warnings and no failures. Hash verification confirmed that all 10 tracked original source/data/artifact files remained unchanged against [the original SHA256 record](verification/original-files.sha256.json). See [verification/files-created.txt](verification/files-created.txt) for the application file inventory.

Manual visual browser QA was not completed: native Chrome automation timed out, then its connection closed. No visual-rendering pass is claimed. The commands above provide separate frontend automated-test and production-build checks; their outcomes should be assessed independently of that browser limitation.

## Application structure

```text
frontend_corveritas2/
├── gui2.py                  # Optional development-process launcher
├── README.md
├── .env.example
├── .gitignore
├── data/README.md           # Read-only source CSV documentation
├── training/train.py        # Portable original-script launcher
├── scripts/verify_prediction.py # HTTP versus independent direct-inference check
├── models/README.md         # Optional local artifact overrides
├── backend/
│   ├── requirements.txt
│   ├── requirements-lock.txt # Full tested Python 3.11 macOS arm64 environment
│   ├── app/                # Routes, schemas, registry, inference, explanations/config
│   └── tests/
├── frontend/
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.ts
│   └── src/                # Pages, forms, reusable UI, API, types and validation
└── verification/            # Execution/check evidence and source integrity records
```

No database or Java service is required. All implementation and generated development files belong under this folder. Original Python training scripts are deliberately preserved in their existing repository location.

## Troubleshooting

- **Backend cannot import TensorFlow/sklearn:** activate `frontend_corveritas2/.venv`, verify `python --version` shows 3.11, then install the pinned requirements into that environment. The parent environment is separate.
- **Model unavailable:** inspect `/api/models` for the exact missing path or load error. Keep neural model/preprocessor pairs together and use dependency versions matching the saved artifacts. The API remains inspectable when no model is ready; it does not fabricate a prediction.
- **Frontend cannot reach the API:** confirm the backend uses port 8000 and the frontend uses port 5173. Use the documented URLs consistently and check the FastAPI startup output.
- **Port already in use:** stop the earlier development server in its terminal, then restart. Do not point the dashboard at a different service by accident.
- **A category is rejected:** use the schema-backed UI values. In particular the API uses `reversable defect`, `st-t abnormality`, `lv hypertrophy` and real JSON Booleans.
- **Missing measurements:** select Unknown. Imputation is disclosed in the result; it does not make the absent measurement known.
- **After retraining or replacing artifacts:** restart FastAPI. Models load once per process, not once per request.
