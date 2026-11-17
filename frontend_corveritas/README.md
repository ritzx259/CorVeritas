# CorVeritas

CorVeritas is a local, educational cardiovascular research prototype. It uses the existing binary MLP, 1D CNN, and Random Forest artifacts when they load successfully; it does not implement disease-subtype classification.

> **Medical safety:** This prototype is not a clinically validated medical device. A model result does not establish or exclude a diagnosis, and its score is not necessarily a calibrated clinical probability. A qualified healthcare professional must interpret medical test results. Acute or concerning symptoms require immediate medical attention regardless of the output.

The application does not ask for names or contact details and does not persist submitted patient features. The API does not log request bodies.

## Project layout

```text
frontend_corveritas/
├── gui.py                         # Optional combined local launcher
├── README.md
├── backend/
│   ├── requirements.txt
│   ├── app/
│   │   ├── api/routes.py
│   │   ├── config.py
│   │   ├── explanations.py
│   │   ├── inference.py
│   │   ├── main.py
│   │   ├── model_registry.py
│   │   └── schemas.py
│   └── tests/
│       └── test_api.py
└── frontend/
    ├── package.json
    └── src/
        ├── api/
        ├── components/
        ├── pages/
        ├── validation/
        └── ...
```

Training scripts, the supplied dataset, model artifacts, and saved graphs remain in their original project folders. No files outside `frontend_corveritas/` are changed by this application.

## Requirements and artifact compatibility

The available training artifacts were inspected in this workspace. They expect these 13 features in order:

```text
age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang,
oldpeak, slope, ca, thal
```

The installed artifact environment reports Python 3.9.6, TensorFlow 2.20.0, scikit-learn 1.6.1, NumPy 2.0.2, pandas 2.3.3, and Joblib 1.5.3. Backend dependencies pin those model-library versions. Use Python 3.9 for the saved artifacts and do not load serialized models from untrusted sources.

The Random Forest `.joblib` contains both fitted preprocessing and estimator steps. The app passes it the raw feature DataFrame exactly once. The MLP and CNN use their matching fitted preprocessors; the CNN input is reshaped to samples × processed features × one channel.

## Run locally in Visual Studio Code on macOS

Open the project root folder in VS Code. Use two integrated terminal tabs. These commands do not copy, move, or alter the model artifacts.

### 1. Create the Python environment and install backend dependencies

Python 3.9 is recommended to match the existing serialized artifacts. From the project root:

```bash
cd frontend_corveritas/backend
../../.venv/bin/python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first command uses this workspace's existing Python 3.9 interpreter without changing the root environment. If you do not have that root `.venv`, use `python3.9 -m venv .venv` or install Python 3.9 first. `tensorflow==2.20.0` and the serialized scikit-learn pipeline version are pinned to the environment inspected for these artifacts.

### 2. Start the FastAPI backend

Keep the backend terminal activated in `frontend_corveritas/backend`:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

### 3. Install and start the React frontend

In a second VS Code terminal, from the project root:

```bash
cd frontend_corveritas/frontend
npm install
npm run dev
```

Open **http://127.0.0.1:5173**. The frontend calls the local backend at `http://127.0.0.1:8000/api`. To use a different backend origin, set `VITE_API_BASE` before starting Vite.

Alternatively, after installing dependencies and activating the backend `.venv`, run `python frontend_corveritas/gui.py` from the project root to start both development servers; press Ctrl+C to stop them.

## Model artifact paths and availability

The API checks these paths relative to the project root `models/` directory:

| Model | Required artifact(s) |
|---|---|
| MLP | `corveritas_stage1.keras`, `preprocessor.pkl` |
| 1D CNN | `corveritas_stage2_cnn.keras`, `preprocessor_stage2_cnn.pkl` |
| Random Forest | `corveritas_stage3_random_forest.joblib` |

The Random Forest preprocessor is already included in the fitted `.joblib` pipeline. Its separately exported preprocessor is not required for inference.

On startup, each model is loaded once and is listed as available only after its artifacts load and the input schema is checked. Missing or incompatible artifacts appear as unavailable with a training command. From the project root, run only the relevant existing script if an artifact must be generated:

```bash
python models/training_models.py
python models/training_model2.py
python models/training_model3.py
```

The first command trains the MLP, the second the 1D CNN, and the third the Random Forest. Model names containing `stage2` or `stage3` are artifact names, not disease stages or subtype classifiers. Training scripts were inspected but not changed.

`CORVERITAS_DEFAULT_MODEL` selects the configured default (`random_forest` by default). It is not silently replaced if unavailable. `CORVERITAS_CORS_ORIGINS` can override the default local Vite origins `http://localhost:5173,http://127.0.0.1:5173`.

## Verify a real prediction

1. Check `http://127.0.0.1:8000/api/health`; `ready` should be true and `available_model_ids` should list successfully loaded models.
2. Check `http://127.0.0.1:8000/api/models` for artifact status and the configured default.
3. In the UI, complete a sample assessment and submit it. The response includes the model id, output score, threshold, missing feature names, explanation status, and disclaimer. The selected model id is never redirected to another model.
4. A development smoke test from the project root with the backend environment active:

```bash
curl -s http://127.0.0.1:8000/api/health
curl -s http://127.0.0.1:8000/api/models
curl -s -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"random_forest","features":{"age":63,"sex":"Male","cp":"typical angina","trestbps":145,"chol":233,"fbs":true,"restecg":"lv hypertrophy","thalch":150,"exang":false,"oldpeak":2.3,"slope":"downsloping","ca":0,"thal":"fixed defect"}}'
```

The API assembles a one-row Pandas DataFrame preserving the training column names and order. Booleans are passed as booleans; omitted optional measurements are `null` and become missing values for the fitted imputers. The exact category string `reversable defect` is retained in requests even though the UI displays “Reversible defect.”

## API

- `GET /api/health` — backend readiness, available model ids, and load errors.
- `GET /api/models` — registered model metadata, real artifact availability, threshold, metric availability, and existing graph links.
- `GET /api/schema` — the 13 training fields, accepted categories, units, and required/optional flags.
- `POST /api/predict` — validated one-row inference for the explicitly selected model.

The MLP and CNN apply their matching saved preprocessors. The Random Forest receives the original DataFrame through its complete fitted pipeline, so preprocessing is not repeated. All three models use the training scripts' 0.50 threshold.

## Explainability, evaluation, and limitations

- The Random Forest attempts a local SHAP TreeExplainer calculation. Encoded contributions are grouped by original feature name and shown only if the expected output is consistent with the model score. Explanations describe model attribution, not causes or clinical effects.
- MLP/CNN local explanations are not implemented; the UI states they are unavailable rather than inventing rules.
- If SHAP is missing or incompatible with an artifact, prediction remains available but its explanation is explicitly unavailable.
- The Random Forest's held-out metrics are preserved from its saved `test_metrics.png` chart (accuracy 0.859, precision 0.858, recall 0.892, F1 0.875, ROC-AUC 0.924, average precision 0.935). The MLP/CNN test metrics were printed during training but are not saved as machine-readable values, so they show “Not available.” No held-out test set is reevaluated by the application; saved graph images are served from their original folders.
- The training files make binary predictions only: `num == 0` versus `num > 0`. They do not predict myocardial infarction, heart failure, arrhythmia, valve disease, or a subtype.
- Dataset cohort, preprocessing choices, model split, and future second-stage design are described on the Methodology page. The input dataset has missing measurements; the persisted fitted pipelines control imputation.

## Tests

From `frontend_corveritas/backend` with its virtual environment active:

```bash
pytest
```

From `frontend_corveritas/frontend`:

```bash
npm test
npm run build
```

Backend tests cover schema/category validation, nullable values and column order, MLP preprocessing, CNN reshape, raw Random Forest pipeline inference, unavailable-model behavior, API response structure, and—when artifacts load—agreement with direct predictions from each saved model.
