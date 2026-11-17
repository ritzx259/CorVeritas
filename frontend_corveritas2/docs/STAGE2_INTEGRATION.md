# Stage 2 integration contract

This integration extends the results page only. The four-step Stage 1 form,
validation, model selection, `/api/predict`, model registry, preprocessing and
Stage 1 result/explanation renderer are preserved. Stage 2 runs only after a
positive Stage 1 result and an explicit **Classify Disease Type** action.

## Inspected artifacts and outputs

Default artifact directory: `../models_stage2/models/`, the default used by the
original `models_stage2.predict.Stage2Predictor`. Metadata selects the **neural**
family, run `20261023T181032Z-cc7796d8`. The separate `models/run2/` experiment is
not silently substituted. The original loader checks the metadata version,
feature/label order, saved transformed feature names, and artifact SHA-256 hashes.

The files include `metadata.json`, `preprocessor.joblib`, `neural.pt`,
`baseline.joblib`, `explanation_background.joblib`, and `splits.csv`. The
selected network is a PyTorch MLP: 15 transformed inputs, hidden layers 128/64,
ReLU, LayerNorm, dropout, and three independent sigmoid scores. The fitted
ColumnTransformer handles median numeric imputation/scaling and fixed one-hot
categorical encoding. No Stage 1 preprocessor is used by Stage 2.

The actual ordered targets from `config.py` and metadata are:

| ID | Model target | Saved neural threshold |
|---|---|---:|
| `coronary_heart_disease` | Reported coronary heart disease history | 0.14829950034618378 |
| `heart_failure` | Reported congestive heart failure history | 0.13373057544231415 |
| `heart_attack_history` | Reported previous myocardial infarction | 0.1541489213705063 |

These are overlapping, self-reported **previous diagnosis histories** from
NHANES, not mutually exclusive disease subtypes or acute diagnoses. The UI keeps
the requested Stage 2 heading, but states the actual target meaning and shows
all three scores and thresholds. It never forces the largest score to become a
single disease label, renormalizes scores to 100%, or describes the scores as
calibrated clinical confidence. Multiple or zero categories may exceed their
own validation-selected thresholds. Cascading the NHANES model behind the UCI
Stage 1 detector has not itself been clinically or externally validated.

## Input mapping

Raw order is `age`, `bmi`, `systolic_bp`, `diastolic_bp`, `total_cholesterol`,
`sex`, `diabetes_history`, `smoking_history`.

| Stage 2 feature | Accepted values | Source in the assessment |
|---|---|---|
| `age` | Required, 20–120 years; encoded at maximum 80 | Stage 1 age; ages 18–19 stay supported by Stage 1 but cannot run Stage 2 |
| `bmi` | Optional, 10–100 kg/m² | Additional field |
| `systolic_bp` | Optional, 50–300 mmHg | Reuse valid `trestbps`; otherwise offer a separate optional Stage 2 field |
| `diastolic_bp` | Optional, 20–200 mmHg, below systolic if both supplied | Additional field |
| `total_cholesterol` | Optional, 40–1000 mg/dL | Reuse valid `chol`; otherwise offer a separate optional Stage 2 field |
| `sex` | Male, Female, Unknown | Stage 1 recorded sex |
| `diabetes_history` | Yes, No, Borderline, Unknown | Additional field; **never inferred from Stage 1 `fbs`** |
| `smoking_history` | Yes, No, Unknown | Additional field: at least 100 lifetime cigarettes, not current smoking status |

Compatible values are reused without modifying the Stage 1 input object. Stage 2
was trained on the mean of valid resting BP readings; its field help states this
measurement context. Invalid-for-Stage-2 values, such as Stage 1 cholesterol 0 or
systolic pressure 40, are not silently passed to that model. Unknown optional
values remain null and go through the saved Stage 2 preprocessing. All eight
raw inputs are validated again by the original `patient_frame`/`clean_features`.

## API and isolation

- `GET /api/stage2/schema`: lazy availability check, actual selected model/run,
  field names/ranges/categories, verified condition names and limitations.
- `POST /api/stage2/predict`: accepts the original `{model_id, features}` under
  `stage1`, plus `additional_features` and an optional Boolean `explain`.

Example structure (synthetic values):

```json
{
  "stage1": {
    "model_id": "random_forest",
    "features": {"age": 63, "sex": "Male", "cp": "asymptomatic"}
  },
  "additional_features": {
    "bmi": 28.4,
    "diastolic_bp": 90,
    "diabetes_history": "Yes",
    "smoking_history": "Unknown"
  },
  "explain": true
}
```

The server verifies eligibility using the original Stage 1 inference function
and that model's configured threshold. A negative verification returns 409 and
never starts the Stage 2 worker. Ineligible ages/invalid additional inputs return
422; missing artifacts, runtime errors, busy workers and timeouts return 503.
The original Stage 1 response is not overwritten or stored by these endpoints.
The original `/api/health`, `/api/models`, `/api/schema`, `/api/predict` and plot
routes retain their meaning and response contracts.

The original Stage 2 code and Torch dependencies run in a lazy, persistent worker
using `../models_stage2/.venv/bin/python`, with CPU inference and one Torch thread.
Stage 1 does not import Torch, install Stage 2 packages, or wait for Stage 2 at
startup. Schema/prediction handlers use the existing FastAPI thread-pool style;
Stage 1 requests can continue while Stage 2 loads or computes an explanation.
The worker is reused, has a 60-second response timeout, and is closed at API
shutdown. Its failure never causes the launcher to stop Stage 1.

Patient values travel only through anonymous stdin/stdout pipes and the existing
local HTTP request. They are not command-line arguments, temporary files,
application logs, browser storage or training data. Worker diagnostics are not
logged with patient inputs. Existing request-validation redaction also applies
to the new routes. No training, data download, test-set evaluation, threshold
selection or model replacement happens during application use.

## Explanation

The original exact grouped interventional Shapley implementation is reused:
256 coalitions over eight raw features, averaged over the saved 16-row training
background. Each condition has its own baseline and eight contributions in
model-score units. The results card offers a contribution chart and reports
reconstruction error. These finite-background associations are not causal or
clinical explanations. If explanation generation fails, classification is
retried without explanations and a warning is shown; no contribution is invented.

## Configuration and operation

The existing `python gui2.py` command is unchanged. The two existing environments
are reused; nothing is installed into Stage 1. To select another trusted Stage 2
artifact directory or interpreter, set these before launching the backend:

```bash
# Optional; relative paths resolve from the CorVeritas workspace root.
export CORVERITAS_STAGE2_MODEL_DIR=models_stage2/models/run2
export CORVERITAS_STAGE2_PYTHON=models_stage2/.venv/bin/python
```

Restart the backend after configuration/artifact changes. The Stage 2 availability
button can retry missing runtime/artifact errors without resubmitting Stage 1.
A negative Stage 1 result displays exactly:
“Disease type classification was not performed because Stage 1 did not indicate cardiovascular disease.”
It also states that the negative result does not guarantee absence of disease.

## Verification

`python scripts/verify_stage2_integration.py --output verification/stage2-integration.json`
checks the actual API against independent original `Stage2Predictor` inference on
two synthetic cases (complete and unknown optional inputs). All three scores in
both cases matched exactly, with valid explanation reconstruction. It also
checks the negative gate and unchanged Stage 1 API output after Stage 2 use and
when Stage 2 artifacts are unavailable. No held-out dataset is rescored.

`verification/stage2-preservation-before.json` records the original source/model
fingerprints. `stage2-preservation-after.json` confirms preservation, including
all Stage 1 model/preprocessor files, the form and validation, the existing API
routes/inference/registry, all inspected Stage 2 source/artifact files and the
original Stage 1 `ResultContent` function body. Only additive integration changes
to the application factory and results-page composition are permitted.
