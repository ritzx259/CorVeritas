# Trusted local model artifacts

The API loads fitted artifacts once at startup. It prefers complete model sets
in this folder, then checks the repository's existing `../../models/` folder.
It never downloads serialized model files. Only load trusted artifacts:
Joblib/Pickle deserialization can execute Python code.

| Model ID | Required files |
| --- | --- |
| `mlp` | `corveritas_stage1.keras`, `preprocessor.pkl` |
| `cnn` | `corveritas_stage2_cnn.keras`, `preprocessor_stage2_cnn.pkl` |
| `random_forest` | `corveritas_stage3_random_forest.joblib` |

The Random Forest Joblib file is a complete fitted pipeline. Its separate
`preprocessor_stage3_random_forest.pkl` is not required for inference and must
not be applied a second time. `stage1`, `stage2` and `stage3` identify historical
filenames; every current model predicts the same binary target.

Use `GET /api/models` to inspect availability, actual source paths, load errors,
thresholds, existing plots and explanation support. A file's existence alone
does not establish availability: loading and compatibility checks must pass.
Set `CORVERITAS_MODEL_DIR` to restrict loading to a single directory. Restart
the API after changing any artifact, dependency version or configuration.

Existing plots are discovered read-only from each model's graph folder:
`model_mlp_graph/`, `model_cnn_graph/`, `model_random_forest_graph/`. The original
scripts printed scalar metrics instead of saving numeric evaluation reports.
Application-local reports now live in `../verification/evaluations/`. They were
calculated once from the frozen model artifacts and original CSV, following
the current source scripts' stratified 20% split and threshold 0.50. The MLP
column is validation performance, since that partition was monitored during
training; CNN and Random Forest columns are test performance. The API checks
the dataset and selected artifact SHA-256 fingerprints before showing these
records. Loading the website never runs an evaluation.

Run `python scripts/record_evaluation_metrics.py` from the application folder
to create missing records. Matching records are reused; stale records are not
automatically replaced. The report includes the source script, dataset and
artifact hashes, sample counts, confusion matrix and evaluation protocol.

Future runs through `training/train.py` also save `<model_id>_metrics.json`:
only the original script's already-computed held-out metrics, plus source/CSV
hashes and run provenance. No second evaluation is performed to create these
records. The API validates finite 0–1 scalar values before showing them. The application-local evaluation reports are separate from these training-run
records and explicitly identify their reconstructed evaluation protocol.

If artifacts are missing, from `frontend_corveritas2/` run one of:

```bash
python training/train.py --model mlp
python training/train.py --model cnn
python training/train.py --model random_forest
```

The wrapper redirects only path assignments in an in-memory AST copy of the
original script. It reads the original CSV, uses noninteractive plotting, and
publishes successful outputs here. Original source/data/artifacts stay intact.
It stages training inside this folder and publishes artifacts only after the
original script completes. Use `--show-plan` to inspect without training and
`--overwrite` to replace a previously trained local model. Stop the API before
replacing loaded model sets, then restart it.

Retraining follows the original scripts, including their fixed test split and
evaluation. It is not a method for repeatedly checking the test set and choosing
the best result. Define a validation and independent evaluation protocol before
any future model-development work.
