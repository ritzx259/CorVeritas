"""Compare live HTTP predictions with independent trusted-artifact inference.

Run with the backend running: python scripts/verify_prediction.py
Uses synthetic inputs and never evaluates, trains, or selects models.
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for variable, directory in (("KERAS_HOME", ROOT / ".cache" / "keras"),
                            ("XDG_CACHE_HOME", ROOT / ".cache"),
                            ("MPLCONFIGDIR", ROOT / ".cache" / "matplotlib")):
    os.environ.setdefault(variable, str(directory))
    Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import httpx
import joblib
import numpy as np
import pandas as pd

ORDER = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal"]
NUMERIC = {"age", "trestbps", "chol", "thalch", "oldpeak", "ca"}
ARTIFACTS = {
    "mlp": ("corveritas_stage1.keras", "preprocessor.pkl"),
    "cnn": ("corveritas_stage2_cnn.keras", "preprocessor_stage2_cnn.pkl"),
    "random_forest": ("corveritas_stage3_random_forest.joblib",),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--model-dir", type=Path, help="Use the same trusted model directory as your API.")
    parser.add_argument("--output", type=Path, help="Optionally save verification metadata; inputs are never saved.")
    args = parser.parse_args()
    configured = args.model_dir or os.getenv("CORVERITAS_MODEL_DIR")
    if configured:
        directory = Path(configured).expanduser()
        directories = [directory if directory.is_absolute() else ROOT / directory]
    else:
        directories = [ROOT / "models", ROOT.parent / "models"]
    complete = dict(age=54, sex="Female", cp="atypical angina", trestbps=128, chol=216,
                    fbs=False, restecg="normal", thalch=157, exang=False, oldpeak=0.7,
                    slope="upsloping", ca=0, thal="normal")
    partial = {name: complete[name] if name in {"age", "sex", "cp"} else None for name in ORDER}
    summary = {"purpose": "Live API versus direct artifact inference on synthetic complete and missing-value inputs", "api_origin": args.api, "models": []}
    with httpx.Client(base_url=args.api, timeout=60, trust_env=False) as client:
        response = client.get("/api/models")
        response.raise_for_status()
        available = [item for item in response.json()["models"] if item["available"]]
        if not available:
            raise RuntimeError("The API has no available trained models. Inspect /api/models for the load errors.")
        for item in available:
            model_id = item["id"]
            names = ARTIFACTS[model_id]
            directory = next((path for path in directories if all((path / name).is_file() for name in names)), None)
            if directory is None:
                raise RuntimeError(f"Cannot locate the trusted artifacts for {model_id}; pass --model-dir.")
            if model_id == "random_forest":
                model = joblib.load(directory / names[0])
                preprocessor = None
            else:
                import tensorflow as tf
                model = tf.keras.models.load_model(directory / names[0], compile=False, safe_mode=True)
                preprocessor = joblib.load(directory / names[1])
            errors = []
            for features in (complete, partial):
                frame = pd.DataFrame({name: pd.Series([np.nan if features[name] is None else features[name]],
                                                      dtype="float64" if name in NUMERIC else "object") for name in ORDER})
                if model_id == "random_forest":
                    positive_index = list(model.classes_).index(1)
                    direct = float(model.predict_proba(frame)[0, positive_index])
                else:
                    matrix = preprocessor.transform(frame)
                    if hasattr(matrix, "toarray"):
                        matrix = matrix.toarray()
                    matrix = np.asarray(matrix, dtype=model.input_dtype)
                    if model_id == "cnn":
                        matrix = matrix.reshape(1, matrix.shape[1], 1)
                    direct = float(np.asarray(model(matrix, training=False)).reshape(-1)[0])
                result = client.post("/api/predict", json={"model_id": model_id, "features": features})
                result.raise_for_status()
                prediction = result.json()
                delta = abs(prediction["model_score"] - direct)
                if delta > 1e-6 or prediction["model_id"] != model_id:
                    raise AssertionError(f"API/direct mismatch for {model_id}: difference {delta}")
                if prediction["predicted_class"] != int(direct >= prediction["threshold"]):
                    raise AssertionError(f"Threshold classification mismatch for {model_id}")
                if prediction["missing_features"] != [name for name in ORDER if features[name] is None]:
                    raise AssertionError(f"Missing-value reporting mismatch for {model_id}")
                explanation = prediction.get("explanations")
                if explanation:
                    reconstruction = explanation["baseline"] + sum(entry["value"] for entry in explanation["contributions"])
                    if abs(reconstruction - direct) > 1e-9:
                        raise AssertionError("Local explanation failed score reconstruction")
                errors.append(delta)
            summary["models"].append({"model_id": model_id, "cases": 2, "max_absolute_difference": max(errors), "passed": True})
            print(f"PASS {model_id}: complete and missing-value HTTP predictions match direct inference (max difference {max(errors):.2g})")
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n")
    print("No training or test-set evaluation was performed.")


if __name__ == "__main__":
    main()
