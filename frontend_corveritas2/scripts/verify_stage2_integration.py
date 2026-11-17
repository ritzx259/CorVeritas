#!/usr/bin/env python3
"""Verify real two-stage integration on synthetic inputs without training or test evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "backend"))
from app.config import REPO_ROOT, Settings, configure_runtime_environment
configure_runtime_environment()
from fastapi.testclient import TestClient
from app.main import create_app
from app.registry import ModelRegistry
from app.stage2_service import Stage2Service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = Settings.from_env()
    registry = ModelRegistry(settings)
    app = create_app(settings, registry)
    positive = {"model_id": "random_forest", "features": {
        "age": 63, "sex": "Male", "cp": "asymptomatic", "trestbps": 145, "chol": 233,
        "fbs": True, "restecg": "lv hypertrophy", "thalch": 150, "exang": True,
        "oldpeak": 2.3, "slope": "flat", "ca": 2, "thal": "reversable defect"}}
    negative = {"model_id": "random_forest", "features": {
        "age": 28, "sex": "Female", "cp": "non-anginal", "trestbps": 110, "chol": 160,
        "fbs": False, "restecg": "normal", "thalch": 190, "exang": False,
        "oldpeak": 0, "slope": "upsloping", "ca": 0, "thal": "normal"}}
    extras = {"bmi": 28.4, "diastolic_bp": 90, "diabetes_history": "Yes", "smoking_history": "Yes"}
    summary = {"purpose": "Real Stage 2 API versus original Stage2Predictor on synthetic inputs; no training, data download or held-out evaluation", "cases": []}
    with TestClient(app) as client:
        before = client.post("/api/predict", json=positive)
        before.raise_for_status()
        assert before.json()["predicted_class"] == 1
        metadata = client.get("/api/stage2/schema").json()
        assert metadata["available"], metadata
        assert [field["name"] for field in metadata["features"]] == ["age", "bmi", "systolic_bp", "diastolic_bp", "total_cholesterol", "sex", "diabetes_history", "smoking_history"]
        patients = []
        api_results = []
        for name, extra in [("complete", extras), ("optional_missing", {})]:
            response = client.post("/api/stage2/predict", json={"stage1": positive, "additional_features": extra})
            response.raise_for_status()
            api_results.append(response.json())
            patients.append({"age": 63, "sex": "Male", "systolic_bp": 145, "total_cholesterol": 233, **extra})
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        program = "import json,sys; from models_stage2.predict import Stage2Predictor; p=Stage2Predictor(); print(json.dumps([p.predict(patient, explain=True) for patient in json.load(sys.stdin)], allow_nan=False))"
        direct = subprocess.run([str(REPO_ROOT / "models_stage2" / ".venv" / "bin" / "python"), "-B", "-c", program],
                                input=json.dumps(patients), text=True, capture_output=True,
                                cwd=REPO_ROOT, env=environment, timeout=90, check=True)
        originals = json.loads(direct.stdout)
        for name, actual, original in zip(("complete", "optional_missing"), api_results, originals):
            assert actual["run_id"] == original["run_id"]
            errors = []
            for got, expected in zip(actual["conditions"], original["conditions"]):
                assert got["condition"] == expected["condition"]
                assert got["condition_name"] == expected["condition_name"]
                assert got["binary_classification"] == expected["binary_classification"]
                assert got["selected_threshold"] == expected["selected_threshold"]
                difference = abs(got["model_score"] - expected["model_score"])
                assert difference < 1e-7
                assert len(got["feature_contributions"]) == 8
                reconstruction = got["explanation_base_score"] + sum(item["contribution"] for item in got["feature_contributions"])
                assert abs(reconstruction - got["model_score"]) < 1e-5
                errors.append(difference)
            summary["cases"].append({"input": name, "conditions": len(actual["conditions"]), "maximum_score_difference": max(errors), "explanation_reconstruction": "passed"})
        rejected = client.post("/api/predict", json=negative)
        rejected.raise_for_status()
        assert rejected.json()["predicted_class"] == 0
        assert client.post("/api/stage2/predict", json={"stage1": negative}).status_code == 409
        assert client.post("/api/predict", json=positive).json() == before.json()
        summary.update({"selected_stage2_model": metadata["model"], "stage2_run_id": metadata["run_id"], "negative_stage1_gate": "passed", "stage1_response_unchanged_after_stage2": True})
    unavailable = Stage2Service(model_dir=APP_ROOT / ".cache" / "stage2-missing-artifacts")
    with TestClient(create_app(settings, registry, unavailable)) as client:
        assert client.get("/api/stage2/schema").json()["available"] is False
        assert client.post("/api/stage2/predict", json={"stage1": positive}).status_code == 503
        assert client.post("/api/predict", json=positive).json() == before.json()
        assert client.get("/api/health").json()["ready"] is True
    summary["stage1_independent_when_stage2_unavailable"] = True
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
