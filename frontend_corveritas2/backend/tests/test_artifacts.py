"""Inference compatibility checks only; these tests do not evaluate test-set metrics."""
import os

import numpy as np
import pandas as pd
import pytest

from app.config import REPO_ROOT, Settings
from app.explanations import explain_forest
from app.inference import predict_score, transformed_neural_input
from app.registry import ModelRegistry, SPECS, validate_preprocessor
from app.schemas import CATEGORIES, FEATURE_ORDER, Features, features_to_frame


@pytest.fixture(scope="module")
def actual_registry():
    settings = Settings.from_env()
    expected = [model_id for model_id, spec in SPECS.items() if any(all((directory / name).is_file() for name in spec["files"]) for directory in settings.model_dirs)]
    if not expected:
        pytest.skip("No trusted local artifacts are present; unit contracts remain tested.")
    registry = ModelRegistry(settings)
    registry.load_all()
    assert set(registry.loaded) == set(expected), registry.errors
    return registry


@pytest.mark.artifacts
def test_csv_schema_matches_actual_fitted_encoders(actual_registry):
    path = REPO_ROOT / "data" / "heart_disease_uci.csv"
    if not path.exists():
        pytest.skip("Original CSV not present in standalone checkout.")
    data = pd.read_csv(path)
    assert data.drop(columns=["id", "dataset", "num"]).columns.tolist() == FEATURE_ORDER
    assert len(data) == 920
    for feature, categories in CATEGORIES.items():
        assert set(data[feature].dropna().unique()) == set(categories)
    for loaded in actual_registry.loaded.values():
        validate_preprocessor(loaded.preprocessor)


@pytest.mark.artifacts
@pytest.mark.parametrize("model_id", ["mlp", "cnn", "random_forest"])
@pytest.mark.parametrize("missing", [False, True])
def test_inference_matches_direct_saved_model(actual_registry, model_id, missing):
    if model_id not in actual_registry.loaded:
        pytest.skip(f"{model_id} artifacts not present.")
    loaded = actual_registry.loaded[model_id]
    if missing:
        features = Features(age=50, sex="Female", cp="non-anginal")
    else:
        features = Features(age=63, sex="Male", cp="typical angina", trestbps=145, chol=233, fbs=True, restecg="lv hypertrophy", thalch=150, exang=False, oldpeak=2.3, slope="downsloping", ca=0, thal="fixed defect")
    frame = features_to_frame(features)
    score = predict_score(loaded, frame)
    if model_id == "random_forest":
        positive = list(loaded.model.classes_).index(1)
        direct = float(loaded.model.predict_proba(frame)[0, positive])
        explanation = explain_forest(loaded.model, frame, positive, score)
        assert abs(explanation["baseline"] + sum(item["value"] for item in explanation["contributions"]) - direct) < 1e-9
        assert explanation["reconstruction_error"] < 1e-9
        assert abs(sum(item["value"] for item in explanation["global_importance"]) - 1) < 1e-9
    else:
        matrix = loaded.preprocessor.transform(frame)
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        matrix = np.asarray(matrix, dtype=np.float32)
        if model_id == "cnn":
            matrix = matrix.reshape(1, matrix.shape[1], 1)
        direct = float(loaded.model.predict(matrix, verbose=0).reshape(-1)[0])
        assert matrix.shape == ((1, 25, 1) if model_id == "cnn" else (1, 25))
    assert score == pytest.approx(direct, abs=1e-6)
    assert 0 <= score <= 1


@pytest.mark.artifacts
def test_encoder_schema_mismatch_is_detected(actual_registry):
    import copy
    from app.registry import validate_preprocessor
    loaded = next(iter(actual_registry.loaded.values()))
    preprocessor = copy.deepcopy(loaded.preprocessor)
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    encoder.categories_[0] = np.array(["Unknown", "Male"])
    with pytest.raises(ValueError, match="categories"):
        validate_preprocessor(preprocessor)


@pytest.mark.artifacts
def test_recorded_metrics_reach_the_models_api(actual_registry):
    import json
    from fastapi.testclient import TestClient
    from app.evaluation_records import EVALUATION_DIR, METRIC_KEYS
    from app.main import create_app

    if not all((EVALUATION_DIR / f"{model_id}_metrics.json").exists() for model_id in SPECS):
        pytest.skip("No application-local evaluation reports in this checkout.")
    with TestClient(create_app(actual_registry.settings, actual_registry)) as client:
        response = client.get("/api/models")
        assert response.status_code == 200
        for model in response.json()["models"]:
            saved = json.loads((EVALUATION_DIR / f"{model['id']}_metrics.json").read_text())
            assert model["metrics"] == {key: saved[key] for key in METRIC_KEYS}
            assert model["metrics_evaluation"] == saved["evaluation"]
            assert model["metrics_note"] == saved["provenance"]
            assert model["metrics_evaluation"]["sample_count"] == 184
            assert model["metrics_evaluation"]["partition"] == ("validation" if model["id"] == "mlp" else "test")
