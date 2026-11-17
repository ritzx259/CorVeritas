"""Stage 2 is additive, gated on Stage 1, and isolated from detection failures."""
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.registry import ModelRegistry
from app.stage2_service import Stage2Service, Stage2Unavailable

FIELDS = [
    {"name": "age", "type": "number", "min": 20, "max": 120, "source_stage1": "age"},
    {"name": "bmi", "type": "number", "min": 10, "max": 100},
    {"name": "systolic_bp", "type": "number", "min": 50, "max": 300, "source_stage1": "trestbps"},
    {"name": "diastolic_bp", "type": "number", "min": 20, "max": 200},
    {"name": "total_cholesterol", "type": "number", "min": 40, "max": 1000, "source_stage1": "chol"},
    {"name": "sex", "type": "category", "categories": ["Female", "Male", "Unknown"], "source_stage1": "sex"},
    {"name": "diabetes_history", "type": "category", "categories": ["No", "Yes", "Borderline", "Unknown"]},
    {"name": "smoking_history", "type": "category", "categories": ["No", "Yes", "Unknown"]},
]


class Stage2Fixture:
    def __init__(self):
        self.available = True
        self.fail = False
        self.calls = []
        self.status_calls = 0
        self.closed = False

    def status(self):
        self.status_calls += 1
        return {"available": self.available, "features": FIELDS, "reason": "Stage 2 fixture unavailable"}

    def predict(self, features, explain=True):
        self.calls.append((features, explain))
        if self.fail:
            raise Stage2Unavailable("Stage 2 fixture timed out; Stage 1 is preserved.")
        return {"model": "neural", "conditions": [], "fixture": True}

    def close(self):
        self.closed = True


def setup(tmp_path, score=0.8):
    settings = Settings((tmp_path,), "random_forest", {"mlp": 0.5, "cnn": 0.5, "random_forest": 0.7}, ())
    registry = ModelRegistry(settings)
    registry._initialized = True
    class Pipeline:
        def predict_proba(self, frame):
            return np.array([[1-score, score]])
    registry.loaded["random_forest"] = SimpleNamespace(id="random_forest", model=Pipeline(), positive_index=1, explanation_ready=False, directory=tmp_path)
    service = Stage2Fixture()
    return create_app(settings, registry, service), service


def payload():
    return {"stage1": {"model_id": "random_forest", "features": {"age": 55, "sex": "Male", "cp": "asymptomatic", "trestbps": 145, "chol": 240, "fbs": True}},
            "additional_features": {"bmi": 28.4, "diastolic_bp": 90, "diabetes_history": "Yes", "smoking_history": "No"}}


def test_stage1_is_independent_and_stage2_is_lazy(tmp_path):
    app, service = setup(tmp_path)
    with TestClient(app) as client:
        response = client.post("/api/predict", json=payload()["stage1"])
        assert response.status_code == 200
        assert response.json()["model_score"] == 0.8
        assert service.status_calls == 0
        assert service.calls == []
        assert client.get("/api/health").json()["ready"] is True
    assert service.closed


def test_positive_gate_reuses_only_compatible_features(tmp_path):
    app, service = setup(tmp_path)
    with TestClient(app) as client:
        response = client.post("/api/stage2/predict", json=payload())
        assert response.status_code == 200
        assert service.calls == [({"age": 55, "sex": "Male", "bmi": 28.4, "systolic_bp": 145, "diastolic_bp": 90,
                                  "total_cholesterol": 240, "diabetes_history": "Yes", "smoking_history": "No"}, True)]
        assert set(service.calls[0][0]) == {field["name"] for field in FIELDS}


def test_negative_gate_never_starts_or_runs_stage2(tmp_path):
    app, service = setup(tmp_path, score=0.6)  # below configured 0.7, despite being above 0.5
    with TestClient(app) as client:
        response = client.post("/api/stage2/predict", json=payload())
        assert response.status_code == 409
        assert response.json()["detail"] == "Disease type classification was not performed because Stage 1 did not indicate cardiovascular disease."
        assert service.status_calls == 0
        assert not service.calls
        assert client.post("/api/predict", json=payload()["stage1"]).json()["predicted_class"] == 0


def test_age_range_does_not_change_stage1(tmp_path):
    app, service = setup(tmp_path)
    data = payload()
    data["stage1"]["features"]["age"] = 19
    with TestClient(app) as client:
        assert client.post("/api/predict", json=data["stage1"]).status_code == 200
        assert client.post("/api/stage2/predict", json=data).status_code == 422
        assert service.status_calls == 0


def test_unavailable_or_failed_stage2_preserves_stage1(tmp_path):
    app, service = setup(tmp_path)
    with TestClient(app) as client:
        original = client.post("/api/predict", json=payload()["stage1"]).json()
        service.available = False
        assert client.get("/api/stage2/schema").json()["available"] is False
        assert client.post("/api/stage2/predict", json=payload()).status_code == 503
        service.available, service.fail = True, True
        assert client.post("/api/stage2/predict", json=payload()).status_code == 503
        assert client.post("/api/predict", json=payload()["stage1"]).json() == original
        assert client.get("/api/health").json()["ready"] is True


def test_unknown_values_use_stage2_preprocessing_and_never_infer_diabetes_from_fbs(tmp_path):
    app, service = setup(tmp_path)
    data = payload()
    data["additional_features"] = {}
    data["stage1"]["features"].update({"trestbps": 40, "chol": 0})  # valid Stage 1, unsupported Stage 2
    with TestClient(app) as client:
        assert client.post("/api/stage2/predict", json=data).status_code == 200
    features = service.calls[0][0]
    assert features["age"] == 55
    assert features["systolic_bp"] is None
    assert features["total_cholesterol"] is None
    assert features["diabetes_history"] is None


@pytest.mark.parametrize("extra", [
    {"age": 65}, {"sex": "Female"}, {"bmi": True}, {"bmi": "28"}, {"bmi": 101},
    {"diastolic_bp": 145}, {"systolic_bp": 160}, {"diabetes_history": "true"},
    {"smoking_history": "Current smoker"}, {"unexpected": 1},
])
def test_invalid_additional_values_and_overrides_are_rejected(tmp_path, extra):
    app, service = setup(tmp_path)
    data = payload()
    data["additional_features"].update(extra)
    with TestClient(app) as client:
        response = client.post("/api/stage2/predict", json=data)
        assert response.status_code == 422
        assert not service.calls


def test_missing_stage2_runtime_is_reported_without_loading_stage1_dependencies(tmp_path):
    service = Stage2Service(model_dir=tmp_path, python=tmp_path / "missing-python", timeout=0.1)
    assert service.status()["available"] is False
    service.close()
