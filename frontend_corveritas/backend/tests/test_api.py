from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import api
from app.config import FEATURE_ORDER
from app.inference import build_feature_frame, predict
from app.main import app
from app.model_registry import LoadedModel, ModelRegistry
from app.schemas import Features


VALID_FEATURES = {
    "age": 63,
    "sex": "Male",
    "cp": "typical angina",
    "trestbps": 145,
    "chol": 233,
    "fbs": True,
    "restecg": "lv hypertrophy",
    "thalch": 150,
    "exang": False,
    "oldpeak": 2.3,
    "slope": "downsloping",
    "ca": 0,
    "thal": "fixed defect",
}


class FakePreprocessor:
    def __init__(self, transformed: np.ndarray | None = None) -> None:
        self.transformed = transformed if transformed is not None else np.array([[1, 2, 3]])
        self.received: pd.DataFrame | None = None

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        self.received = frame
        return self.transformed


class FakeKerasModel:
    def __init__(self, output: float = 0.72) -> None:
        self.output = output
        self.received: np.ndarray | None = None

    def predict(self, values: np.ndarray, verbose: int = 0) -> np.ndarray:
        self.received = values
        return np.array([[self.output]])


class FakeEstimator:
    classes_ = np.array([0, 1])


class FakePipeline:
    named_steps = {"model": FakeEstimator(), "preprocessor": object()}

    def __init__(self, probability: float = 0.7) -> None:
        self.probability = probability
        self.received: pd.DataFrame | None = None

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        self.received = frame
        return np.array([[1 - self.probability, self.probability]])


def fake_loaded_model(model_id: str, model: Any, preprocessor: Any = None) -> LoadedModel:
    return LoadedModel(model_id, {"name": model_id}, model, preprocessor)


def test_feature_schema_rejects_invalid_categories_and_missing_required_fields() -> None:
    with pytest.raises(ValueError):
        Features(**{**VALID_FEATURES, "cp": "non angina"})
    with pytest.raises(ValueError):
        Features(**{key: value for key, value in VALID_FEATURES.items() if key != "age"})


def test_exact_training_category_and_missing_measurements_are_preserved() -> None:
    features = {**VALID_FEATURES, "thal": "reversable defect", "chol": None, "fbs": None}
    frame = build_feature_frame(features)
    assert frame.columns.tolist() == FEATURE_ORDER
    assert frame.loc[0, "thal"] == "reversable defect"
    assert np.isnan(frame.loc[0, "chol"])
    assert np.isnan(frame.loc[0, "fbs"])
    assert frame["chol"].dtype == np.dtype("float64")
    assert frame["fbs"].dtype == object


def test_mlp_preprocesses_and_uses_float32_input() -> None:
    preprocessor = FakePreprocessor(np.array([[1.0, 2.0, 3.0]], dtype=np.float64))
    model = FakeKerasModel()
    result = predict(fake_loaded_model("mlp", model, preprocessor), VALID_FEATURES)
    assert model.received is not None
    assert model.received.dtype == np.float32
    assert model.received.shape == (1, 3)
    assert preprocessor.received is not None
    assert preprocessor.received.columns.tolist() == FEATURE_ORDER
    assert result.model_score == pytest.approx(0.72)


def test_cnn_preprocessed_matrix_gets_channel_dimension() -> None:
    preprocessor = FakePreprocessor(np.arange(5, dtype=np.float64).reshape(1, 5))
    model = FakeKerasModel(0.2)
    result = predict(fake_loaded_model("cnn", model, preprocessor), VALID_FEATURES)
    assert model.received is not None
    assert model.received.shape == (1, 5, 1)
    assert model.received.dtype == np.float32
    assert result.predicted_class == 0


def test_random_forest_receives_raw_frame_without_double_preprocessing(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.inference as inference

    pipeline = FakePipeline(0.7)
    monkeypatch.setattr(
        inference,
        "local_random_forest_explanation",
        lambda *args: (None, "Explanation unavailable in test."),
    )
    result = predict(fake_loaded_model("random_forest", pipeline), VALID_FEATURES)
    assert pipeline.received is not None
    assert pipeline.received.columns.tolist() == FEATURE_ORDER
    assert result.model_score == pytest.approx(0.7)
    assert result.predicted_class == 1


def test_api_returns_structured_prediction_and_no_model_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.api.routes as routes

    model = fake_loaded_model("mlp", FakeKerasModel(), FakePreprocessor())

    class AvailableRegistry:
        available = {"mlp": model}
        errors: dict[str, str] = {}
        default_model_id = "mlp"

        def get(self, model_id: str) -> LoadedModel | None:
            return self.available.get(model_id)

        def list_models(self) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(routes, "registry", AvailableRegistry())
    client = TestClient(app)
    response = client.post(
        "/api/predict",
        json={"model_id": "mlp", "features": VALID_FEATURES},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == "mlp"
    assert body["predicted_class"] == 1
    assert isinstance(body["model_score"], float)
    assert body["threshold"] == 0.5
    assert "disclaimer" in body
    assert "explanations" not in body
    invalid_response = client.post(
        "/api/predict",
        json={"model_id": "mlp", "features": {**VALID_FEATURES, "cp": "non angina"}},
    )
    assert invalid_response.status_code == 422

    unavailable_response = client.post(
        "/api/predict",
        json={"model_id": "cnn", "features": VALID_FEATURES},
    )
    assert unavailable_response.status_code == 404
    assert "fallback" in unavailable_response.json()["detail"]
    monkeypatch.setattr(routes, "registry", AvailableRegistry())
    routes.registry.errors["cnn"] = "CNN artifact unavailable."
    unavailable_response = client.post(
        "/api/predict",
        json={"model_id": "cnn", "features": VALID_FEATURES},
    )
    assert unavailable_response.status_code == 503
    assert "unavailable" in unavailable_response.json()["detail"].lower()


def test_unavailable_artifact_registry_explains_training_command(tmp_path: Path) -> None:
    registry = ModelRegistry(model_dir=tmp_path)
    assert "mlp" not in registry.available
    assert "training_models.py" in registry.errors["mlp"]


def test_saved_random_forest_metrics_match_original_metrics_plot() -> None:
    from app.api.routes import registry

    random_forest = next(model for model in registry.list_models() if model["id"] == "random_forest")
    assert random_forest["metrics"] == {
        "accuracy": 0.859,
        "precision": 0.858,
        "recall": 0.892,
        "f1": 0.875,
        "roc_auc": 0.924,
        "average_precision": 0.935,
    }
    assert "test-metrics chart" in random_forest["metrics_source"]


def test_loaded_random_forest_handles_unknown_optional_measurements() -> None:
    from app.api.routes import registry

    registered = registry.get("random_forest")
    if registered is None:
        pytest.skip(f"Random Forest artifact did not load: {registry.errors.get('random_forest')}")
    features = {
        **VALID_FEATURES,
        "trestbps": None,
        "chol": None,
        "fbs": None,
        "restecg": None,
        "thalch": None,
        "exang": None,
        "oldpeak": None,
        "slope": None,
        "ca": None,
        "thal": None,
    }
    result = predict(registered, features)
    assert 0 <= result.model_score <= 1
    assert set(result.missing_features) == {
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalch",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    }


@pytest.mark.parametrize("model_id", ["mlp", "cnn", "random_forest"])
def test_saved_artifact_inference_matches_direct_prediction(model_id: str) -> None:
    import app.api.routes as routes

    registered = routes.registry.get(model_id)
    if registered is None:
        pytest.skip(f"Actual {model_id} artifacts did not load: {routes.registry.errors.get(model_id)}")

    row = pd.read_csv(Path(__file__).resolve().parents[3] / "data" / "heart_disease_uci.csv").iloc[0]
    feature_values: dict[str, Any] = {}
    for name in FEATURE_ORDER:
        value = row[name]
        if pd.isna(value):
            feature_values[name] = None
        elif name in ("fbs", "exang"):
            feature_values[name] = bool(value)
        elif name in ("age", "ca"):
            feature_values[name] = int(value)
        elif name in ("trestbps", "chol", "thalch", "oldpeak"):
            feature_values[name] = float(value)
        else:
            feature_values[name] = str(value)
    validated = Features(**feature_values).ordered_dict()

    if model_id == "random_forest":
        expected = float(registered.model.predict_proba(build_feature_frame(validated))[0, 1])
    else:
        processed = np.asarray(
            registered.preprocessor.transform(build_feature_frame(validated)),
            dtype=np.float32,
        )
        if model_id == "cnn":
            processed = processed.reshape(processed.shape[0], processed.shape[1], 1)
        expected = float(np.asarray(registered.model.predict(processed, verbose=0)).reshape(-1)[0])
    actual = predict(registered, validated)
    assert actual.model_score == pytest.approx(expected, abs=1e-6)
    if model_id == "random_forest":
        assert actual.explanation_status.startswith("Local SHAP")
        assert actual.explanations
        assert all("__" not in item.feature for item in actual.explanations)
