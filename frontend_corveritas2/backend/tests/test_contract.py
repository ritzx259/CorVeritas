from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.inference import positive_class_index, predict_score, transformed_neural_input
from app.main import create_app
from app.registry import ModelRegistry, validate_preprocessor
from app.schemas import FEATURE_ORDER, Features, features_to_frame, schema_metadata


def configuration(tmp_path):
    return Settings((tmp_path,), "random_forest", {"mlp": 0.5, "cnn": 0.5, "random_forest": 0.7}, ("http://localhost:5173",))


def test_schema_uses_exact_dataset_categories():
    definitions = {field["name"]: field for field in schema_metadata()["fields"]}
    assert list(definitions) == FEATURE_ORDER
    assert [option["value"] for option in definitions["thal"]["categories"]] == ["normal", "fixed defect", "reversable defect"]
    assert definitions["thal"]["categories"][-1]["label"] == "reversible defect"
    assert [name for name in FEATURE_ORDER if definitions[name]["required"]] == ["age", "sex", "cp"]


@pytest.mark.parametrize("change", [
    {"age": None}, {"age": True}, {"age": "50"}, {"sex": "male"}, {"cp": "Typical Angina"},
    {"thal": "reversible defect"}, {"restecg": "ST-T abnormality"}, {"fbs": "Yes"},
    {"exang": 1}, {"ca": 4}, {"ca": 1.0}, {"chol": float("nan")}, {"oldpeak": float("inf")},
    {"trestbps": True}, {"chol": "200"}, {"unexpected": 1},
])
def test_strict_feature_validation(change):
    with pytest.raises(ValidationError):
        Features(**({"age": 50, "sex": "Male", "cp": "asymptomatic"} | change))


def test_frame_order_missing_and_boolean_types():
    features = Features(age=50, sex="Female", cp="non-anginal", fbs=False, exang=True, thal="reversable defect")
    frame = features_to_frame(features)
    assert frame.columns.tolist() == FEATURE_ORDER
    assert frame.loc[0, "age"] == 50
    assert frame.loc[0, "fbs"] is False
    assert frame.loc[0, "exang"] is True
    assert frame["fbs"].dtype == object
    assert np.isnan(frame.loc[0, "trestbps"])
    assert np.isnan(frame.loc[0, "restecg"])
    assert frame.loc[0, "thal"] == "reversable defect"
    assert features.chol is None  # no arbitrary numerical defaults


class Preprocessor:
    def __init__(self):
        self.received = None

    def transform(self, frame):
        self.received = frame
        return np.array([[1, 2, 3]], dtype=np.float64)


@pytest.mark.parametrize("cnn,shape", [(False, (1, 3)), (True, (1, 3, 1))])
def test_neural_preprocessing_dtype_and_shape(cnn, shape):
    preprocessor = Preprocessor()
    model = SimpleNamespace(input_dtype="float32", input_shape=(None, *shape[1:]))
    frame = features_to_frame(Features(age=50, sex="Male", cp="asymptomatic"))
    matrix = transformed_neural_input(preprocessor, model, frame, cnn)
    assert matrix.shape == shape
    assert matrix.dtype == np.float32
    assert preprocessor.received is frame
    np.testing.assert_array_equal(matrix.reshape(-1), [1, 2, 3])


def test_neural_preprocessor_mismatch_rejected():
    with pytest.raises(ValueError, match="input shape"):
        transformed_neural_input(Preprocessor(), SimpleNamespace(input_dtype="float32", input_shape=(None, 4)), object())


def test_rf_positive_class_order_and_raw_frame():
    class Pipeline:
        classes_ = np.array([1, 0])
        def predict_proba(self, frame):
            assert frame.columns.tolist() == FEATURE_ORDER
            assert frame.loc[0, "sex"] == "Male"
            return np.array([[0.8, 0.2]])
    model = Pipeline()
    loaded = SimpleNamespace(id="random_forest", model=model, positive_index=positive_class_index(model.classes_))
    frame = features_to_frame(Features(age=50, sex="Male", cp="asymptomatic"))
    assert predict_score(loaded, frame) == 0.8
    with pytest.raises(ValueError):
        positive_class_index([0, 2])


def test_missing_models_are_not_available_and_default_is_not_replaced(tmp_path):
    registry = ModelRegistry(configuration(tmp_path))
    registry.load_all()
    registry.load_all()
    assert registry.loaded == {}
    assert registry.default_model_id is None
    assert all(not model["available"] for model in registry.metadata()["models"])
    assert all("missing" in model["reason"] for model in registry.metadata()["models"])


def test_unavailable_api_validation_and_privacy(tmp_path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        assert client.get("/api/health").json()["ready"] is False
        assert client.get("/api/schema").status_code == 200
        assert client.get("/api/models/random_forest/plots/../secret").status_code == 404
        payload = {"model_id": "random_forest", "features": {"age": 50, "sex": "Male", "cp": "asymptomatic"}}
        assert client.post("/api/predict", json=payload).status_code == 503
        payload["features"]["sex"] = "private-invalid-value"
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 422
        assert "private-invalid-value" not in response.text
        assert "input" not in response.json()["detail"][0]
        cors = client.options("/api/predict", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
        assert cors.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_successful_api_schema_threshold_and_no_fallback(tmp_path):
    class Pipeline:
        def predict_proba(self, frame):
            assert frame.loc[0, "age"] == 50
            assert frame.loc[0, "fbs"] is True
            assert np.isnan(frame.loc[0, "chol"])
            return np.array([[0.4, 0.6]])
    settings = configuration(tmp_path)
    registry = ModelRegistry(settings)
    registry._initialized = True
    registry.loaded["random_forest"] = SimpleNamespace(id="random_forest", model=Pipeline(), positive_index=1, explanation_ready=False)
    with TestClient(create_app(settings, registry)) as client:
        payload = {"model_id": "random_forest", "features": {"age": 50, "sex": "Male", "cp": "asymptomatic", "fbs": True}}
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["predicted_class"] == 0  # configured 0.7 threshold, not hardcoded 0.5
        assert body["model_score"] == 0.6
        assert body["threshold"] == 0.7
        assert "chol" in body["missing_features"]
        assert "age" not in body["missing_features"]
        assert "explanations" not in body
        assert "not necessarily calibrated" in body["disclaimer"]
        payload["model_id"] = "mlp"
        assert client.post("/api/predict", json=payload).status_code == 503


def test_invalid_score_is_server_error(tmp_path):
    settings = configuration(tmp_path)
    registry = ModelRegistry(settings)
    registry._initialized = True
    model = SimpleNamespace(predict_proba=lambda frame: np.array([[0, np.nan]]))
    registry.loaded["random_forest"] = SimpleNamespace(id="random_forest", model=model, positive_index=1, explanation_ready=False)
    with TestClient(create_app(settings, registry)) as client:
        response = client.post("/api/predict", json={"model_id": "random_forest", "features": {"age": 50, "sex": "Male", "cp": "asymptomatic"}})
        assert response.status_code == 500


def test_metrics_only_accept_genuine_well_formed_records(tmp_path):
    import json
    loaded = SimpleNamespace(id="mlp", directory=tmp_path)
    path = tmp_path / "mlp_metrics.json"
    assert ModelRegistry.recorded_metrics(loaded) == (None, None)
    path.write_text(json.dumps({"accuracy": 0.9}))
    assert ModelRegistry.recorded_metrics(loaded) == (None, None)
    path.write_text(json.dumps({"accuracy": 1.1, "provenance": "Original training run"}))
    assert ModelRegistry.recorded_metrics(loaded) == (None, None)
    path.write_text(json.dumps({"accuracy": 0.8, "provenance": "Original training run"}))
    assert ModelRegistry.recorded_metrics(loaded) == ({"accuracy": 0.8}, "Original training run")
