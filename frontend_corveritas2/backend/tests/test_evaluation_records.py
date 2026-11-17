"""Reject evaluation evidence for a different artifact or dataset."""
import json
from types import SimpleNamespace

import pytest

from app import evaluation_records
from app.evaluation_records import saved_evaluation, sha256


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    artifacts = tmp_path / "models"
    artifacts.mkdir()
    model = artifacts / "model.keras"
    preprocessor = artifacts / "preprocessor.pkl"
    model.write_bytes(b"model version one")
    preprocessor.write_bytes(b"preprocessor version one")
    dataset = tmp_path / "data.csv"
    dataset.write_text("sample,target\n1,0\n2,1\n")
    directory = tmp_path / "evaluations"
    directory.mkdir()
    monkeypatch.setattr(evaluation_records, "EVALUATION_DIR", directory)
    monkeypatch.setattr(evaluation_records, "DATASET_PATH", dataset)
    record = {
        "model_id": "mlp", "accuracy": 0.75, "average_precision": 0.82,
        "provenance": "Frozen model evaluated on the source-defined validation partition.",
        "evaluation": {"partition": "validation", "sample_count": 2, "threshold": 0.5, "used_during_training": True},
        "dataset": {"sha256": sha256(dataset)},
        "artifact_sha256": {path.name: sha256(path) for path in (model, preprocessor)},
    }
    path = directory / "mlp_metrics.json"
    path.write_text(json.dumps(record))
    return SimpleNamespace(loaded=SimpleNamespace(id="mlp", directory=artifacts),
                           required=[model.name, preprocessor.name], path=path,
                           record=record, model=model, preprocessor=preprocessor, dataset=dataset)


def test_matching_evaluation_includes_partition_and_source(evidence):
    result = saved_evaluation(evidence.loaded, evidence.required)
    assert result["metrics"] == {"accuracy": 0.75, "average_precision": 0.82}
    assert result["evaluation"]["partition"] == "validation"
    assert result["source"] == str(evidence.path)


@pytest.mark.parametrize("changed", ["model", "preprocessor", "dataset"])
def test_stale_evidence_is_rejected(evidence, changed):
    getattr(evidence, changed).write_bytes(b"different saved input")
    assert saved_evaluation(evidence.loaded, evidence.required) is None


@pytest.mark.parametrize("change", [
    {"model_id": "cnn"}, {"accuracy": float("nan")}, {"accuracy": True},
    {"accuracy": 1.5}, {"artifact_sha256": {}}, {"provenance": ""},
    {"evaluation": {"partition": "training", "sample_count": 2, "threshold": 0.5}},
])
def test_invalid_evidence_is_rejected(evidence, change):
    evidence.path.write_text(json.dumps(evidence.record | change))
    assert saved_evaluation(evidence.loaded, evidence.required) is None
