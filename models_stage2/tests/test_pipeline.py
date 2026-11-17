"""Integrity checks use artificial fixtures only, never as research training data."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

from models_stage2.artifacts import ArtifactPredictor, file_hash, write_json
from models_stage2.config import FEATURES, LABELS, LIMITATIONS, MODULE_COLUMNS, VERSION
from models_stage2.download_data import read_module
from models_stage2.evaluate import condition_metrics, evaluate_saved, select_thresholds
from models_stage2.explain import exact_grouped_shap
from models_stage2.model import MultiLabelMLP, seed_everything
from models_stage2.predict import Stage2Predictor, patient_frame
from models_stage2.preprocess import build_cohort, clean_features, make_preprocessor, map_diagnosis, split_participants
from models_stage2.train import fit_neural


@pytest.fixture
def clinical_frame():
    return pd.DataFrame({"age": [25, 35, 45, 55, 65, 75, 50, 70], "bmi": [22, 24, 26, 28, 30, 32, 34, 36],
        "systolic_bp": [110, 115, 120, 125, 130, 135, 140, 145], "diastolic_bp": [70, 72, 74, 76, 78, 80, 82, 84],
        "total_cholesterol": [150, 170, 190, 210, 230, 250, 180, 220],
        "sex": ["Male", "Female"]*4, "diabetes_history": ["No", "Yes", "Borderline", np.nan]*2,
        "smoking_history": ["Yes", "No"]*4})[FEATURES]


def test_diagnosis_unknown_never_becomes_negative():
    mapped = map_diagnosis(pd.Series([1, 2, 7, 9, np.nan, 0, 3]))
    assert mapped.iloc[:2].tolist() == [1, 0]
    assert mapped.iloc[2:].isna().all()


def test_merge_adult_labels_bp_and_allowlist():
    ids = [1, 2, 3, 4]
    modules = {name: pd.DataFrame({c: np.full(4, np.nan) for c in columns}).assign(SEQN=ids)
               for name, columns in MODULE_COLUMNS.items()}
    demo = modules["DEMO_L"]
    demo["RIDAGEYR"] = [19, 40, 60, 80]
    demo["RIAGENDR"] = [1, 2, 1, 2]
    demo["RIDSTATR"] = 2
    mcq = modules["MCQ_L"]
    mcq["MCQ160c"], mcq["MCQ160b"], mcq["MCQ160e"] = [2, 1, 7, 2], [2, 1, 2, 2], [2, 2, 1, 2]
    modules["BPXO_L"]["BPXOSY1"] = [120, 120, 130, np.nan]
    modules["BPXO_L"]["BPXOSY2"] = [120, 140, 130, np.nan]
    modules["BPXO_L"]["BPXOSY3"] = [120, 0, 130, np.nan]
    modules["BPXO_L"]["BPXODI1"] = [80, 80, 85, 80]
    modules["BPXO_L"]["BPXODI2"] = [80, 90, 85, np.nan]
    modules["DIQ_L"]["DIQ010"] = [2, 3, 2, 9]
    modules["SMQ_L"]["SMQ020"] = [2, 1, 2, 9]
    cohort, audit = build_cohort(modules)
    assert cohort.SEQN.tolist() == [2, 4]
    assert audit["excluded_incomplete_or_invalid_labels"] == 1
    assert cohort.loc[0, LABELS].tolist() == [1, 1, 0]
    assert cohort.loc[0, "systolic_bp"] == 130
    assert cohort.loc[0, "diastolic_bp"] == 85
    assert np.isnan(cohort.loc[1, "diastolic_bp"])
    assert cohort.loc[0, "diabetes_history"] == "Borderline"
    assert not any(c.startswith("MCQ") or c in ("ca", "thal", "oldpeak", "cp", "SEQN") for c in FEATURES)


def test_preprocessing_fits_training_only_and_preserves_order(clinical_frame):
    training = clinical_frame.iloc[:6].copy()
    training.loc[0, "bmi"] = np.nan
    transformer = make_preprocessor().fit(training)
    learned = transformer.named_transformers_["numeric"].named_steps["impute"].statistics_.copy()
    held_out = clinical_frame.iloc[6:].copy()
    held_out["bmi"] = 99
    transformed = transformer.transform(held_out)
    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(learned, transformer.named_transformers_["numeric"].named_steps["impute"].statistics_)
    assert learned[1] == np.median([24, 26, 28, 30, 32])
    assert list(transformer.feature_names_in_) == FEATURES
    assert len(transformer.get_feature_names_out()) == transformed.shape[1]


@pytest.mark.parametrize("fallback", [False, True])
def test_split_participants_disjoint_reproducible(clinical_frame, fallback):
    frame = pd.concat([clinical_frame]*60, ignore_index=True)
    frame["SEQN"] = np.arange(len(frame))
    combos = np.array([[int(i & 1 > 0), int(i & 2 > 0), int(i & 4 > 0)] for i in range(8)])
    frame[LABELS] = np.tile(combos, (60, 1))
    first, _ = split_participants(frame, force_fallback=fallback)
    second, _ = split_participants(frame, force_fallback=fallback)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
        assert abs(len(first[name])/len(frame)-(.7 if name == "train" else .15)) < .02
    assert not set(first["train"]) & set(first["test"])
    assert not set(first["validation"]) & set(first["test"])


def test_threshold_selection_and_specificity():
    truth = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]])
    scores = np.array([[.01]*3, [.02]*3, [.10]*3, [.20]*3])
    thresholds = select_thresholds(truth, scores)
    assert thresholds == [.10]*3
    result = condition_metrics(truth[:, 0], scores[:, 0], thresholds[0])
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]
    assert result["specificity"] == 1 and result["f1"] == 1
    assert result["brier_score"] > 0


@pytest.mark.parametrize("patient,match", [({"age": 19}, "age"), ({"age": None}, "age"),
    ({"age": float("inf")}, "age"), ({"age": 55, "systolic_bp": 70, "diastolic_bp": 90}, "systolic"),
    ({"age": 55, "total_cholesterol": "240 mmol"}, "total_cholesterol"),
    ({"age": 55, "sex": "other"}, "sex"), ({"age": 55, "ca": 1}, "Unexpected"),
    ({"age": 55, "bmi": True}, "boolean")])
def test_input_validation(patient, match):
    with pytest.raises(ValueError, match=match):
        patient_frame(patient)


def test_input_aliases_and_age_topcode():
    frame = patient_frame({"age": 90, "sex": "male", "smoking_history": "Former smoker"})
    assert frame.age.iloc[0] == 80
    assert frame.smoking_history.iloc[0] == "Yes"
    assert frame.bmi.isna().all()


def test_html_xpt_rejected(tmp_path):
    path = tmp_path / "MCQ_L.xpt"
    path.write_text("<html>access denied</html>")
    with pytest.raises(ValueError, match="not a SAS"):
        read_module(path, "MCQ_L")


@pytest.fixture
def artifact_dir(tmp_path, clinical_frame):
    seed_everything()
    transformer = make_preprocessor().fit(clinical_frame)
    x = transformer.transform(clinical_frame)
    y = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    baseline = MultiOutputClassifier(LogisticRegression()).fit(x, y)
    neural = MultiLabelMLP(x.shape[1])
    joblib.dump(transformer, tmp_path / "preprocessor.joblib")
    joblib.dump(baseline, tmp_path / "baseline.joblib")
    joblib.dump(clinical_frame.iloc[:3], tmp_path / "explanation_background.joblib")
    torch.save(neural.state_dict(), tmp_path / "neural.pt")
    metadata = {"version": VERSION, "features": FEATURES, "labels": LABELS, "selected_model": "baseline",
        "transformed_features": list(transformer.get_feature_names_out()), "architecture": {"input_dim": x.shape[1]},
        "models": {n: {"thresholds": [.12, .24, .36]} for n in ("baseline", "neural")},
        "run_id": "test-fixture-only", "limitations": LIMITATIONS,
        "artifact_sha256": {name: file_hash(tmp_path/name) for name in
            ("preprocessor.joblib", "baseline.joblib", "explanation_background.joblib", "neural.pt")}}
    write_json(tmp_path / "metadata.json", metadata)
    return tmp_path


@pytest.mark.parametrize("model_name", ["baseline", "neural"])
def test_save_load_predictions_and_exact_shap(artifact_dir, clinical_frame, model_name):
    predictor = ArtifactPredictor(artifact_dir, model_name)
    sample = clinical_frame.iloc[:1]
    explanation = exact_grouped_shap(predictor, sample)
    assert np.max(explanation["reconstruction_error"]) < 1e-6
    np.testing.assert_allclose(explanation["scores"], predictor.scores(sample)[0], atol=1e-6)
    result = Stage2Predictor(artifact_dir, model_name).predict({"age": 55, "sex": "Male", "smoking_history": "Former smoker"})
    assert len(result["conditions"]) == 3
    assert [c["selected_threshold"] for c in result["conditions"]] == [.12, .24, .36]
    assert len(result["conditions"][0]["feature_contributions"]) == 8
    json.dumps(result, allow_nan=False)


def test_artifact_corruption_rejected(artifact_dir):
    with (artifact_dir / "neural.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        ArtifactPredictor(artifact_dir)


def test_cached_evaluation_does_not_rescore_or_need_training_data(tmp_path):
    result = {"run_id": "cached-only", "models": {}, "selected_model_from_validation": "baseline"}
    metadata = {"run_id": "cached-only", "result_dir": str(tmp_path), "selected_model": "baseline", "limitations": LIMITATIONS}
    write_json(tmp_path / "metadata.json", metadata)
    write_json(tmp_path / "test_evaluation.json", result)
    # No model, preprocessor or cohort files exist. Cached evaluation must still work.
    assert evaluate_saved(tmp_path) == result
    assert (tmp_path / "report.md").exists()


def test_neural_training_backprop_and_early_stop(clinical_frame):
    torch.set_num_threads(1)
    x = np.asarray(make_preprocessor().fit_transform(clinical_frame), dtype=np.float32)
    y = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    model, history, info = fit_neural(x, y, x, y, torch.device("cpu"), batch_size=3, epochs=2, patience=1)
    assert 1 <= len(history) <= 2
    assert info["best_epoch"] >= 1
    assert np.isfinite(history[-1]["training_optimization_loss"])
    assert model(torch.as_tensor(x[:1])).shape == (1, 3)
