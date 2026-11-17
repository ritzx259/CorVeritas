"""Validate the metric definitions and source split inspection without rescoring artifacts."""
import importlib.util

import numpy as np
import pytest

from app.config import APP_ROOT, REPO_ROOT

spec = importlib.util.spec_from_file_location("record_evaluation_metrics", APP_ROOT / "scripts" / "record_evaluation_metrics.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


@pytest.mark.parametrize("script,partition", [
    ("training_models.py", "validation"), ("training_model2.py", "test"), ("training_model3.py", "test"),
])
def test_protocol_matches_original_training_source(script, partition):
    protocol = evaluation.source_protocol(REPO_ROOT / "models" / script)
    assert protocol == {"test_size": 0.2, "random_state": 42, "partition": partition,
                        "threshold": 0.5, "used_during_training": partition == "validation"}


def test_metrics_use_thresholded_classes_and_continuous_ranking_scores():
    metrics, confusion = evaluation.calculate_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.5, 0.9]), 0.5)
    assert metrics == pytest.approx({"accuracy": 0.75, "precision": 2 / 3, "recall": 1,
                                    "f1": 0.8, "roc_auc": 0.75, "average_precision": 5 / 6})
    assert confusion == [[1, 1], [0, 2]]


def test_changed_source_protocol_requires_review(tmp_path):
    path = tmp_path / "training.py"
    path.write_text("print('training source without the expected split')")
    with pytest.raises(ValueError, match="Review changed"):
        evaluation.source_protocol(path)
