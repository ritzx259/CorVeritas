#!/usr/bin/env python3
"""Record metrics from frozen local artifacts and the original script-defined split.

No training or threshold selection occurs. Matching saved reports are reused;
existing stale reports are never silently overwritten. Nothing is written to
the repository's original models or dataset directories.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "backend"))

from app.config import REPO_ROOT, Settings, configure_runtime_environment

configure_runtime_environment()

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from app.evaluation_records import DATASET_PATH, EVALUATION_DIR, sha256
from app.inference import transformed_neural_input
from app.registry import ModelRegistry, SPECS
from app.schemas import FEATURE_ORDER


def source_protocol(source):
    """Inspect the current source without importing or executing training code."""
    tree = ast.parse(source.read_text())
    constants = {}
    splits = []
    threshold = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "train_test_split":
            splits.append(node)
        if any(isinstance(target, ast.Name) and target.id == "predictions" for target in node.targets):
            # All three inspected sources use (probabilities >= 0.50).astype(int).
            comparison = node.value.func.value if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) else None
            if isinstance(comparison, ast.Compare) and len(comparison.ops) == 1 and isinstance(comparison.ops[0], ast.GtE):
                threshold = float(ast.literal_eval(comparison.comparators[0]))
    if len(splits) != 1 or threshold is None or not 0 < threshold < 1:
        raise ValueError(f"Review changed split/threshold logic in {source.name} before evaluation.")
    split = splits[0]
    if [arg.id for arg in split.value.args if isinstance(arg, ast.Name)] != ["X", "y"]:
        raise ValueError("Expected a stratified split of the original X and y.")
    options = {}
    for option in split.value.keywords:
        options[option.arg] = constants[option.value.id] if isinstance(option.value, ast.Name) and option.value.id in constants else option.value
    stratify = options.pop("stratify", None)
    if not isinstance(stratify, ast.Name) or stratify.id != "y" or set(options) != {"test_size", "random_state"}:
        raise ValueError("Review changed split options before evaluation.")
    options = {name: ast.literal_eval(value) if isinstance(value, ast.AST) else value for name, value in options.items()}
    targets = [name.id for name in split.targets[0].elts]
    partition = "validation" if targets == ["X_train", "X_val", "y_train", "y_val"] else "test" if targets == ["X_train", "X_test", "y_train", "y_test"] else None
    if partition is None:
        raise ValueError("Unrecognized evaluation partition.")
    return {**options, "partition": partition, "threshold": threshold,
            "used_during_training": partition == "validation"}


def calculate_metrics(labels, scores, threshold):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(scores) != len(labels) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid binary model scores.")
    predictions = (scores >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }, confusion_matrix(labels, predictions, labels=[0, 1]).tolist()


def main():
    dataset = pd.read_csv(DATASET_PATH)
    features = dataset.drop(columns=["id", "dataset", "num"])
    labels = (dataset["num"] > 0).astype(int)
    if features.columns.tolist() != FEATURE_ORDER:
        raise ValueError("Dataset columns differ from the source model schema.")
    registry = ModelRegistry(Settings.from_env())
    registry.load_all()
    if set(registry.loaded) != set(SPECS):
        raise RuntimeError(f"Cannot evaluate unavailable models: {registry.errors}")
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    for model_id, loaded in registry.loaded.items():
        destination = EVALUATION_DIR / f"{model_id}_metrics.json"
        if destination.exists():
            if ModelRegistry.recorded_metrics(loaded)[0] is None:
                raise RuntimeError(f"Stale report at {destination}; review/archive it before a new evaluation.")
            print(f"{model_id}: reused saved metrics; no new predictions.")
            continue
        source = REPO_ROOT / "models" / SPECS[model_id]["script"]
        protocol = source_protocol(source)
        train, evaluation, _, truth = train_test_split(features, labels, test_size=protocol["test_size"],
                                                      random_state=protocol["random_state"], stratify=labels)
        if model_id == "random_forest":
            scores = loaded.model.predict_proba(evaluation)[:, loaded.positive_index]
        else:
            matrix = transformed_neural_input(loaded.preprocessor, loaded.model, evaluation, cnn=model_id == "cnn")
            scores = np.asarray(loaded.model(matrix, training=False)).reshape(-1)
        metrics, confusion = calculate_metrics(truth, scores, protocol["threshold"])
        partition = protocol["partition"]
        caveat = ("This validation partition was used during model training, so these are not independent test scores. "
                  if protocol["used_during_training"] else "The partition is reconstructed from the original training script. ")
        created = datetime.now(timezone.utc).isoformat()
        report = {
            "model_id": model_id, **metrics,
            "provenance": f"Evaluated the saved {model_id} artifact on the source-defined {len(truth)}-row {partition} partition "
                          f"({protocol['test_size']:.0%}, stratified, random_state={protocol['random_state']}, threshold={protocol['threshold']:.2f}). "
                          f"{caveat}No retraining or threshold tuning. Saved {created}. Not external or clinical validation.",
            "evaluation": {**protocol, "sample_count": len(truth), "training_sample_count": len(train)},
            "dataset": {"path": str(DATASET_PATH.relative_to(REPO_ROOT)), "sha256": sha256(DATASET_PATH), "row_count": len(dataset)},
            "training_source": {"path": str(source.relative_to(REPO_ROOT)), "sha256": sha256(source)},
            "artifact_sha256": {name: sha256(loaded.directory / name) for name in SPECS[model_id]["files"]},
            "evaluation_indices_sha256": hashlib.sha256(np.asarray(evaluation.index, dtype="<i8").tobytes()).hexdigest(),
            "confusion_matrix": {"label_order": [0, 1], "values": confusion},
            "class_counts": {str(label): int(count) for label, count in truth.value_counts().sort_index().items()},
            "software": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
            "created_at": created,
        }
        # Exclusive creation prevents accidental replacement of evaluation evidence.
        with destination.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"{model_id} ({partition}, n={len(truth)}): {json.dumps(metrics)}", flush=True)


if __name__ == "__main__":
    main()
