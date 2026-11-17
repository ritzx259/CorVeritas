#!/usr/bin/env python3
"""Run an unchanged original training script with local, portable paths.

This launcher changes only path configuration assignments in the in-memory AST.
It never edits the original training source or writes artifacts into ../models.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile


APP_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = APP_ROOT.parent
SCRIPTS = {
    "mlp": ("training_models.py", "corveritas_stage1.keras", "preprocessor.pkl"),
    "cnn": (
        "training_model2.py",
        "corveritas_stage2_cnn.keras",
        "preprocessor_stage2_cnn.pkl",
    ),
    "random_forest": (
        "training_model3.py",
        "corveritas_stage3_random_forest.joblib",
        "preprocessor_stage3_random_forest.pkl",
    ),
}


class LocalTrainingPaths(ast.NodeTransformer):
    """Replace the known configuration assignments, preserving training logic."""

    def __init__(self, dataset: Path, output: Path, use_path_objects: bool):
        self.values = {
            "ROOT_DIR": APP_ROOT,
            "DATA_PATH": dataset,
            "MODEL_DIR": output,
        }
        self.use_path_objects = use_path_objects
        self.changed: set[str] = set()

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in self.values:
                value: ast.expr = ast.Constant(str(self.values[target.id]))
                if self.use_path_objects:
                    value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[value], keywords=[])
                node.value = ast.copy_location(value, node.value)
                self.changed.add(target.id)
        return node


def training_program(model: str, dataset: Path, output: Path):
    """Validate and compile a source copy; no TensorFlow import or training yet."""
    source = REPOSITORY_ROOT / "models" / SCRIPTS[model][0]
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    paths = LocalTrainingPaths(dataset, output, use_path_objects=model != "mlp")
    tree = paths.visit(tree)
    expected = {"DATA_PATH", "MODEL_DIR"}
    if model != "mlp":
        expected.add("ROOT_DIR")
    if paths.changed != expected:
        raise RuntimeError(
            f"Original script configuration changed: expected {sorted(expected)}, "
            f"found {sorted(paths.changed)}. Review the wrapper before training."
        )
    ast.fix_missing_locations(tree)
    return source, compile(tree, str(source), "exec")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(SCRIPTS))
    parser.add_argument(
        "--show-plan", action="store_true",
        help="Validate and display path changes without importing ML libraries or training.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace previously trained local artifacts after successful training.",
    )
    args = parser.parse_args()
    dataset = REPOSITORY_ROOT / "data" / "heart_disease_uci.csv"
    destination = APP_ROOT / "models"
    if not dataset.is_file():
        parser.error(f"Dataset is missing: {dataset}")
    try:
        source, _ = training_program(args.model, dataset, destination)
    except (OSError, SyntaxError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Original source (read only): {source}")
    print(f"Dataset (read only):        {dataset}")
    print(f"Artifact destination:      {destination}")
    print("Changes: portable data/output paths and noninteractive plotting only.")
    if args.show_plan:
        print("Plan validated. No training was performed.")
        return 0
    expected_outputs = SCRIPTS[args.model][1:]
    existing = [name for name in expected_outputs if (destination / name).exists()]
    if existing and not args.overwrite:
        parser.error("Local trained files already exist: " + ", ".join(existing) + ". Use --overwrite to replace them.")

    destination.mkdir(parents=True, exist_ok=True)
    cache = APP_ROOT / ".cache"
    for name in ("matplotlib", "joblib", "tmp"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["KERAS_HOME"] = str(cache / "keras")
    os.environ["XDG_CACHE_HOME"] = str(cache)
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ["JOBLIB_TEMP_FOLDER"] = str(cache / "joblib")
    os.environ["TMPDIR"] = str(cache / "tmp")
    tempfile.tempdir = str(cache / "tmp")
    sys.dont_write_bytecode = True

    # Publish only after a successful complete run. A failed script cannot leave
    # a new preprocessor paired with an older saved neural network in models/.
    with tempfile.TemporaryDirectory(prefix=f".training-{args.model}-", dir=destination) as staging:
        _, program = training_program(args.model, dataset, Path(staging))
        namespace = {"__name__": "__main__", "__file__": str(source)}
        exec(program, namespace)
        absent = [name for name in expected_outputs if not (Path(staging) / name).is_file()]
        if absent:
            raise RuntimeError("Training completed without expected files: " + ", ".join(absent))
        # Persist the metrics the original script has just calculated once.
        # Do not rerun predictions or consult the held-out test data here.
        metric_names = {
            "accuracy": "accuracy", "precision": "precision", "recall": "recall",
            "f1": "f1", "roc_auc": "auc", "average_precision": "average_precision",
        }
        metrics = {}
        for public_name, source_name in metric_names.items():
            if source_name in namespace:
                value = float(namespace[source_name])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise RuntimeError(f"Original script produced an invalid {public_name} metric.")
                metrics[public_name] = value
        metrics["provenance"] = (
            f"Calculated by original {source.name} during this training run; "
            "stratified 20% held-out test partition, random_state=42; "
            f"saved {datetime.now(timezone.utc).isoformat()}. "
            f"Source SHA256: {hashlib.sha256(source.read_bytes()).hexdigest()}; "
            f"dataset SHA256: {hashlib.sha256(dataset.read_bytes()).hexdigest()}. "
            "Not independent external or clinical validation."
        )
        (Path(staging) / f"{args.model}_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        for artifact in Path(staging).iterdir():
            target = destination / artifact.name
            if artifact.is_dir():
                shutil.copytree(artifact, target, dirs_exist_ok=True)
            else:
                artifact.replace(target)
    print(f"\nTraining completed. Saved artifacts and plots under {destination}")
    print("Restart the backend to load the new files. This does not establish clinical validity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
