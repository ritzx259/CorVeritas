"""Verify original training logic is preserved while writes are redirected."""
import ast
import hashlib
import importlib.util
from pathlib import Path

import pytest


WRAPPER = Path(__file__).resolve().parents[1] / "train.py"
spec = importlib.util.spec_from_file_location("corveritas_training_wrapper", WRAPPER)
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def without_configuration(tree):
    tree.body = [
        node for node in tree.body
        if not (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {"ROOT_DIR", "DATA_PATH", "MODEL_DIR"}
            for target in node.targets
        ))
    ]
    return ast.dump(tree, include_attributes=False)


@pytest.mark.parametrize("model", ["mlp", "cnn", "random_forest"])
def test_original_algorithm_preserved_and_paths_stay_local(model):
    source = train.REPOSITORY_ROOT / "models" / train.SCRIPTS[model][0]
    original = source.read_bytes()
    dataset = train.REPOSITORY_ROOT / "data" / "heart_disease_uci.csv"
    output = train.APP_ROOT / "models"
    original_tree = ast.parse(original)
    transformed = train.LocalTrainingPaths(dataset, output, model != "mlp").visit(ast.parse(original))
    assignments = {}
    for node in transformed.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"ROOT_DIR", "DATA_PATH", "MODEL_DIR"}:
                    value = node.value.args[0] if isinstance(node.value, ast.Call) else node.value
                    assignments[target.id] = ast.literal_eval(value)
    assert Path(assignments["DATA_PATH"]) == dataset
    assert Path(assignments["MODEL_DIR"]).is_relative_to(train.APP_ROOT)
    if model != "mlp":
        assert Path(assignments["ROOT_DIR"]) == train.APP_ROOT
    assert without_configuration(transformed) == without_configuration(original_tree)
    _, program = train.training_program(model, dataset, output)
    assert program.co_filename == str(source)
    assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(original).digest()
