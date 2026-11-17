"""Artifact persistence and prediction primitives, without a training dependency."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch

try:
    from .config import FEATURES, LABELS, MODELS, VERSION
    from .model import MultiLabelMLP, neural_scores
    from .preprocess import clean_features
except ImportError:
    from config import FEATURES, LABELS, MODELS, VERSION
    from model import MultiLabelMLP, neural_scores
    from preprocess import clean_features


def write_json(path, data):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(data, indent=2, default=convert, allow_nan=False))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline_scores(model, matrix):
    # MultiOutputClassifier returns one n-by-2 array per label, not softmax scores.
    return np.column_stack([p[:, list(est.classes_).index(1)] for p, est in zip(model.predict_proba(matrix), model.estimators_)])


class ArtifactPredictor:
    """Loads trusted local artifacts only. joblib files must never be accepted from strangers."""
    def __init__(self, model_dir=MODELS, model_name=None):
        self.directory = Path(model_dir)
        path = self.directory / "metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"No trained Stage 2 artifacts at {path}. Run train.py first.")
        self.metadata = json.loads(path.read_text())
        if self.metadata["version"] != VERSION or self.metadata["features"] != FEATURES or self.metadata["labels"] != LABELS:
            raise ValueError("Artifact schema/version mismatch. Use the matching Stage 2 code and artifacts.")
        for name, expected in self.metadata["artifact_sha256"].items():
            if file_hash(self.directory / name) != expected:
                raise ValueError(f"Artifact checksum mismatch: {name}")
        self.name = model_name or self.metadata["selected_model"]
        if self.name not in ("baseline", "neural"):
            raise ValueError("model_name must be baseline or neural.")
        self.preprocessor = joblib.load(self.directory / "preprocessor.joblib")
        if list(self.preprocessor.get_feature_names_out()) != self.metadata["transformed_features"]:
            raise ValueError("Preprocessing feature ordering does not match saved metadata.")
        self.thresholds = np.array(self.metadata["models"][self.name]["thresholds"], dtype=float)
        self.background = joblib.load(self.directory / "explanation_background.joblib")
        if self.name == "baseline":
            self.model = joblib.load(self.directory / "baseline.joblib")
        else:
            self.model = MultiLabelMLP(**self.metadata["architecture"])
            self.model.load_state_dict(torch.load(self.directory / "neural.pt", map_location="cpu", weights_only=True))
            self.model.eval()

    def scores(self, frame, validate=True):
        inputs = clean_features(frame, strict=True) if validate else frame[FEATURES]
        matrix = np.asarray(self.preprocessor.transform(inputs), dtype=np.float32)
        return baseline_scores(self.model, matrix) if self.name == "baseline" else neural_scores(self.model, matrix)
