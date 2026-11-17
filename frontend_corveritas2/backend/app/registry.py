"""Load and validate each trusted local model exactly once at API startup."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib

from .config import Settings, configure_runtime_environment
from .evaluation_records import saved_evaluation
from .explanations import explain_forest
from .inference import positive_class_index, predict_score
from .schemas import CATEGORIES, FEATURE_ORDER, Features, features_to_frame

SPECS = {
    "mlp": {
        "name": "Multi-Layer Perceptron",
        "algorithm": "TensorFlow / Keras MLP",
        "files": ["corveritas_stage1.keras", "preprocessor.pkl"],
        "architecture": "Dense 128 → 64 → 32 → 1 sigmoid; batch normalization and dropout",
        "graph_dir": "model_mlp_graph",
        "script": "training_models.py",
    },
    "cnn": {
        "name": "1D Convolutional Network",
        "algorithm": "TensorFlow / Keras 1D CNN",
        "files": ["corveritas_stage2_cnn.keras", "preprocessor_stage2_cnn.pkl"],
        "architecture": "Conv1D 32 → 32 → 64 → 64; max pooling, normalization, dropout; Dense 64 → 1 sigmoid",
        "graph_dir": "model_cnn_graph",
        "script": "training_model2.py",
    },
    "random_forest": {
        "name": "Random Forest",
        "algorithm": "Scikit-learn Random Forest",
        "files": ["corveritas_stage3_random_forest.joblib"],
        "architecture": "Fitted tree ensemble with preprocessing; stratified five-fold randomized search on training data",
        "graph_dir": "model_random_forest_graph",
        "script": "training_model3.py",
    },
}


@dataclass
class LoadedModel:
    id: str
    model: Any
    preprocessor: Any
    directory: Path
    positive_index: int = 0
    processed_features: int = 0
    explanation_ready: bool = False


def validate_preprocessor(preprocessor):
    if list(preprocessor.feature_names_in_) != FEATURE_ORDER:
        raise ValueError("Saved feature order does not match the inspected 13-column schema.")
    categorical = preprocessor.named_transformers_["categorical"]
    names = next(list(columns) for name, _, columns in preprocessor.transformers_ if name == "categorical")
    encoder = categorical.named_steps["encoder"]
    if len(encoder.categories_) != len(names):
        raise ValueError("Saved encoder category groups differ from the validated schema.")
    for name, categories in zip(names, encoder.categories_):
        if name not in CATEGORIES or set(categories.tolist()) != set(CATEGORIES[name]):
            raise ValueError(f"Saved encoder categories differ from the validated schema for {name}.")
    if set(names) != set(CATEGORIES):
        raise ValueError("Saved categorical columns differ from the validated schema.")


class ModelRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.loaded: dict[str, LoadedModel] = {}
        self.errors: dict[str, str] = {}
        self._initialized = False

    def load_all(self):
        if self._initialized:
            return
        self._initialized = True
        configure_runtime_environment()
        # This is a compatibility probe only, never reported as a patient result.
        probe = features_to_frame(Features(age=50, sex="Male", cp="typical angina"))
        for model_id, spec in SPECS.items():
            directory = next((directory for directory in self.settings.model_dirs if all((directory / name).is_file() for name in spec["files"])), None)
            if directory is None:
                self.errors[model_id] = "Required trusted model files are missing: " + ", ".join(spec["files"])
                continue
            try:
                # Only load local files from configured directories. Never accept uploads.
                if model_id == "random_forest":
                    model = joblib.load(directory / spec["files"][0])
                    preprocessor = model.named_steps["preprocessor"]
                    forest = model.named_steps["model"]
                    if list(model.feature_names_in_) != FEATURE_ORDER or set(model.classes_.tolist()) != {0, 1}:
                        raise ValueError("Saved pipeline is not the expected binary cardiovascular classifier.")
                    loaded = LoadedModel(model_id, model, preprocessor, directory, positive_class_index(model.classes_))
                    if not hasattr(forest, "estimators_"):
                        raise ValueError("Expected a fitted tree ensemble.")
                else:
                    import tensorflow as tf
                    model = tf.keras.models.load_model(directory / spec["files"][0], compile=False, safe_mode=True)
                    preprocessor = joblib.load(directory / spec["files"][1])
                    loaded = LoadedModel(model_id, model, preprocessor, directory)
                validate_preprocessor(preprocessor)
                transformed = preprocessor.transform(probe)
                loaded.processed_features = int(transformed.shape[1])
                score = predict_score(loaded, probe)
                if model_id == "random_forest":
                    # Explanations have their own capability; failure never fabricates values.
                    try:
                        explain_forest(model, probe, loaded.positive_index, score)
                        loaded.explanation_ready = True
                    except (ValueError, AttributeError, IndexError, TypeError):
                        loaded.explanation_ready = False
                self.loaded[model_id] = loaded
            except Exception as error:
                # Record model-specific errors without stopping models which do work.
                self.errors[model_id] = f"Load or compatibility check failed ({type(error).__name__}): {str(error)[:300]}"

    @property
    def default_model_id(self) -> Optional[str]:
        return self.settings.default_model if self.settings.default_model in self.loaded else None

    def plots(self, model_id):
        spec = SPECS[model_id]
        loaded = self.loaded.get(model_id)
        directories = (loaded.directory,) if loaded else self.settings.model_dirs
        directory = next((directory / spec["graph_dir"] for directory in directories if (directory / spec["graph_dir"]).is_dir()), None)
        return {path.name: path for path in sorted(directory.glob("*.png"))} if directory else {}

    def metadata(self):
        models = []
        for model_id, spec in SPECS.items():
            loaded = self.loaded.get(model_id)
            architecture = spec["architecture"]
            if loaded and model_id == "random_forest":
                forest = loaded.model.named_steps["model"]
                architecture = f"{forest.n_estimators} trees; max depth {forest.max_depth}; min leaf {forest.min_samples_leaf}; complete fitted preprocessing pipeline"
            evaluation = saved_evaluation(loaded, spec["files"])
            metrics = evaluation["metrics"] if evaluation else None
            provenance = evaluation["provenance"] if evaluation else None
            models.append({
                "id": model_id,
                "name": spec["name"],
                "algorithm": spec["algorithm"],
                "available": loaded is not None,
                "reason": None if loaded else self.errors.get(model_id, "Not initialized"),
                "threshold": self.settings.thresholds[model_id],
                "architecture": architecture,
                "input_features": 13,
                "processed_features": loaded.processed_features if loaded else None,
                "metrics": metrics,
                "metrics_source": evaluation["source"] if evaluation else None,
                "metrics_evaluation": evaluation["evaluation"] if evaluation else None,
                "metrics_note": provenance or (
                    "Numeric evaluation metrics were printed during original training but not "
                    "saved as machine-readable records. Existing training and held-out evaluation "
                    "figures are shown without recomputing test scores."
                ),
                "plots": [
                    {"name": name.removesuffix(".png").replace("_", " ").title(),
                     "url": f"/api/models/{model_id}/plots/{name}"}
                    for name in self.plots(model_id)
                ],
                "explanation_method": "Exact tree-path probability decomposition" if loaded and loaded.explanation_ready else None,
                "training_command": f"python training/train.py --model {model_id}",
                "required_artifacts": spec["files"],
                "artifact_directory": str(loaded.directory) if loaded else None,
            })
        return {"models": models, "default_model_id": self.default_model_id, "configured_default_model_id": self.settings.default_model}

    @staticmethod
    def recorded_metrics(loaded):
        """Accept training records or fingerprint-matched application evaluations."""
        if loaded is None:
            return None, None
        record = saved_evaluation(loaded, SPECS[loaded.id]["files"])
        return (record["metrics"], record["provenance"]) if record else (None, None)
