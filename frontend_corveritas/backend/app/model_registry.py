import logging
from pathlib import Path
from typing import Any, Optional

import joblib

from app.config import (
    DEFAULT_MODEL_ID,
    FEATURE_ORDER,
    MODEL_ARTIFACTS,
    MODEL_DIR,
)


LOGGER = logging.getLogger(__name__)
THRESHOLD = 0.5


class LoadedModel:
    def __init__(
        self,
        model_id: str,
        metadata: dict[str, Any],
        model: Any,
        preprocessor: Optional[Any] = None,
    ) -> None:
        self.model_id = model_id
        self.metadata = metadata
        self.model = model
        self.preprocessor = preprocessor


class ModelRegistry:
    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        default_model_id: str = DEFAULT_MODEL_ID,
    ) -> None:
        self.model_dir = model_dir
        self.default_model_id = default_model_id
        self.available: dict[str, LoadedModel] = {}
        self.errors: dict[str, str] = {}
        self._load_all()

    def _load_all(self) -> None:
        for model_id, metadata in MODEL_ARTIFACTS.items():
            model_path = self.model_dir / metadata["model_file"]
            preprocessor_file = metadata["preprocessor_file"]
            preprocessor_path = (
                self.model_dir / preprocessor_file if preprocessor_file else None
            )
            required_paths = [model_path]
            if preprocessor_path is not None:
                required_paths.append(preprocessor_path)

            missing = [path.name for path in required_paths if not path.is_file()]
            if missing:
                self.errors[model_id] = (
                    f"Missing artifact(s): {', '.join(missing)}. "
                    f"Train it with `{metadata['training_command']}` from the project root."
                )
                continue

            try:
                if model_id == "random_forest":
                    model = joblib.load(model_path)
                    preprocessor = model.named_steps["preprocessor"]
                    estimator = model.named_steps["model"]
                    if list(model.feature_names_in_) != FEATURE_ORDER:
                        raise ValueError("Random Forest input feature order does not match the API schema.")
                    self._validate_classes(estimator)
                else:
                    preprocessor = joblib.load(preprocessor_path)
                    if list(preprocessor.feature_names_in_) != FEATURE_ORDER:
                        raise ValueError("Preprocessor input feature order does not match the API schema.")
                    import tensorflow as tf

                    model = tf.keras.models.load_model(model_path, compile=False)
                    input_size = len(preprocessor.get_feature_names_out())
                    expected = (
                        (None, input_size)
                        if model_id == "mlp"
                        else (None, input_size, 1)
                    )
                    actual = tuple(model.input_shape)
                    if actual != expected:
                        raise ValueError(
                            f"Model input shape {actual} does not match preprocessor output {expected}."
                        )
                self.available[model_id] = LoadedModel(
                    model_id,
                    metadata,
                    model,
                    preprocessor,
                )
                LOGGER.info("Loaded %s model from %s", model_id, model_path)
            except Exception as exc:
                self.errors[model_id] = (
                    f"Could not load {model_id}: {type(exc).__name__}: {exc}. "
                    f"Check its matching artifact files and rerun `{metadata['training_command']}`."
                )
                LOGGER.exception("Failed to load model %s", model_id)

    @staticmethod
    def _validate_classes(estimator: Any) -> None:
        classes = list(estimator.classes_)
        if classes != [0, 1]:
            raise ValueError(
                "Random Forest must expose class ordering [0, 1]; "
                f"found {classes!r}."
            )

    def get(self, model_id: str) -> Optional[LoadedModel]:
        return self.available.get(model_id)

    def list_models(self) -> list[dict[str, Any]]:
        result = []
        for model_id, metadata in MODEL_ARTIFACTS.items():
            loaded = self.available.get(model_id)
            result.append(
                {
                    "id": model_id,
                    "name": metadata["name"],
                    "architecture": metadata["architecture"],
                    "description": metadata["description"],
                    "available": loaded is not None,
                    "error": self.errors.get(model_id),
                    "is_default": model_id == self.default_model_id
                    and loaded is not None,
                    "threshold": THRESHOLD,
                    "metrics": metadata.get(
                        "metrics",
                        {
                            "accuracy": None,
                            "precision": None,
                            "recall": None,
                            "f1": None,
                            "roc_auc": None,
                            "average_precision": None,
                        },
                    ),
                    "metrics_source": metadata.get(
                        "metrics_source",
                        "Test metrics are not recorded in the saved artifacts.",
                    ),
                    "graphs": self._available_graphs(model_id, metadata),
                    "input_features": FEATURE_ORDER,
                    "explanation": (
                        "Patient-specific SHAP feature contributions are attempted at prediction time."
                        if model_id == "random_forest"
                        else "Local explanation unavailable for this neural network."
                    ),
                }
            )
        return result

    @staticmethod
    def _available_graphs(model_id: str, metadata: dict[str, Any]) -> list[str]:
        directory = MODEL_DIR / metadata["graph_directory"]
        return [
            f"/api/model-assets/{model_id}/{filename}"
            for filename in metadata["graphs"]
            if (directory / filename).is_file()
        ]
