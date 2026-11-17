import logging
from typing import Any, Optional

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex recorded in dataset",
    "cp": "Chest pain category",
    "trestbps": "Resting blood pressure",
    "chol": "Serum cholesterol",
    "fbs": "Fasting blood sugar result",
    "restecg": "Resting ECG result",
    "thalch": "Maximum heart rate",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression",
    "slope": "ST segment slope",
    "ca": "Major vessels visualized",
    "thal": "Thal result",
}


def local_random_forest_explanation(
    estimator: Any,
    preprocessor: Any,
    frame: pd.DataFrame,
    positive_class_index: int,
    model_score: float,
) -> tuple[Optional[list[dict[str, Any]]], str]:
    try:
        import shap
    except ModuleNotFoundError as exc:
        if exc.name == "shap":
            return None, "Explanation currently unavailable (SHAP is not installed)."
        LOGGER.warning("Random Forest explanation dependency unavailable: %s", exc)
        return None, f"Explanation currently unavailable (missing dependency: {exc.name})."
    except ImportError as exc:
        LOGGER.warning("Random Forest explanation import failed: %s", exc)
        return None, "Explanation currently unavailable (SHAP dependencies are incompatible)."

    transformed = preprocessor.transform(frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype=np.float64)

    try:
        explainer = shap.TreeExplainer(
            estimator,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )
        raw_values = explainer.shap_values(transformed)
    except (ValueError, TypeError, AttributeError) as exc:
        LOGGER.warning("Random Forest local explanation unavailable: %s", exc)
        return None, "Explanation currently unavailable for this model artifact."

    values_array = np.asarray(raw_values)
    if values_array.ndim == 3:
        contributions = values_array[0, :, positive_class_index]
    elif values_array.ndim == 2:
        contributions = values_array[0]
    else:
        LOGGER.warning(
            "Unexpected SHAP output shape for Random Forest: %s",
            values_array.shape,
        )
        return None, "Explanation currently unavailable for this model artifact."

    base_values = np.asarray(explainer.expected_value)
    if base_values.ndim > 0:
        base_value = float(base_values.reshape(-1)[positive_class_index])
    else:
        base_value = float(base_values)
    if not np.isclose(base_value + contributions.sum(), model_score, atol=1e-3):
        LOGGER.warning("SHAP contributions failed the model-output additivity check.")
        return None, "Explanation currently unavailable for this model artifact."

    transformed_names = preprocessor.get_feature_names_out()
    grouped: dict[str, float] = {}
    for name, contribution in zip(transformed_names, contributions):
        raw_name = str(name).split("__", maxsplit=1)[-1]
        original_name = next(
            (
                feature
                for feature in FEATURE_LABELS
                if raw_name == feature or raw_name.startswith(feature + "_")
            ),
            None,
        )
        if original_name is not None:
            grouped[original_name] = grouped.get(original_name, 0.0) + float(
                contribution
            )

    results = [
        {"feature": FEATURE_LABELS[name], "contribution": value}
        for name, value in sorted(
            grouped.items(), key=lambda item: abs(item[1]), reverse=True
        )
        if abs(value) > 1e-8
    ]
    return results, "Local SHAP contributions for this input (not causal effects)."
