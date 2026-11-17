from typing import Any

import numpy as np
import pandas as pd

from app.config import FEATURE_ORDER, MODEL_LIMITATIONS
from app.explanations import local_random_forest_explanation
from app.model_registry import THRESHOLD, LoadedModel
from app.schemas import PredictionResponse


def build_feature_frame(features: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [[features[name] for name in FEATURE_ORDER]],
        columns=FEATURE_ORDER,
    )
    frame["age"] = pd.Series([int(features["age"])], dtype="int64")
    for name in ("trestbps", "chol", "thalch", "oldpeak", "ca"):
        value = features[name]
        frame[name] = pd.Series(
            [np.nan if value is None else float(value)],
            dtype="float64",
        )
    for name in ("sex", "cp", "fbs", "restecg", "exang", "slope", "thal"):
        value = features[name]
        frame[name] = pd.Series(
            [np.nan if value is None else value],
            dtype="object",
        )
    return frame


def predict(registered: LoadedModel, features: dict[str, Any]) -> PredictionResponse:
    frame = build_feature_frame(features)
    missing_features = [name for name in FEATURE_ORDER if features[name] is None]
    explanations = None
    explanation_status = "Explanation currently unavailable for this model."

    if registered.model_id == "random_forest":
        probabilities = np.asarray(registered.model.predict_proba(frame))[0]
        classes = list(registered.model.named_steps["model"].classes_)
        positive_index = classes.index(1)
        score = float(probabilities[positive_index])
        explanations, explanation_status = local_random_forest_explanation(
            registered.model.named_steps["model"],
            registered.model.named_steps["preprocessor"],
            frame,
            positive_index,
            score,
        )
    else:
        transformed = registered.preprocessor.transform(frame)
        transformed = np.asarray(transformed, dtype=np.float32)
        if registered.model_id == "cnn":
            transformed = transformed.reshape(
                transformed.shape[0], transformed.shape[1], 1
            )
        output = registered.model.predict(transformed, verbose=0)
        score = float(np.asarray(output).reshape(-1)[0])

    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Model returned an invalid score: {score!r}")
    predicted_class = int(score >= THRESHOLD)
    return PredictionResponse(
        model_id=registered.model_id,
        predicted_class=predicted_class,
        predicted_label=(
            "Model classified as disease present"
            if predicted_class
            else "Model classified as disease absent"
        ),
        model_score=score,
        threshold=THRESHOLD,
        missing_features=missing_features,
        explanation_status=explanation_status,
        explanations=explanations,
        disclaimer=MODEL_LIMITATIONS,
    )
