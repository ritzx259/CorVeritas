"""Inference only: no training, threshold tuning, or test-set evaluation."""
import numpy as np


def positive_class_index(classes) -> int:
    indices = np.flatnonzero(np.asarray(classes) == 1)
    if len(indices) != 1:
        raise ValueError("Artifact must contain exactly one class labelled 1.")
    return int(indices[0])


def transformed_neural_input(preprocessor, model, frame, cnn=False):
    matrix = preprocessor.transform(frame)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    dtype = getattr(model, "input_dtype", "float32")
    dtype = getattr(dtype, "name", dtype)
    matrix = np.asarray(matrix, dtype=str(dtype))
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Preprocessing did not produce a finite two-dimensional matrix.")
    if cnn:
        matrix = matrix.reshape(matrix.shape[0], matrix.shape[1], 1)
    expected_shape = tuple(model.input_shape)
    if tuple(matrix.shape[1:]) != expected_shape[1:]:
        raise ValueError("Saved preprocessor output does not match the neural model input shape.")
    return matrix


def predict_score(loaded, frame) -> float:
    if loaded.id == "random_forest":
        score = loaded.model.predict_proba(frame)[0, loaded.positive_index]
    else:
        matrix = transformed_neural_input(loaded.preprocessor, loaded.model, frame, cnn=loaded.id == "cnn")
        # Direct eager inference avoids creating a tf.data thread pool per request.
        output = loaded.model(matrix, training=False)
        if hasattr(output, "numpy"):
            output = output.numpy()
        if np.asarray(output).size != 1:
            raise ValueError("Expected one binary sigmoid output.")
        score = np.asarray(output).reshape(-1)[0]
    score = float(score)
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Model returned an invalid binary score.")
    return score
