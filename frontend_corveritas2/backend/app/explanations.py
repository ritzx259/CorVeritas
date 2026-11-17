"""Exact path-based decomposition of a fitted forest's positive-class score.

This is the tree-interpreter method, not SHAP. Each split contributes the change
in node positive-class fraction along the actual input's path. The ensemble's
mean root fraction plus mean changes equals its predict_proba output.
"""
import numpy as np

from .schemas import FEATURE_ORDER, LABELS


def original_feature_indices(preprocessor) -> list[str]:
    names = preprocessor.get_feature_names_out()
    mapping = []
    for name in names:
        bare = str(name).split("__", 1)[-1]
        matches = [feature for feature in FEATURE_ORDER if bare == feature or bare.startswith(feature + "_")]
        if len(matches) != 1:
            raise ValueError("Cannot safely map transformed features to the clinical schema.")
        mapping.append(matches[0])
    return mapping


def explain_forest(pipeline, frame, positive_index: int, score: float) -> dict:
    preprocessor = pipeline.named_steps["preprocessor"]
    forest = pipeline.named_steps["model"]
    transformed = preprocessor.transform(frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    # sklearn trees compare float32 inputs, including when given float64 arrays.
    row = np.asarray(transformed, dtype=np.float32)[0]
    mapping = original_feature_indices(preprocessor)
    if len(mapping) != len(row):
        raise ValueError("Explanation feature count does not match transformed input.")
    contributions = np.zeros(len(row), dtype=np.float64)
    baseline = 0.0
    for estimator in forest.estimators_:
        tree = estimator.tree_
        values = tree.value[:, 0, :].astype(np.float64)
        totals = values.sum(axis=1)
        if np.any(totals <= 0):
            raise ValueError("Tree contains an invalid class distribution.")
        probabilities = values[:, positive_index] / totals
        baseline += probabilities[0]
        node = 0
        while tree.children_left[node] != tree.children_right[node]:
            feature = int(tree.feature[node])
            child = int(tree.children_left[node] if row[feature] <= tree.threshold[node] else tree.children_right[node])
            contributions[feature] += probabilities[child] - probabilities[node]
            node = child
    count = len(forest.estimators_)
    baseline /= count
    contributions /= count
    error = abs(baseline + float(contributions.sum()) - score)
    if not np.isfinite(error) or error > 1e-9:
        raise ValueError("Explanation failed its prediction reconstruction check.")
    grouped = {feature: 0.0 for feature in FEATURE_ORDER}
    importance = {feature: 0.0 for feature in FEATURE_ORDER}
    for index, feature in enumerate(mapping):
        grouped[feature] += float(contributions[index])
        importance[feature] += float(forest.feature_importances_[index])
    def entries(values):
        return [{"feature": feature, "label": LABELS[feature], "value": value} for feature, value in sorted(values.items(), key=lambda item: abs(item[1]), reverse=True)]
    return {"method": "Exact tree-path probability decomposition", "baseline": float(baseline), "score": score, "reconstruction_error": float(error), "contributions": entries(grouped), "global_importance": entries(importance), "note": "Local contributions are changes in the positive-class model score along this patient's actual tree paths, averaged across trees. One-hot columns are summed into original features. The baseline is the mean root-node positive-class fraction, not population disease prevalence. Contributions depend on tree split order and are associations, not causal effects or medical explanations. Global importance is impurity-based training importance, which can be biased; it is distinct from this individual explanation."}
