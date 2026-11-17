"""Exact grouped interventional SHAP for eight clinical inputs; optional SHAP library plots."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

try:
    from .artifacts import ArtifactPredictor, baseline_scores, file_hash, write_json
    from .config import FEATURES, LABELS, MODELS, SEED
    from .evaluate import save_figure
    from .model import neural_scores
except ImportError:
    from artifacts import ArtifactPredictor, baseline_scores, file_hash, write_json
    from config import FEATURES, LABELS, MODELS, SEED
    from evaluate import save_figure
    from model import neural_scores


def exact_grouped_shap(predictor, sample):
    """Enumerate all 2**8 coalitions, averaging over a saved training-only background.

    One-hot columns stay grouped as their original clinical feature. Exact for this
    finite-background interventional game, not a causal/conditional explanation.
    """
    if len(sample) != 1:
        raise ValueError("Individual explanation expects exactly one patient.")
    background = predictor.background[FEATURES]
    n_features, n_background = len(FEATURES), len(background)
    hybrids = []
    for coalition in range(1 << n_features):
        hybrid = background.copy()
        for j, name in enumerate(FEATURES):
            if coalition & (1 << j):
                hybrid[name] = sample.iloc[0][name]
        hybrids.append(hybrid)
    # Do not revalidate artificial coalition combinations as if they were real patients.
    values = predictor.scores(pd.concat(hybrids, ignore_index=True), validate=False).reshape(1 << n_features, n_background, 3).mean(axis=1)
    contributions = np.zeros((n_features, 3))
    for j in range(n_features):
        for coalition in range(1 << n_features):
            if coalition & (1 << j):
                continue
            size = coalition.bit_count()
            weight = 1. / (n_features * math.comb(n_features-1, size))
            contributions[j] += weight * (values[coalition | (1 << j)]-values[coalition])
    return {"base_scores": values[0], "contributions": contributions,
            "scores": values[-1], "reconstruction_error": np.abs(values[0]+contributions.sum(axis=0)-values[-1]),
            "method": "Exact grouped interventional Shapley values on model-score scale",
            "background_size": n_background,
            "limitations": "Training-background associations, not causal effects. Mixed feature coalitions may be physiologically implausible or outside the data distribution."}


def contribution_descriptions(explanation, label_index):
    values = explanation["contributions"][:, label_index]
    return [{"feature": FEATURES[j], "contribution": float(values[j]),
        "description": f"{FEATURES[j]} {'increases' if values[j] > 0 else 'decreases' if values[j] < 0 else 'does not change'} the model score by {abs(values[j]):.4f} relative to the sampled training background; association, not causation."}
        for j in np.argsort(-np.abs(values))]


def raw_permutation_importance(predictor, frame, repeats=5):
    """Held-out validation AP decrease, with raw categorical groups permuted together."""
    truth = frame[LABELS].to_numpy(dtype=int)
    features = frame[FEATURES]
    reference = average_precision_score(truth, predictor.scores(features), average=None)
    rng = np.random.default_rng(SEED)
    rows = []
    for name in FEATURES:
        decreases = []
        for _ in range(repeats):
            shuffled = features.copy()
            shuffled[name] = shuffled[name].to_numpy()[rng.permutation(len(shuffled))]
            scores = predictor.scores(shuffled, validate=False)
            decreases.append(reference-average_precision_score(truth, scores, average=None))
        decreases = np.array(decreases)
        for j, label in enumerate(LABELS):
            rows.append({"feature": name, "condition": label, "mean_validation_ap_decrease": float(decreases[:, j].mean()),
                         "std_validation_ap_decrease": float(decreases[:, j].std())})
    return pd.DataFrame(rows)


def shap_library_summary(predictor, frame, directory, samples, plot_dir=None):
    """Optional sampled permutation SHAP in the transformed feature space."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("Optional SHAP library unavailable. Install shap>=0.46,<0.50; exact grouped explanations need no SHAP dependency.") from exc
    import matplotlib.pyplot as plt
    plot_dir = Path(plot_dir) if plot_dir else directory / "plots"
    background = np.asarray(predictor.preprocessor.transform(predictor.background), dtype=np.float32)
    matrix = np.asarray(predictor.preprocessor.transform(frame[FEATURES].iloc[:samples]), dtype=np.float32)
    def scores(x):
        return baseline_scores(predictor.model, x) if predictor.name == "baseline" else neural_scores(predictor.model, x)
    explainer = shap.Explainer(scores, shap.maskers.Independent(background), algorithm="permutation",
                               feature_names=predictor.metadata["transformed_features"], seed=SEED)
    explanation = explainer(matrix, max_evals=2*matrix.shape[1]+1)
    np.savez_compressed(directory / "shap_library_values.npz", values=explanation.values,
                        base_values=explanation.base_values, data=matrix)
    for j, label in enumerate(LABELS):
        shap.plots.beeswarm(explanation[:, :, j], show=False, max_display=15)
        save_figure(plt.gcf(), plot_dir / f"shap_library_{label}.png")


def generate_explanations(model_dir=MODELS, samples=20, use_shap=False):
    if samples < 1:
        raise ValueError("Explanation sample count must be positive.")
    predictor = ArtifactPredictor(model_dir)
    metadata = predictor.metadata
    cohort_path = Path(metadata["cohort_path"])
    if file_hash(cohort_path) != metadata["cohort_sha256"]:
        raise ValueError("Cohort changed after training.")
    split_path = Path(model_dir) / "splits.csv"
    if file_hash(split_path) != metadata["splits_sha256"]:
        raise ValueError("Saved partitions changed after training.")
    cohort, splits = pd.read_csv(cohort_path), pd.read_csv(split_path)
    validation = cohort.set_index("SEQN").loc[splits.loc[splits.partition == "validation", "SEQN"]].reset_index()
    sample = validation.sample(n=min(samples, len(validation)), random_state=SEED).reset_index(drop=True)
    directory = Path(metadata["result_dir"]) / "explanations"
    plot_dir = Path(metadata.get("plot_dir", str(Path(metadata["result_dir"]) / "plots"))) / "explanations"
    directory.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    individual = []
    all_values = []
    for _, row in sample.iterrows():
        explanation = exact_grouped_shap(predictor, pd.DataFrame([row[FEATURES]]))
        all_values.append(explanation["contributions"])
        individual.append({"SEQN": int(row.SEQN), **explanation})
    values = np.stack(all_values)
    np.savez_compressed(directory / "grouped_shap_values.npz", seqn=sample.SEQN.to_numpy(),
                        values=values, feature_names=np.array(FEATURES), label_names=np.array(LABELS))
    write_json(directory / "individual_explanations.json", individual)
    importance = pd.DataFrame(np.mean(np.abs(values), axis=0), index=FEATURES, columns=LABELS)
    importance.to_csv(directory / "global_mean_absolute_shap.csv")
    importance.plot.barh(figsize=(11, 6), title="Mean absolute grouped SHAP (sampled validation participants)")
    save_figure(plt.gcf(), plot_dir / "global_importance.png")
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    rng = np.random.default_rng(SEED)
    for j, ax in enumerate(axes):
        for k, name in enumerate(FEATURES):
            ax.scatter(values[:, k, j], k+rng.normal(0, .06, len(values)), alpha=.7, s=15)
        ax.axvline(0, color="gray", linewidth=1)
        ax.set(yticks=range(len(FEATURES)), yticklabels=FEATURES, xlabel="Contribution to model score", title=LABELS[j])
    save_figure(fig, plot_dir / "grouped_shap_summary.png")
    first = individual[0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for j, ax in enumerate(axes):
        ax.barh(FEATURES, first["contributions"][:, j])
        ax.set(xlabel="Contribution to model score", title=LABELS[j])
    save_figure(fig, plot_dir / "individual_contributions.png")
    permutation = raw_permutation_importance(predictor, validation)
    permutation.to_csv(directory / "validation_permutation_importance.csv", index=False)
    permutation.pivot(index="feature", columns="condition", values="mean_validation_ap_decrease").plot.barh(figsize=(11, 6), title="Validation permutation importance (AP decrease)")
    save_figure(plt.gcf(), plot_dir / "permutation_importance.png")
    write_json(directory / "method.json", {"model": predictor.name, "samples": len(sample),
        "method": first["method"], "background_size": len(predictor.background), "limitations": first["limitations"],
        "human_readable_example": {label: contribution_descriptions(first, j) for j, label in enumerate(LABELS)},
        "global_warning": "Small random validation sample; importance estimates and permutation scores are unstable with rare outcomes. Correlated inputs redistribute importance."})
    if use_shap:
        shap_library_summary(predictor, sample, directory, len(sample), plot_dir)
    print(f"Saved explanations for {len(sample)} validation participants: {directory}", flush=True)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODELS)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--shap", action="store_true", help="Also generate optional SHAP library summaries")
    args = parser.parse_args()
    generate_explanations(args.model_dir, args.samples, args.shap)


if __name__ == "__main__":
    main()
