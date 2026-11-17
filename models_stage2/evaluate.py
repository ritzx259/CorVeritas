"""Descriptive EDA and cached, once-per-run untouched test-set evaluation."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    log_loss, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve, auc)

try:
    from .artifacts import ArtifactPredictor, file_hash, write_json
    from .config import FEATURES, LABELS, MODELS, NUMERIC, PROCESSED, RESULTS
except ImportError:
    from artifacts import ArtifactPredictor, file_hash, write_json
    from config import FEATURES, LABELS, MODELS, NUMERIC, PROCESSED, RESULTS


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def exploratory_analysis(frame, audit, result_dir=RESULTS):
    result_dir = Path(result_dir)
    plots = result_dir / "plots"
    result_dir.mkdir(parents=True, exist_ok=True)
    frame.describe(include="all").to_csv(result_dir / "summary_statistics.csv")
    frame[FEATURES].isna().sum().rename("missing").to_csv(result_dir / "missing_features.csv")
    distribution = pd.DataFrame({"positive": frame[LABELS].sum(), "negative": (1-frame[LABELS]).sum()})
    distribution.to_csv(result_dir / "label_counts.csv")
    combinations = frame.groupby(LABELS).size().rename("count").reset_index()
    combinations.to_csv(result_dir / "label_combinations.csv", index=False)
    report = {"shape": list(frame.shape), "columns": {c: str(frame[c].dtype) for c in frame},
        "participants_with_multiple_conditions": int((frame[LABELS].sum(axis=1) > 1).sum()),
        "participants_with_any_condition": int((frame[LABELS].sum(axis=1) > 0).sum()), "audit": audit,
        "interpretation": "Unweighted descriptive cohort counts, not population prevalence estimates."}
    write_json(result_dir / "eda.json", report)
    fig, ax = plt.subplots(figsize=(9, 4))
    distribution.plot.bar(ax=ax, rot=12)
    ax.set(ylabel="Participants", title="Reported diagnosis history: class imbalance (unweighted)")
    save_figure(fig, plots / "class_imbalance.png")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for name, ax in zip(FEATURES, axes.flat):
        if name in NUMERIC:
            ax.hist(frame[name].dropna(), bins=30, color="steelblue")
        else:
            frame[name].fillna("Unknown").value_counts().plot.bar(ax=ax, rot=15)
        ax.set(title=name, ylabel="Participants")
    save_figure(fig, plots / "feature_distributions.png")
    correlation = frame[NUMERIC].corr()
    correlation.to_csv(result_dir / "numerical_correlations.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    img = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(NUMERIC)), NUMERIC, rotation=45, ha="right")
    ax.set_yticks(range(len(NUMERIC)), NUMERIC)
    for a in range(len(NUMERIC)):
        for b in range(len(NUMERIC)):
            ax.text(b, a, f"{correlation.iloc[a,b]:.2f}", ha="center", va="center")
    fig.colorbar(img, ax=ax)
    ax.set_title("Numerical inputs: pairwise Pearson correlation")
    save_figure(fig, plots / "input_correlations.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    frame[LABELS].sum(axis=1).value_counts().sort_index().plot.bar(ax=ax, rot=0)
    ax.set(xlabel="Number of positive reported conditions", ylabel="Participants", title="Label co-occurrence")
    save_figure(fig, plots / "condition_cooccurrence.png")


def select_thresholds(y, scores):
    """Maximize validation F1; deterministic ties favor the higher threshold."""
    thresholds = []
    for j in range(3):
        if len(np.unique(y[:, j])) != 2:
            raise ValueError(f"Validation {LABELS[j]} needs both classes for threshold selection.")
        precision, recall, values = precision_recall_curve(y[:, j], scores[:, j])
        f1 = np.divide(2*precision[:-1]*recall[:-1], precision[:-1]+recall[:-1],
                       out=np.zeros_like(values), where=(precision[:-1]+recall[:-1]) > 0)
        best = np.flatnonzero(np.isclose(f1, np.max(f1), rtol=0, atol=1e-12))[-1]
        thresholds.append(float(values[best]))
    return thresholds


def expected_calibration_error(y, scores, bins=10):
    edges = np.linspace(0, 1, bins+1)
    indexes = np.clip(np.digitize(scores, edges, right=True)-1, 0, bins-1)
    return float(sum(np.mean(indexes == k) * abs(np.mean(y[indexes == k])-np.mean(scores[indexes == k]))
                     for k in range(bins) if (indexes == k).any()))


def condition_metrics(y, scores, threshold):
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    precision, recall, _ = precision_recall_curve(y, scores) if both else (None, None, None)
    return {"threshold": float(threshold), "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, predicted, zero_division=0)),
        "specificity": float(tn/(tn+fp)) if tn+fp else None,
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)) if both else None,
        "average_precision": float(average_precision_score(y, scores)) if both else None,
        "pr_auc_trapezoidal": float(auc(recall, precision)) if both else None,
        "brier_score": float(brier_score_loss(y, scores)),
        "log_loss": float(log_loss(y, scores, labels=[0, 1])),
        "ece_10_equal_width_bins": expected_calibration_error(y, scores),
        "positive_support": int(tp+fn), "negative_support": int(tn+fp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "zero_division_policy": "Undefined precision/recall/F1 set to zero; unavailable AUCs are null."}


def metrics_table(y, scores, thresholds):
    return {label: condition_metrics(y[:, j], scores[:, j], thresholds[j]) for j, label in enumerate(LABELS)}


def plot_predictions(y, scores, thresholds, name, directory):
    directory = Path(directory)
    for kind in ("roc", "precision_recall", "confusion", "calibration"):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for j, ax in enumerate(axes):
            truth, score = y[:, j], scores[:, j]
            if kind == "roc":
                x, z, _ = roc_curve(truth, score)
                ax.plot(x, z, label=f"AUC {roc_auc_score(truth, score):.3f}")
                ax.plot([0, 1], [0, 1], "--", color="gray")
                ax.set(xlabel="False positive rate", ylabel="Sensitivity")
            elif kind == "precision_recall":
                p, r, _ = precision_recall_curve(truth, score)
                ax.plot(r, p, label=f"AP {average_precision_score(truth, score):.3f}")
                ax.axhline(truth.mean(), linestyle="--", color="gray", label="Sample prevalence")
                ax.set(xlabel="Recall", ylabel="Precision")
            elif kind == "calibration":
                observed, predicted = calibration_curve(truth, score, n_bins=8, strategy="quantile")
                ax.plot(predicted, observed, "o-", label=f"Brier {brier_score_loss(truth, score):.3f}")
                ax.plot([0, 1], [0, 1], "--", color="gray")
                ax.set(xlabel="Mean uncalibrated score", ylabel="Observed positive fraction")
            else:
                matrix = confusion_matrix(truth, score >= thresholds[j], labels=[0, 1])
                ax.imshow(matrix, cmap="Blues")
                for a in range(2):
                    for b in range(2):
                        ax.text(b, a, matrix[a,b], ha="center", va="center")
                ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Reported label")
            ax.set_title(LABELS[j].replace("_", " "))
            if kind != "confusion":
                ax.legend(fontsize=8)
        fig.suptitle(f"{name}: {kind.replace('_', ' ')} (unweighted test set)")
        save_figure(fig, directory / f"{name}_{kind}.png")


def subgroup_metrics(frame, scores, thresholds, min_total=30, min_class=5):
    groups = {"sex=Female": frame.sex == "Female", "sex=Male": frame.sex == "Male",
              "age=20-39": frame.age < 40, "age=40-59": frame.age.between(40, 59.999), "age=60+": frame.age >= 60}
    result = {}
    for group, mask in groups.items():
        subset = frame.loc[mask, LABELS].to_numpy(dtype=int)
        per_label = {}
        for j, label in enumerate(LABELS):
            positive, negative = int(subset[:, j].sum()), int(len(subset)-subset[:, j].sum())
            if len(subset) < min_total or min(positive, negative) < min_class:
                per_label[label] = {"status": "insufficient support", "positive_support": positive, "negative_support": negative,
                                    "minimum_required": {"total": min_total, "per_class": min_class}}
            else:
                per_label[label] = condition_metrics(subset[:, j], scores[np.asarray(mask), j], thresholds[j])
        result[group] = per_label
    return result


def write_research_report(result, metadata, result_dir):
    lines = ["# CorVeritas Stage 2: executed evaluation", "",
        f"Run: `{metadata['run_id']}`. Recommended family from validation: **{metadata['selected_model']}**.", "",
        "All values below are unweighted test-sample metrics for self-reported previous diagnosis history.",
        "The test set was used after model and threshold selection. Scores are uncalibrated.", "",
        "| Model | Condition | Positive / negative | Threshold | Precision | Sensitivity | Specificity | F1 | ROC-AUC | AP | PR-AUC | Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model, details in result["models"].items():
        for label, values in details["conditions"].items():
            metrics = [values[k] for k in ("threshold", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc", "average_precision", "pr_auc_trapezoidal", "brier_score")]
            cells = " | ".join(f"{v:.4f}" if v is not None else "NA" for v in metrics)
            lines.append(f"| {model} | {label} | {values['positive_support']} / {values['negative_support']} | {cells} |")
    lines += ["", "## Interpretation", "",
        "Average precision (AP) is shown separately from trapezoidal precision-recall AUC. The sample positive fraction is the no-skill AP reference. High ROC-AUC alone can coexist with low precision under strong class imbalance.", "",
        "The saved recommended family is determined by validation macro AP. Test results do not change that selection. Differences between models should not be treated as significant without suitable uncertainty analysis and external replication.", "",
        "Subgroup tables, confusion matrices, ECE and log loss are in test_evaluation.json. Age/sex cells require at least 30 participants and 5 of each class; smaller cells are explicitly withheld. Sparse positive cases make calibration and subgroup findings unstable.", "",
        "Plots: `" + metadata.get("plot_dir", str(Path(result_dir)/"plots")) + "`.", "",
        "## Limitations", ""]
    lines += ["- " + limitation for limitation in metadata["limitations"]]
    (Path(result_dir) / "report.md").write_text("\n".join(lines)+"\n")


def evaluate_saved(model_dir=MODELS):
    directory = Path(model_dir)
    metadata = json.loads((directory / "metadata.json").read_text())
    result_dir = Path(metadata["result_dir"])
    marker = result_dir / "test_evaluation.json"
    if marker.exists():
        cached = json.loads(marker.read_text())
        if cached["run_id"] != metadata["run_id"]:
            raise ValueError("Evaluation directory belongs to a different training run.")
        write_research_report(cached, metadata, result_dir)
        print(f"Using cached test evaluation: {marker}. Test set is not rescored.")
        return cached
    cohort_path = Path(metadata["cohort_path"])
    if file_hash(cohort_path) != metadata["cohort_sha256"]:
        raise ValueError("Prepared cohort changed after training; refuse to evaluate against a different dataset.")
    frame = pd.read_csv(cohort_path)
    split_path = directory / "splits.csv"
    if file_hash(split_path) != metadata["splits_sha256"]:
        raise ValueError("Participant partitions changed after training.")
    splits = pd.read_csv(split_path)
    test_ids = splits.loc[splits.partition == "test", "SEQN"]
    test = frame.set_index("SEQN").loc[test_ids].reset_index()
    y = test[LABELS].to_numpy(dtype=int)
    result = {"run_id": metadata["run_id"], "selected_model_from_validation": metadata["selected_model"],
              "selection_metric": "validation macro average precision", "models": {}}
    rows = []
    for name in ("baseline", "neural"):
        predictor = ArtifactPredictor(directory, name)
        scores = predictor.scores(test[FEATURES])
        metrics = metrics_table(y, scores, predictor.thresholds)
        result["models"][name] = {"conditions": metrics, "subgroups": subgroup_metrics(test, scores, predictor.thresholds)}
        output = test[["SEQN", *LABELS]].copy()
        for j, label in enumerate(LABELS):
            output[label + "_score"] = scores[:, j]
        output.to_csv(result_dir / f"{name}_test_predictions.csv", index=False)
        for label, values in metrics.items():
            rows.append({"model": name, "condition": label, **{k: v for k, v in values.items() if k not in ("confusion_matrix", "zero_division_policy")}})
        plot_predictions(y, scores, predictor.thresholds, name,
                         Path(metadata.get("plot_dir", str(result_dir / "plots"))))
    pd.DataFrame(rows).to_csv(result_dir / "model_comparison.csv", index=False)
    write_json(marker, result)
    write_research_report(result, metadata, result_dir)
    print(pd.DataFrame(rows)[["model", "condition", "average_precision", "roc_auc", "f1", "brier_score"]].to_string(index=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODELS)
    args = parser.parse_args()
    evaluate_saved(args.model_dir)


if __name__ == "__main__":
    main()
