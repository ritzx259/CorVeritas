"""Train and select independent logistic classifiers and a multi-label PyTorch MLP."""
import argparse
import copy
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.multioutput import MultiOutputClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .artifacts import baseline_scores, file_hash, write_json
    from .config import FEATURES, LABELS, LIMITATIONS, MODELS, PROCESSED, RESULTS, SEED, VERSION, ensure_dirs
    from .evaluate import evaluate_saved, metrics_table, save_figure, select_thresholds
    from .model import MultiLabelMLP, neural_scores, seed_everything, select_device
    from .preprocess import clean_features, make_preprocessor, prepare, split_participants
except ImportError:
    from artifacts import baseline_scores, file_hash, write_json
    from config import FEATURES, LABELS, LIMITATIONS, MODELS, PROCESSED, RESULTS, SEED, VERSION, ensure_dirs
    from evaluate import evaluate_saved, metrics_table, save_figure, select_thresholds
    from model import MultiLabelMLP, neural_scores, seed_everything, select_device
    from preprocess import clean_features, make_preprocessor, prepare, split_participants


def macro_ap(y, scores):
    return float(average_precision_score(y, scores, average="macro"))


def fit_neural(x_train, y_train, x_valid, y_valid, device, lr=.001, weighted=True,
               batch_size=32, epochs=100, patience=15, seed=SEED):
    seed_everything(seed)
    model = MultiLabelMLP(x_train.shape[1]).to(device)
    positive = y_train.sum(axis=0)
    if np.any(positive == 0) or np.any(positive == len(y_train)):
        raise ValueError("Training needs both classes for each label.")
    weights = (len(y_train)-positive)/positive if weighted else np.ones(3)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(weights, dtype=torch.float32, device=device))
    validation_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=5)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.as_tensor(x_train), torch.as_tensor(y_train, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    xv = torch.as_tensor(x_valid, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_valid, dtype=torch.float32, device=device)
    best_ap, stale, best_epoch, best_weights = -1., 0, 0, None
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        total_loss, total_unweighted = 0., 0.
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_function(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; check input values.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            optimizer.step()
            total_loss += loss.item()*len(x)
            total_unweighted += validation_loss(logits.detach(), y).item()*len(x)
        model.eval()
        with torch.inference_mode():
            logits = model(xv)
            vl = validation_loss(logits, yv).item()
            scores = torch.sigmoid(logits).cpu().numpy()
        ap = macro_ap(y_valid, scores)
        scheduler.step(ap)
        history.append({"epoch": epoch, "training_optimization_loss": total_loss/len(x_train),
            "training_unweighted_bce": total_unweighted/len(x_train), "validation_unweighted_bce": vl,
            "validation_macro_ap": ap, "learning_rate": optimizer.param_groups[0]["lr"]})
        if ap > best_ap + 1e-5:
            best_ap, best_epoch, stale = ap, epoch, 0
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"  epoch {epoch:3d}: train BCE={total_unweighted/len(x_train):.4f} val BCE={vl:.4f} val macro AP={ap:.4f}", flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best_weights)
    model.to("cpu")
    return model, history, {"learning_rate": lr, "positive_weighting": weighted, "positive_weights": weights.tolist(),
        "batch_size": batch_size, "maximum_epochs": epochs, "patience": patience,
        "best_epoch": best_epoch, "validation_macro_ap": best_ap, "optimizer": "AdamW", "weight_decay": .01,
        "gradient_clip_norm": 1., "scheduler": "ReduceLROnPlateau(validation macro AP)",
        "early_stopping": "validation macro AP, min_delta=1e-5", "seed": seed, "device": str(device)}


def split_report(frame, partitions, method):
    result = {"method": method, "seed": SEED, "partitions": {}}
    for name, idx in partitions.items():
        part = frame.iloc[idx]
        result["partitions"][name] = {"count": len(part), "fraction": len(part)/len(frame),
            "positive": part[LABELS].sum().to_dict(), "negative": (1-part[LABELS]).sum().to_dict(),
            "prevalence_unweighted": part[LABELS].mean().to_dict(),
            "multiple_conditions": int((part[LABELS].sum(axis=1)>1).sum()),
            "combinations": {"".join(map(str, key)): int(value) for key, value in part.groupby(LABELS).size().items()}}
    return result


def train(args):
    ensure_dirs()
    model_dir = Path(args.model_dir).resolve()
    # Preserve previous experiments and their once-only test evaluations.
    if (model_dir / "metadata.json").exists():
        raise FileExistsError(f"Artifacts already exist in {model_dir}. Use a fresh --model-dir for a new experiment.")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience and batch-size must be positive.")
    if args.threads < 1 or any(not np.isfinite(lr) or lr <= 0 for lr in args.learning_rates):
        raise ValueError("threads and learning rates must be positive and finite.")
    cohort_path = PROCESSED / "cohort.csv"
    if not cohort_path.exists():
        prepare()
    frame = pd.read_csv(cohort_path)
    if frame.empty or frame.SEQN.duplicated().any() or frame[LABELS].isna().any().any():
        raise ValueError("Prepared cohort is empty, has duplicate participants or missing labels; rerun preprocess.py.")
    if not np.isin(frame[LABELS].to_numpy(), [0, 1]).all():
        raise ValueError("Prepared labels must be binary.")
    frame[FEATURES] = clean_features(frame[FEATURES], strict=True)
    partitions, method = split_participants(frame)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    result_dir = RESULTS / "runs" / run_id
    plot_dir = RESULTS / "plots" / run_id
    result_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    report = split_report(frame, partitions, method)
    write_json(result_dir / "split_distribution.json", report)
    print(json.dumps(report, indent=2), flush=True)
    split_frame = pd.concat([pd.DataFrame({"SEQN": frame.iloc[idx].SEQN, "partition": name}) for name, idx in partitions.items()])
    split_frame.to_csv(model_dir / "splits.csv", index=False)
    train_frame, validation = [frame.iloc[partitions[name]] for name in ("train", "validation")]
    transformer = make_preprocessor()
    x_train = np.asarray(transformer.fit_transform(train_frame[FEATURES]), dtype=np.float32)
    x_valid = np.asarray(transformer.transform(validation[FEATURES]), dtype=np.float32)
    if not np.isfinite(x_train).all() or not np.isfinite(x_valid).all():
        raise ValueError("Preprocessing produced nonfinite values.")
    y_train, y_valid = [part[LABELS].to_numpy(dtype=int) for part in (train_frame, validation)]
    seed_everything()
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    print(f"Device: {device}. Model selection uses validation macro average precision.", flush=True)
    candidates, best_baseline, best_baseline_ap, baseline_params = [], None, -1., None
    for c in (.1, 1., 10.):
        for class_weight in (None, "balanced"):
            candidate = MultiOutputClassifier(LogisticRegression(C=c, class_weight=class_weight, max_iter=2000, random_state=SEED))
            candidate.fit(x_train, y_train)
            ap = macro_ap(y_valid, baseline_scores(candidate, x_valid))
            info = {"model": "baseline", "C": c, "class_weight": class_weight, "validation_macro_ap": ap}
            candidates.append(info)
            if ap > best_baseline_ap:
                best_baseline, best_baseline_ap, baseline_params = candidate, ap, info
    best_neural, best_neural_ap, neural_params = None, -1., None
    for lr in args.learning_rates:
        for weighted in (False, True):
            print(f"Neural candidate: learning_rate={lr}, positive_weighting={weighted}", flush=True)
            candidate, history, info = fit_neural(x_train, y_train, x_valid, y_valid, device,
                lr=lr, weighted=weighted, batch_size=args.batch_size, epochs=args.epochs, patience=args.patience)
            tag = f"neural_lr{lr}_weighted{weighted}"
            pd.DataFrame(history).to_csv(result_dir / f"{tag}_history.csv", index=False)
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].plot([h["epoch"] for h in history], [h["training_unweighted_bce"] for h in history], label="Train (dropout active)")
            axes[0].plot([h["epoch"] for h in history], [h["validation_unweighted_bce"] for h in history], label="Validation")
            axes[0].set(xlabel="Epoch", ylabel="Unweighted BCE"); axes[0].legend()
            axes[1].plot([h["epoch"] for h in history], [h["validation_macro_ap"] for h in history])
            axes[1].set(xlabel="Epoch", ylabel="Validation macro AP")
            save_figure(fig, plot_dir / f"{tag}_training.png")
            candidates.append({"model": "neural", **info})
            if info["validation_macro_ap"] > best_neural_ap:
                best_neural, best_neural_ap, neural_params = copy.deepcopy(candidate), info["validation_macro_ap"], info
    models = {"baseline": best_baseline, "neural": best_neural}
    selected = "neural" if best_neural_ap > best_baseline_ap else "baseline"
    model_metadata = {}
    for name, model in models.items():
        scores = baseline_scores(model, x_valid) if name == "baseline" else neural_scores(model, x_valid)
        thresholds = select_thresholds(y_valid, scores)
        model_metadata[name] = {"hyperparameters": baseline_params if name == "baseline" else neural_params,
            "thresholds": thresholds, "validation_metrics": metrics_table(y_valid, scores, thresholds)}
        write_json(result_dir / f"{name}_validation_metrics.json", model_metadata[name])
        np.savez_compressed(result_dir / f"{name}_validation_predictions.npz", seqn=validation.SEQN.to_numpy(), labels=y_valid, scores=scores)
    joblib.dump(transformer, model_dir / "preprocessor.joblib")
    joblib.dump(best_baseline, model_dir / "baseline.joblib")
    torch.save(best_neural.state_dict(), model_dir / "neural.pt")
    background = train_frame[FEATURES].sample(n=min(16, len(train_frame)), random_state=SEED).reset_index(drop=True)
    joblib.dump(background, model_dir / "explanation_background.joblib")
    metadata = {"version": VERSION, "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES, "transformed_features": list(transformer.get_feature_names_out()), "labels": LABELS,
        "architecture": {"input_dim": x_train.shape[1], "hidden": [128, 64], "dropout": [.3, .2]},
        "selected_model": selected, "selection_rule": "Highest validation macro average precision; ties favor baseline",
        "threshold_rule": "Per-label maximum validation F1; ties favor higher threshold", "models": model_metadata,
        "score_interpretation": "uncalibrated model scores for reported diagnosis history", "calibration": "not fitted",
        "cohort_path": str(cohort_path.resolve()), "cohort_sha256": file_hash(cohort_path),
        "splits_sha256": file_hash(model_dir / "splits.csv"), "result_dir": str(result_dir.resolve()),
        "plot_dir": str(plot_dir.resolve()),
        "limitations": LIMITATIONS, "python": sys.version, "platform": platform.platform(),
        "packages": {pkg: importlib.metadata.version(pkg) for pkg in ("numpy", "pandas", "scikit-learn", "torch", "joblib", "matplotlib")},
        "artifact_sha256": {name: file_hash(model_dir / name) for name in
            ("preprocessor.joblib", "baseline.joblib", "neural.pt", "explanation_background.joblib")}}
    write_json(model_dir / "metadata.json", metadata)
    write_json(result_dir / "candidate_comparison.json", candidates)
    pd.DataFrame(candidates).to_csv(result_dir / "candidate_comparison.csv", index=False)
    print(f"Selected {selected} using validation only. Artifacts: {model_dir}", flush=True)
    print(f"Thresholds: {json.dumps({n: m['thresholds'] for n, m in model_metadata.items()})}", flush=True)
    if not args.skip_test:
        evaluate_saved(model_dir)
    if not args.skip_explain:
        try:
            from .explain import generate_explanations
        except ImportError:
            from explain import generate_explanations
        generate_explanations(model_dir, samples=args.explain_samples)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODELS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[.001, .0003])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--skip-explain", action="store_true")
    parser.add_argument("--explain-samples", type=int, default=20)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
