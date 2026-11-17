"""Reusable Stage 2 interface. Model loading never trains or downloads data."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .artifacts import ArtifactPredictor
    from .config import CONDITIONS, FEATURES, LABELS, MODELS
    from .explain import contribution_descriptions, exact_grouped_shap
    from .preprocess import clean_features
except ImportError:
    from artifacts import ArtifactPredictor
    from config import CONDITIONS, FEATURES, LABELS, MODELS
    from explain import contribution_descriptions, exact_grouped_shap
    from preprocess import clean_features


def patient_frame(patient):
    if not isinstance(patient, dict):
        raise ValueError("Patient input must be a dictionary / JSON object.")
    unknown = set(patient)-set(FEATURES)
    if unknown:
        raise ValueError(f"Unexpected input fields: {sorted(unknown)}. Expected: {FEATURES}")
    if patient.get("age") is None:
        raise ValueError("age is required to confirm adult eligibility (20+). Other missing fields are imputed.")
    values = {name: patient.get(name, np.nan) for name in FEATURES}
    for name in FEATURES:
        if values[name] is None:
            values[name] = np.nan
        if isinstance(values[name], (bool, list, dict, tuple)):
            raise ValueError(f"{name} must be a single clinical value, not a boolean or container.")
    aliases = {
        "sex": {"male": "Male", "female": "Female", "unknown": "Unknown"},
        "diabetes_history": {"yes": "Yes", "no": "No", "borderline": "Borderline", "unknown": "Unknown"},
        "smoking_history": {"yes": "Yes", "no": "No", "unknown": "Unknown", "former smoker": "Yes",
                            "current smoker": "Yes", "never smoker": "No"},
    }
    for name, options in aliases.items():
        if isinstance(values[name], str):
            key = values[name].strip().lower()
            if key not in options:
                raise ValueError(f"Unsupported {name}: {values[name]!r}. Valid values: {list(options)}")
            values[name] = options[key]
    if pd.isna(pd.to_numeric(values["age"], errors="coerce")):
        raise ValueError("age must be a finite number for an adult aged 20+.")
    return clean_features(pd.DataFrame([values], columns=FEATURES), strict=True)


class Stage2Predictor:
    def __init__(self, model_dir=MODELS, model_name=None):
        self.predictor = ArtifactPredictor(model_dir, model_name)

    def predict(self, patient, explain=True):
        frame = patient_frame(patient)
        scores = self.predictor.scores(frame)[0]
        explanation = exact_grouped_shap(self.predictor, frame) if explain else None
        missing = [name for name in FEATURES if pd.isna(frame.iloc[0][name]) or frame.iloc[0][name] == "Unknown"]
        warnings = []
        if missing:
            warnings.append("Missing values handled by the saved training preprocessing: " + ", ".join(missing))
        if float(patient["age"]) >= 80:
            warnings.append("Age encoded as 80 to match the NHANES 80+ top code.")
        if isinstance(patient.get("smoking_history"), str) and "smoker" in patient["smoking_history"].lower():
            warnings.append("Smoking is encoded as lifetime 100+ cigarettes, not current smoking status. Former/current smoker implies this lifetime threshold; confirm it in the frontend.")
        conditions = []
        for j, label in enumerate(LABELS):
            threshold = float(self.predictor.thresholds[j])
            entry = {"condition": label, "condition_name": CONDITIONS[j], "model_score": float(scores[j]),
                "score_type": "uncalibrated_model_score", "selected_threshold": threshold,
                "binary_classification": int(scores[j] >= threshold),
                "classification_meaning": "Above/below a validation-selected research threshold for reported diagnosis history; not a diagnosis.",
                "feature_contributions": contribution_descriptions(explanation, j) if explanation else []}
            if explanation:
                entry["explanation_base_score"] = float(explanation["base_scores"][j])
                entry["explanation_reconstruction_error"] = float(explanation["reconstruction_error"][j])
            conditions.append(entry)
        return {"model_version": self.predictor.metadata["version"], "run_id": self.predictor.metadata["run_id"],
            "model": self.predictor.name, "conditions": conditions, "warnings": warnings,
            "limitations": self.predictor.metadata["limitations"],
            "explanation_method": explanation["method"] if explanation else None,
            "explanation_limitations": explanation["limitations"] if explanation else None}


def predict_patient(patient, model_dir=MODELS, explain=True, model_name=None):
    """One-shot convenience API. Reuse Stage2Predictor for a long-lived frontend."""
    return Stage2Predictor(model_dir, model_name).predict(patient, explain=explain)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", help="Clinical values as a JSON object")
    group.add_argument("--input", type=Path, help="Path to a patient JSON file")
    parser.add_argument("--model-dir", type=Path, default=MODELS)
    parser.add_argument("--model", choices=["baseline", "neural"])
    parser.add_argument("--no-explain", action="store_true")
    args = parser.parse_args()
    try:
        patient = json.loads(args.input.read_text() if args.input else args.json)
        result = predict_patient(patient, args.model_dir, not args.no_explain, args.model)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, f"Stage 2 prediction error: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
