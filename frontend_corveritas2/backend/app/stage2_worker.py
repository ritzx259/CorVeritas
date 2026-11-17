"""Private JSON-lines worker, executed only by the isolated Stage 2 interpreter.

Patient values travel through anonymous pipes; they are not written to disk,
command arguments, logs, or a second HTTP service. Original Stage 2 code is read only.
"""
import contextlib
import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))
protocol_output = sys.stdout
os.environ.setdefault("MPLBACKEND", "Agg")


def emit(value):
    protocol_output.write(json.dumps(value, allow_nan=False) + "\n")
    protocol_output.flush()


def schema(config, predictor):
    labels = {
        "age": ("Age", "years", "Required: Stage 2 was trained on adults aged 20+. Ages 80+ use the saved 80-year top code."),
        "bmi": ("Body mass index", "kg/m²", "Use a documented BMI measurement, or leave unknown."),
        "systolic_bp": ("Systolic blood pressure", "mmHg", "Stage 2 was trained on the mean of valid resting systolic readings."),
        "diastolic_bp": ("Diastolic blood pressure", "mmHg", "Use a documented resting diastolic measurement, below systolic pressure when both are known."),
        "total_cholesterol": ("Total cholesterol", "mg/dL", "Use a documented total cholesterol result, or leave unknown."),
        "sex": ("Recorded sex", None, "Recorded sex categories follow the saved NHANES preprocessing."),
        "diabetes_history": ("Reported diabetes history", None, "A reported diabetes diagnosis, not the Stage 1 fasting-glucose threshold. Pregnancy-only diabetes is not Yes; borderline remains distinct."),
        "smoking_history": ("100+ lifetime cigarettes", None, "Yes means at least 100 cigarettes over the lifetime. This is not current smoking status."),
    }
    sources = {"age": "age", "sex": "sex", "systolic_bp": "trestbps", "total_cholesterol": "chol"}
    categories = dict(zip(config.CATEGORICAL, config.CATEGORIES))
    fields = []
    for name in config.FEATURES:
        label, unit, help_text = labels[name]
        field = {"name": name, "label": label, "unit": unit, "help": help_text,
                 "required": name == "age", "type": "number" if name in config.NUMERIC else "category",
                 "source_stage1": sources.get(name)}
        if name in config.RANGES:
            field["min"], field["max"] = config.RANGES[name]
        if name in categories:
            field["categories"] = categories[name]
        fields.append(field)
    metadata = predictor.predictor.metadata
    return {"available": True, "model": predictor.predictor.name,
            "model_name": "Stage 2 multi-label MLP" if predictor.predictor.name == "neural" else "Stage 2 logistic regression baseline",
            "run_id": metadata["run_id"], "model_version": metadata["version"],
            "features": fields, "conditions": [{"id": label, "name": name} for label, name in zip(config.LABELS, config.CONDITIONS)],
            "output_type": "independent_multi_label_scores", "explanation_supported": True,
            "limitations": metadata["limitations"]}


def main():
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import torch
            torch.set_num_threads(1)
            from models_stage2 import config
            from models_stage2.predict import Stage2Predictor, patient_frame
            predictor = Stage2Predictor(model_dir=Path(sys.argv[1]))
            metadata = schema(config, predictor)
        emit({"ok": True, "result": metadata})
    except Exception as error:
        emit({"ok": False, "error_type": type(error).__name__})
        return 1
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                patient_frame(request["features"])
                if request.get("explain", True):
                    try:
                        result = predictor.predict(request["features"], explain=True)
                    except Exception:
                        # A failed explanation never discards successful classification.
                        result = predictor.predict(request["features"], explain=False)
                        result["warnings"].append("Stage 2 classification completed, but its explanation is unavailable.")
                else:
                    result = predictor.predict(request["features"], explain=False)
                result["model_name"] = metadata["model_name"]
            emit({"ok": True, "result": result})
        except Exception as error:
            emit({"ok": False, "error_type": type(error).__name__})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
