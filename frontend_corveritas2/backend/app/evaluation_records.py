"""Read saved metrics; never evaluate models while serving the website."""
import hashlib
import json
import math

from .config import APP_ROOT, REPO_ROOT

METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")
EVALUATION_DIR = APP_ROOT / "verification" / "evaluations"
DATASET_PATH = REPO_ROOT / "data" / "heart_disease_uci.csv"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def saved_evaluation(loaded, required_files):
    if loaded is None:
        return None
    paths = ((loaded.directory / f"{loaded.id}_metrics.json", False),
             (EVALUATION_DIR / f"{loaded.id}_metrics.json", True))
    for path, fingerprint_required in paths:
        try:
            record = json.loads(path.read_text())
            provenance = record.get("provenance")
            values = {key: record[key] for key in METRIC_KEYS if key in record}
            if not values or not isinstance(provenance, str) or not provenance.strip():
                continue
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
                continue
            evaluation = record.get("evaluation")
            if fingerprint_required:
                # The application-local records describe the exact selected artifacts,
                # not a different model sharing its name or a Stage 2 diagnosis model.
                if record.get("model_id") != loaded.id:
                    continue
                if record.get("artifact_sha256") != {name: sha256(loaded.directory / name) for name in required_files}:
                    continue
                if record.get("dataset", {}).get("sha256") != sha256(DATASET_PATH):
                    continue
                if not isinstance(evaluation, dict) or evaluation.get("partition") not in {"validation", "test"}:
                    continue
                count = evaluation.get("sample_count")
                threshold = evaluation.get("threshold")
                if type(count) is not int or count <= 0 or type(threshold) not in (float, int) or not 0 < threshold < 1:
                    continue
            return {"metrics": values, "provenance": provenance, "evaluation": evaluation, "source": str(path)}
        except (ValueError, TypeError, OSError, AttributeError):
            continue
    return None
