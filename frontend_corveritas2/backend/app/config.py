"""Local configuration. Serialized artifacts must come from trusted training runs."""
import os
from dataclasses import dataclass
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = APP_ROOT.parent


def configure_runtime_environment():
    """Confine default ML configuration/cache writes to this application folder."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    defaults = {
        "KERAS_HOME": APP_ROOT / ".cache" / "keras",
        "XDG_CACHE_HOME": APP_ROOT / ".cache",
        "MPLCONFIGDIR": APP_ROOT / ".cache" / "matplotlib",
    }
    for name, directory in defaults.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(directory))


@dataclass(frozen=True)
class Settings:
    model_dirs: tuple[Path, ...]
    default_model: str
    thresholds: dict[str, float]
    cors_origins: tuple[str, ...]

    @classmethod
    def from_env(cls):
        configured = os.getenv("CORVERITAS_MODEL_DIR")
        if configured:
            directory = Path(configured).expanduser()
            if not directory.is_absolute():
                directory = APP_ROOT / directory
            directories = (directory.resolve(),)
        else:
            directories = (APP_ROOT / "models", REPO_ROOT / "models")
        thresholds = {}
        for model_id in ("mlp", "cnn", "random_forest"):
            threshold = float(os.getenv(f"CORVERITAS_THRESHOLD_{model_id.upper()}", "0.5"))
            if not 0 < threshold < 1:
                raise ValueError(f"Threshold for {model_id} must be between zero and one.")
            thresholds[model_id] = threshold
        origins = tuple(origin.strip() for origin in os.getenv(
            "CORVERITAS_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",") if origin.strip())
        return cls(directories, os.getenv("CORVERITAS_DEFAULT_MODEL", "random_forest"), thresholds, origins)
