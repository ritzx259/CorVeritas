"""Fixed public data schema and defaults for the independent Stage 2 prototype."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/"
SEED = 42
VERSION = "corveritas-stage2-1.0.0"
TARGETS = {
    "coronary_heart_disease": "MCQ160c",
    "heart_failure": "MCQ160b",
    "heart_attack_history": "MCQ160e",
}
LABELS = list(TARGETS)
CONDITIONS = ["Reported coronary heart disease history", "Reported congestive heart failure history",
              "Reported previous myocardial infarction"]
NUMERIC = ["age", "bmi", "systolic_bp", "diastolic_bp", "total_cholesterol"]
CATEGORICAL = ["sex", "diabetes_history", "smoking_history"]
FEATURES = NUMERIC + CATEGORICAL
CATEGORIES = [["Female", "Male", "Unknown"], ["No", "Yes", "Borderline", "Unknown"],
              ["No", "Yes", "Unknown"]]
# Broad engineering guards, not clinical reference intervals or CDC hard edits.
RANGES = {"age": (20, 120), "bmi": (10, 100), "systolic_bp": (50, 300),
          "diastolic_bp": (20, 200), "total_cholesterol": (40, 1000)}
MODULE_COLUMNS = {
    "MCQ_L": ["SEQN", *TARGETS.values()],
    "DEMO_L": ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDSTATR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"],
    "BMX_L": ["SEQN", "BMXBMI"],
    "BPXO_L": ["SEQN", *[f"BPXOSY{i}" for i in range(1, 4)], *[f"BPXODI{i}" for i in range(1, 4)]],
    "TCHOL_L": ["SEQN", "LBXTC", "WTPH2YR"],
    "DIQ_L": ["SEQN", "DIQ010"],
    "SMQ_L": ["SEQN", "SMQ020"],
}
LIMITATIONS = [
    "Research prototype, not a validated diagnostic tool or a substitute for medical assessment.",
    "Scores estimate associations with self-reported previous diagnoses, not future disease onset.",
    "Uncalibrated model scores; positive weighting can materially alter probability calibration.",
    "Cross-sectional measurements may follow diagnosis, treatment or lifestyle changes; contributions are not causal.",
    "Unweighted NHANES sample evaluation is not a US population estimate; survey design is not modeled.",
    "Complete-label selection, missing measurements, small positive counts and US sampling limit generalization.",
]


def ensure_dirs():
    for path in (RAW, PROCESSED, MODELS, RESULTS, RESULTS / "plots"):
        path.mkdir(parents=True, exist_ok=True)
