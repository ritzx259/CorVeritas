"""A strict public contract matching the inspected CSV and fitted encoders."""
from typing import Annotated, Any, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

FEATURE_ORDER = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal"]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
CATEGORICAL_FEATURES = [name for name in FEATURE_ORDER if name not in NUMERIC_FEATURES]
CATEGORIES = {
    "sex": ["Male", "Female"],
    "cp": ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
    "fbs": [False, True],
    "restecg": ["normal", "st-t abnormality", "lv hypertrophy"],
    "exang": [False, True],
    "slope": ["upsloping", "flat", "downsloping"],
    "thal": ["normal", "fixed defect", "reversable defect"],
}
LABELS = {"age": "Age", "sex": "Sex recorded in dataset", "cp": "Chest pain category", "trestbps": "Resting blood pressure", "chol": "Serum cholesterol", "fbs": "Fasting blood sugar above 120 mg/dL", "restecg": "Resting ECG result", "thalch": "Maximum heart rate achieved", "exang": "Exercise-induced angina", "oldpeak": "ST depression", "slope": "Peak-exercise ST segment slope", "ca": "Major vessels visualized", "thal": "Thallium stress-test result"}
DISCLAIMER = (
    "Research and educational use only. These models are not clinically validated medical devices. "
    "Predictions neither establish nor exclude a diagnosis. Model scores are machine-learning estimates, "
    "not necessarily calibrated clinical risk probabilities. A qualified healthcare professional must "
    "interpret clinical measurements and results. Seek immediate medical attention for acute concerning "
    "symptoms regardless of this result."
)


class Features(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    age: Annotated[StrictInt, Field(ge=18, le=120)]
    sex: Literal["Male", "Female"]
    cp: Literal["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
    trestbps: Optional[Annotated[float, Field(ge=40, le=300)]] = None
    chol: Optional[Annotated[float, Field(ge=0, le=1000)]] = None
    fbs: Optional[StrictBool] = None
    restecg: Optional[Literal["normal", "st-t abnormality", "lv hypertrophy"]] = None
    thalch: Optional[Annotated[float, Field(ge=30, le=250)]] = None
    exang: Optional[StrictBool] = None
    oldpeak: Optional[Annotated[float, Field(ge=-5, le=15)]] = None
    slope: Optional[Literal["upsloping", "flat", "downsloping"]] = None
    ca: Optional[Annotated[StrictInt, Field(ge=0, le=3)]] = None
    thal: Optional[Literal["normal", "fixed defect", "reversable defect"]] = None

    @field_validator("trestbps", "chol", "thalch", "oldpeak", mode="before")
    @classmethod
    def numerical_json_values(cls, value):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError("Provide a JSON number or null, not text or a boolean.")
        return value


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: Literal["mlp", "cnn", "random_forest"]
    features: Features


class Contribution(BaseModel):
    feature: str
    label: str
    value: float


class Explanation(BaseModel):
    method: str
    baseline: float
    score: float
    reconstruction_error: float
    contributions: list[Contribution]
    global_importance: list[Contribution]
    note: str


class PredictionResponse(BaseModel):
    model_id: str
    predicted_class: Literal[0, 1]
    predicted_label: str
    model_score: Annotated[float, Field(ge=0, le=1)]
    threshold: float
    missing_features: list[str]
    explanation_status: str
    explanations: Optional[Explanation] = None
    disclaimer: str


def features_to_frame(features: Features) -> pd.DataFrame:
    """Keep original order; explicit object dtype preserves bools plus np.nan."""
    values = features.model_dump()
    frame = pd.DataFrame(index=[0])
    for name in FEATURE_ORDER:
        value = values[name]
        if name in NUMERIC_FEATURES:
            frame[name] = pd.Series([np.nan if value is None else value], dtype="float64")
        else:
            frame[name] = pd.Series([np.nan if value is None else value], dtype="object")
    return frame[FEATURE_ORDER]


def schema_metadata() -> dict[str, Any]:
    definitions = [
        ("age", "integer", True, "years", 18, 120, "demographic", "Adults only. Enter completed years."),
        ("sex", "category", True, None, None, None, "demographic", "The original dataset uses only these two recorded categories; this is a dataset limitation."),
        ("cp", "category", True, None, None, None, "reported", "Use the chest-pain category recorded by a healthcare professional. Asymptomatic means no reported chest pain."),
        ("trestbps", "number", False, "mmHg", 40, 300, "clinical", "Resting systolic blood pressure measured at admission."),
        ("chol", "number", False, "mg/dL", 0, 1000, "clinical", "A laboratory measurement of serum cholesterol. Unknown should be left unavailable, not zero."),
        ("fbs", "boolean", False, None, None, None, "clinical", "A fasting blood test result, not a symptom or a diagnosis of diabetes."),
        ("restecg", "category", False, None, None, None, "clinical", "Resting electrocardiogram interpretation; requires a clinical test."),
        ("thalch", "number", False, "bpm", 30, 250, "clinical", "Maximum heart rate achieved during the recorded exercise test."),
        ("exang", "boolean", False, None, None, None, "reported", "Chest pain or discomfort induced by exercise, as recorded in the assessment."),
        ("oldpeak", "number", False, "mm", -5, 15, "clinical", "Exercise-induced ST depression relative to rest, interpreted from ECG testing."),
        ("slope", "category", False, None, None, None, "clinical", "Slope of the peak-exercise ST segment on ECG."),
        ("ca", "integer", False, "vessels", 0, 3, "clinical", "Number of major vessels visualized by fluoroscopy; requires clinical imaging."),
        ("thal", "category", False, None, None, None, "clinical", "Thallium stress-test category, not thalassemia. Requires a test result."),
    ]
    fields = []
    for name, type_, required, unit, minimum, maximum, source, description in definitions:
        categories = None
        if name in CATEGORIES:
            categories = [{"value": value, "label": ("Yes" if value is True else "No" if value is False else "reversible defect" if value == "reversable defect" else value)} for value in CATEGORIES[name]]
        fields.append(dict(name=name, label=LABELS[name], type=type_, required=required, unit=unit, min=minimum, max=maximum, source=source, description=description, categories=categories))
    return {"fields": fields, "feature_order": FEATURE_ORDER, "dataset": {"name": "UCI Heart Disease (combined four-source CSV)", "rows": 920, "features": 13, "target": "0 = no recorded disease; original num 1–4 = recorded disease (binary target)", "split": "Stratified 80% training / 20% test, random_state=42"}}
