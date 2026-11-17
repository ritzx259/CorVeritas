"""Additive Stage 2 endpoints; the original Stage 1 endpoints stay unchanged."""
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

from .inference import predict_score
from .schemas import PredictionRequest
from .stage2_service import Stage2Unavailable

router = APIRouter(prefix="/api/stage2", tags=["Stage 2"])


class AdditionalFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    bmi: Annotated[float, Field(ge=10, le=100)] | None = None
    systolic_bp: Annotated[float, Field(ge=50, le=300)] | None = None
    diastolic_bp: Annotated[float, Field(ge=20, le=200)] | None = None
    total_cholesterol: Annotated[float, Field(ge=40, le=1000)] | None = None
    diabetes_history: Literal["Yes", "No", "Borderline", "Unknown"] | None = None
    smoking_history: Literal["Yes", "No", "Unknown"] | None = None

    @field_validator("bmi", "systolic_bp", "diastolic_bp", "total_cholesterol", mode="before")
    @classmethod
    def numbers_only(cls, value):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError("Provide a JSON number or null.")
        return value


class Stage2Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage1: PredictionRequest
    additional_features: AdditionalFeatures = Field(default_factory=AdditionalFeatures)
    explain: StrictBool = True


def reusable_features(features, fields):
    """Only reuse explicit, compatible mappings supplied by the Stage 2 schema."""
    result = {}
    for field in fields:
        value = features.get(field.get("source_stage1"))
        if value is None:
            continue
        if field["type"] == "number" and (isinstance(value, bool) or not field["min"] <= value <= field["max"]):
            continue
        if field["type"] == "category" and value not in field["categories"]:
            continue
        result[field["name"]] = value
    return result


@router.get("/schema")
def schema(request: Request):
    return request.app.state.stage2.status()


@router.post("/predict")
def predict(payload: Stage2Request, request: Request):
    # Verify eligibility on the server using the unchanged Stage 1 inference.
    # Never trust a client-provided positive flag, or replace its displayed result.
    registry = request.app.state.registry
    loaded = registry.loaded.get(payload.stage1.model_id)
    if loaded is None:
        raise HTTPException(503, "Stage 1 eligibility could not be verified because its selected model is unavailable. The existing Stage 1 result is preserved.")
    from .schemas import features_to_frame
    try:
        score = predict_score(loaded, features_to_frame(payload.stage1.features))
    except Exception:
        raise HTTPException(503, "Stage 1 eligibility could not be verified. The existing Stage 1 result is preserved.") from None
    if score < registry.settings.thresholds[payload.stage1.model_id]:
        raise HTTPException(409, "Disease type classification was not performed because Stage 1 did not indicate cardiovascular disease.")
    if payload.stage1.features.age < 20:
        raise HTTPException(422, "Stage 2 supports adults aged 20–120. Stage 1 remains valid for its own supported age range.")
    service = request.app.state.stage2
    metadata = service.status()
    if not metadata["available"]:
        raise HTTPException(503, metadata["reason"])
    features = reusable_features(payload.stage1.features.model_dump(), metadata["features"])
    for name, value in payload.additional_features.model_dump(exclude_none=True).items():
        if name in features and features[name] != value:
            raise HTTPException(422, "Additional Stage 2 values must not override the compatible values reused from Stage 1.")
        features[name] = value
    if features.get("systolic_bp") is not None and features.get("diastolic_bp") is not None and features["systolic_bp"] <= features["diastolic_bp"]:
        raise HTTPException(422, "Stage 2 systolic pressure must be greater than diastolic pressure.")
    patient = {field["name"]: features.get(field["name"]) for field in metadata["features"]}
    try:
        return service.predict(patient, explain=payload.explain)
    except Stage2Unavailable as error:
        raise HTTPException(503, str(error)) from None
