"""API routes deliberately avoid logging or persisting submitted features."""
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from .explanations import explain_forest
from .inference import predict_score
from .registry import SPECS
from .schemas import DISCLAIMER, PredictionRequest, PredictionResponse, features_to_frame, schema_metadata

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


@router.get("/health")
def health(request: Request):
    registry = request.app.state.registry
    return {"status": "ok" if len(registry.loaded) == len(SPECS) else "degraded", "ready": bool(registry.loaded), "available_models": list(registry.loaded), "default_model_id": registry.default_model_id}


@router.get("/models")
def models(request: Request):
    return request.app.state.registry.metadata()


@router.get("/schema")
def schema():
    return schema_metadata()


@router.get("/models/{model_id}/plots/{filename}")
def plot(model_id: str, filename: str, request: Request):
    registry = request.app.state.registry
    # Exact allowlist rather than user-controlled path concatenation.
    if model_id not in SPECS or filename not in registry.plots(model_id):
        raise HTTPException(status_code=404, detail="Recorded model figure not found.")
    return FileResponse(registry.plots(model_id)[filename], media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@router.post("/predict", response_model=PredictionResponse, response_model_exclude_none=True)
def predict(payload: PredictionRequest, request: Request):
    registry = request.app.state.registry
    loaded = registry.loaded.get(payload.model_id)
    if loaded is None:
        raise HTTPException(status_code=503, detail=f"Requested model is unavailable. {registry.errors.get(payload.model_id, '')} Check /api/models for training instructions.")
    frame = features_to_frame(payload.features)
    try:
        score = predict_score(loaded, frame)
    except Exception as error:
        # Do not include patient values, exception data, or request bodies in logs.
        logger.error("Inference failed for model %s (%s)", payload.model_id, type(error).__name__)
        raise HTTPException(status_code=500, detail="The selected model could not process this assessment. Verify its matched artifact and preprocessor, then retry.") from None
    explanations = None
    if loaded.explanation_ready:
        try:
            explanations = explain_forest(loaded.model, frame, loaded.positive_index, score)
        except Exception as error:
            logger.warning("Explanation unavailable for model %s (%s)", payload.model_id, type(error).__name__)
    threshold = registry.settings.thresholds[payload.model_id]
    predicted = int(score >= threshold)
    return PredictionResponse(model_id=payload.model_id, predicted_class=predicted, predicted_label="Disease-associated pattern detected" if predicted else "No disease-associated pattern detected", model_score=score, threshold=threshold, missing_features=[name for name, value in payload.features.model_dump().items() if value is None], explanation_status="available" if explanations else "Explanation currently unavailable.", explanations=explanations, disclaimer=DISCLAIMER)
