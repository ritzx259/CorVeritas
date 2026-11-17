import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import FEATURE_SCHEMA, MODEL_ARTIFACTS, MODEL_DIR, MODEL_LIMITATIONS
from app.inference import predict
from app.model_registry import ModelRegistry
from app.schemas import PredictionRequest, PredictionResponse


LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
registry = ModelRegistry()


@router.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if registry.available else "degraded",
        "ready": bool(registry.available),
        "available_model_ids": list(registry.available),
        "model_errors": registry.errors,
    }


@router.get("/models")
def models() -> dict[str, Any]:
    return {
        "default_model_id": (
            registry.default_model_id
            if registry.default_model_id in registry.available
            else None
        ),
        "models": registry.list_models(),
        "disclaimer": MODEL_LIMITATIONS,
    }


@router.get("/schema")
def schema() -> dict[str, Any]:
    return {"features": FEATURE_SCHEMA, "feature_order": [f["name"] for f in FEATURE_SCHEMA]}


@router.post(
    "/predict",
    response_model=PredictionResponse,
    response_model_exclude_none=True,
)
def create_prediction(request: PredictionRequest) -> PredictionResponse:
    registered = registry.get(request.model_id)
    if registered is None:
        if request.model_id in registry.errors:
            raise HTTPException(
                status_code=503,
                detail=registry.errors[request.model_id],
            )
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model_id `{request.model_id}`. No model was selected as a fallback.",
        )
    try:
        return predict(registered, request.features.ordered_dict())
    except (ValueError, TypeError) as exc:
        LOGGER.exception("Inference failed for model %s", request.model_id)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


@router.get("/model-assets/{model_id}/{filename}")
def model_asset(model_id: str, filename: str) -> FileResponse:
    metadata = MODEL_ARTIFACTS.get(model_id)
    if metadata is None or filename not in metadata["graphs"]:
        raise HTTPException(status_code=404, detail="Model asset not found.")
    path = MODEL_DIR / metadata["graph_directory"] / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Model asset not found.")
    return FileResponse(path)
