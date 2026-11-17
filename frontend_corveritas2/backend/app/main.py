"""Run from the project folder: uvicorn app.main:app --app-dir backend."""
from contextlib import asynccontextmanager

from .config import configure_runtime_environment

# Keep local inference modest; callers can explicitly override these settings.
configure_runtime_environment()

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings
from .registry import ModelRegistry
from .routes import router
from .stage2_routes import router as stage2_router
from .stage2_service import Stage2Service


def create_app(settings=None, registry=None, stage2=None):
    settings = settings or Settings.from_env()
    registry = registry or ModelRegistry(settings)
    stage2 = stage2 if stage2 is not None else Stage2Service()

    @asynccontextmanager
    async def lifespan(app):
        registry.load_all()
        try:
            yield
        finally:
            stage2.close()

    application = FastAPI(title="CorVeritas Research API", version="1.0.0", lifespan=lifespan, description="Local binary cardiovascular model inference. Educational use only.")
    application.state.registry = registry
    application.state.stage2 = stage2
    application.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        # FastAPI's default includes invalid inputs; redact health data in errors.
        details = [{"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]} for item in error.errors()]
        return JSONResponse(status_code=422, content={"detail": details})

    application.include_router(router)
    application.include_router(stage2_router)
    return application


app = create_app()
