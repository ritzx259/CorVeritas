import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import FRONTEND_ORIGINS


logging.basicConfig(level=logging.INFO)
app = FastAPI(
    title="CorVeritas API",
    version="1.0.0",
    description="Local binary cardiovascular disease research assessment API.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(router)
