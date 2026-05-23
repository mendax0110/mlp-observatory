from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mlp_observatory.api.health import router as health_router
from mlp_observatory.api.runs import router as runs_router
from mlp_observatory.api.stream import router as stream_router
from mlp_observatory.app.container import create_container
from mlp_observatory.core.logging import configure_logging
from mlp_observatory.core.settings import settings


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="MLP Observatory", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.container = create_container()
    app.include_router(health_router)
    app.include_router(runs_router)
    app.include_router(stream_router)
    return app
