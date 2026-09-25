"""
main.py — FastAPI application entry point.

AI Model Lab API.
Supports dynamic loading of multiple models via a unified Model Registry.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.logging_config import get_logger, setup_logging
from app.registry.registry import ModelRegistry
from app.routes import models as models_router
from app.routes import predict as predict_router
from app.schemas import HealthResponse, RootResponse

setup_logging()
logger = get_logger(__name__)


# ── Lifespan: load models once on startup ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up AI Model Lab API …")
    
    # Initialize registry and attach to app state
    registry = ModelRegistry()
    app.state.registry = registry
    
    # Try to load all registered models
    registry.load_all()
    
    yield
    logger.info("Shutting down AI Model Lab API.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Global exception handler — never expose stack traces ─────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )


# ── Base Routes ───────────────────────────────────────────────────────────────

@app.get(
    "/",
    response_model=RootResponse,
    summary="API root",
    description="Returns basic API metadata.",
)
def root():
    return RootResponse(
        name=settings.APP_NAME,
        version=settings.APP_VERSION,
        status="running",
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns API health and whether AT LEAST ONE ML model is loaded.",
)
def health(request: Request):
    registry = request.app.state.registry
    return HealthResponse(
        status="healthy",
        model_loaded=registry.any_loaded(),
        # We can add full status dict later if needed
    )

# ── Mount Feature Routers ─────────────────────────────────────────────────────

app.include_router(models_router.router)
app.include_router(predict_router.router)
