"""
main.py — FastAPI application entry point.

Endpoints:
  GET  /          — API info
  GET  /health    — Health check (reflects real model state)
  POST /predict   — Run sentiment inference

Model is loaded ONCE in the lifespan startup handler.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import model as ml
from app.config import settings
from app.logging_config import get_logger, setup_logging
from app.preprocessing import preprocess
from app.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    Probabilities,
    RootResponse,
)

setup_logging()
logger = get_logger(__name__)


# ── Lifespan: load model once on startup ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up NLP Text Classifier API …")
    try:
        ml.load_model()
    except Exception as exc:
        logger.error("Model failed to load: %s", exc)
        # App still starts — /health will report model_loaded=false
    yield
    logger.info("Shutting down.")


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


# ── Routes ───────────────────────────────────────────────────────────────────

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
    description="Returns API health and whether the ML model is loaded.",
)
def health():
    return HealthResponse(
        status="healthy",
        model_loaded=ml.is_loaded(),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict sentiment",
    description=(
        "Classify a movie review as **positive** or **negative** using the "
        "TensorFlow / TF Hub sentiment model. "
        "Returns the predicted label, confidence score, full probability "
        "breakdown, and measured inference latency."
    ),
)
def predict(request: PredictRequest):
    if not ml.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Please try again later.",
        )

    try:
        clean_text = preprocess(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        result = ml.predict(clean_text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Inference error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Inference failed. Please try again.",
        )

    return PredictResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=Probabilities(**result["probabilities"]),
        inference_time_ms=result["inference_time_ms"],
    )
