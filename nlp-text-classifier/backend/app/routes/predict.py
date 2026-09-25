"""
routes/predict.py — Unified inference routing.

POST /predict/nlp-sentiment
POST /predict/vision-transformer

For backward compatibility, the old POST /predict endpoint is maintained
and defaults to nlp-sentiment.
"""

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.logging_config import get_logger
from app.schemas import PredictRequest, PredictResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/predict", tags=["predict"])


def _run_inference(model_id: str, payload: any, request: Request):
    """Helper to dispatch payload to the correct adapter."""
    registry = request.app.state.registry
    adapter = registry.get(model_id)

    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    
    if not adapter.is_loaded():
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_id}' is temporarily unavailable."
        )

    try:
        result = adapter.predict(payload)
        return result
    except ValueError as exc:
        # Validation error from preprocessing
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Inference error for %s: %s", model_id, exc)
        raise HTTPException(
            status_code=500,
            detail="Inference failed. Please try again."
        )


# ── Backward Compatibility ────────────────────────────────────────────────────

@router.post("", response_model=PredictResponse, summary="Predict (Legacy NLP)")
def predict_legacy(payload: PredictRequest, request: Request):
    """
    Backward-compatible endpoint for the original NLP frontend.
    Hardcodes routing to 'nlp-sentiment'.
    """
    return _run_inference("nlp-sentiment", payload.text, request)


# ── Text Models ───────────────────────────────────────────────────────────────

@router.post("/nlp-sentiment", response_model=PredictResponse, summary="NLP Sentiment")
def predict_nlp(payload: PredictRequest, request: Request):
    return _run_inference("nlp-sentiment", payload.text, request)


# ── Vision Models ─────────────────────────────────────────────────────────────

@router.post("/vision-transformer", summary="Vision Transformer")
async def predict_vit(request: Request, file: UploadFile = File(...)):
    """
    Accepts multipart/form-data with a file upload.
    Validates file type and size before inference.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Invalid file type. Please upload an image."
        )

    # Read bytes and validate size (< 5MB)
    file_bytes = await file.read()
    if len(file_bytes) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image too large. Please upload an image smaller than 5MB."
        )

    return _run_inference("vision-transformer", file_bytes, request)
