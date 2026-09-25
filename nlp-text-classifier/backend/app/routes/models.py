"""
routes/models.py — Model registry API routes.

GET /models            → list all registered models (summary)
GET /models/{model_id} → full metadata for one model
"""

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", summary="List all models")
def list_models(request: Request):
    registry = request.app.state.registry
    return [
        {
            "id": a.model_id,
            "name": a.display_name,
            "type": a.model_type,
            "status": "live" if a.is_loaded() else "unavailable",
        }
        for a in registry.all()
    ]


@router.get("/{model_id}", summary="Get model metadata")
def get_model(model_id: str, request: Request):
    registry = request.app.state.registry
    adapter = registry.get(model_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    return adapter.metadata()
