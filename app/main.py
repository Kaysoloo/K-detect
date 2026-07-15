from __future__ import annotations

import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from app.database import init_db, list_predictions, save_prediction
from app.ml.predictor import predict_tomato_quality

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}

# Project root for resolving file paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(ROOT, "frontend", "index.html")

app = FastAPI(
    title="K-Detect API",
    description="AI-powered tomato quality detection MVP",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "K-Detect"}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Upload a JPG, JPEG, or PNG tomato image.")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    try:
        result = predict_tomato_quality(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process image: {exc}") from exc
    prediction_id = save_prediction(file.filename or "uploaded-image", result)
    return {"id": prediction_id, **result}


@app.get("/api/history")
def history(limit: int = 25) -> dict:
    limit = max(1, min(limit, 100))
    return {"items": list_predictions(limit=limit)}
