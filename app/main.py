from __future__ import annotations

import os
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.database import init_db, save_prediction
from app.ml.predictor import predict_tomato_quality

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(ROOT, "frontend", "index.html")

app = FastAPI(
    title="K-Detect API",
    description="AI-powered tomato quality detection",
    version="0.2.0",
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
    return {"status": "ok", "service": "K-Detect", "version": "0.2.0"}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Upload a JPG, JPEG, PNG, or WebP image.")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    try:
        result = predict_tomato_quality(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process image: {exc}") from exc

    # Silently save for analytics but don't expose history endpoint
    try:
        save_prediction(file.filename or "uploaded-image", result)
    except Exception:
        pass  # never fail on DB write

    return result
