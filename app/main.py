from __future__ import annotations

import os, time
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.database import init_db, save_prediction
from app.ml.predictor import predict_tomato_quality

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(ROOT, "frontend", "index.html")

app = FastAPI(title="K-Detect API", description="AI-powered tomato quality detection", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory rate limiter
_rate_log: dict[str, list[float]] = {}

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown")
    now = time.time()
    window = 60  # 60 seconds
    max_req = 30  # max requests per window

    if client_ip not in _rate_log:
        _rate_log[client_ip] = []
    _rate_log[client_ip] = [t for t in _rate_log[client_ip] if now - t < window]
    _rate_log[client_ip].append(now)

    if len(_rate_log[client_ip]) > max_req:
        return JSONResponse(status_code=429, content={"detail": "Too many requests. Slow down."})

    return await call_next(request)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "K-Detect",
        "version": "0.3.0",
        "model": "SVM Phase 1 — 38 handcrafted features",
        "classes": ["Good", "Medium", "Poor"],
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    # Validate type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{file.content_type}'. Upload JPG, PNG, or WebP.",
        )

    # Read
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Size check
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(image_bytes) / 1e6:.1f} MB). Max is 10 MB.",
        )

    # Predict
    try:
        result = predict_tomato_quality(image_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not process this image. It may be corrupted or unsupported.",
        ) from exc

    # Silently log
    try:
        save_prediction(file.filename or "upload", result)
    except Exception:
        pass

    return result
