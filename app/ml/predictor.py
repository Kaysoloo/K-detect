"""
K-Detect tomato quality predictor — SVM Phase 1.
Uses trained SVM model with handcrafted CV features.
"""

from __future__ import annotations

import os
from io import BytesIO
from typing import Any

import cv2
import joblib
import numpy as np
from PIL import Image

# ── Load model bundle ──────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "kdetect_svm.joblib")
_bundle = None

def _get_bundle():
    global _bundle
    if _bundle is None:
        if os.path.exists(_MODEL_PATH):
            _bundle = joblib.load(_MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model not found at {_MODEL_PATH}. Run scripts/train_model.py first.")
    return _bundle


# ── Feature extraction (must match train_model.py exactly) ──
def extract_features(image_bytes: bytes) -> np.ndarray:
    """Extract the same 38-dim feature vector used during training."""
    # Load image from bytes
    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    img = cv2.GaussianBlur(img, (3, 3), 0)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    feats = []

    # Color: mean, std, median per channel (3x3x3 = 27)
    for sp in [rgb, hsv, lab]:
        for ch in range(3):
            c = sp[:, :, ch].astype(np.float32) / 255.0
            feats.append(float(np.mean(c)))
            feats.append(float(np.std(c)))
            feats.append(float(np.median(c)))

    # Texture: GLCM (5)
    gq = (gray // 8).clip(0, 31).astype(np.uint8)
    glcm = np.zeros((32, 32), dtype=np.float64)
    pairs = np.column_stack([gq[:, :-1].ravel(), gq[:, 1:].ravel()])
    np.add.at(glcm, (pairs[:, 0], pairs[:, 1]), 1)
    total = glcm.sum()
    if total > 0:
        glcm /= total
        ix = np.arange(32)
        I, J = np.meshgrid(ix, ix, indexing='ij')
        contrast = np.sum(glcm * (I - J) ** 2)
        dissim = np.sum(glcm * np.abs(I - J))
        homo = np.sum(glcm / (1 + (I - J) ** 2))
        energy = np.sum(glcm ** 2)
        im = np.sum(I * glcm)
        jm = np.sum(J * glcm)
        istd = np.sqrt(max(1e-9, np.sum(((I - im) ** 2) * glcm)))
        jstd = np.sqrt(max(1e-9, np.sum(((J - jm) ** 2) * glcm)))
        corr = np.sum(((I - im) * (J - jm) * glcm)) / (istd * jstd)
        feats.extend([float(contrast), float(dissim), float(homo), float(energy), float(corr)])
    else:
        feats.extend([0.0] * 5)

    # Dark spot ratios (2)
    gf = gray.astype(np.float32)
    feats.append(float((gf < 40).mean()))
    feats.append(float((gf < 20).mean()))

    # Shape (3)
    hsv_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv_u8, (0, 35, 35), (20, 255, 255))
    m2 = cv2.inRange(hsv_u8, (160, 35, 35), (180, 255, 255))
    m3 = cv2.inRange(hsv_u8, (20, 30, 30), (85, 255, 255))
    mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area > 30:
            perim = cv2.arcLength(cnt, True)
            circ = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0
            _, _, w, h = cv2.boundingRect(cnt)
            asp = float(w) / float(h) if h > 0 else 0
            ha = cv2.contourArea(cv2.convexHull(cnt))
            sol = float(area) / ha if ha > 0 else 0
            feats.extend([float(circ), float(asp), float(sol)])
        else:
            feats.extend([0.0] * 3)
    else:
        feats.extend([0.0] * 3)

    # Coverage (1)
    feats.append(float(mask.mean()))

    arr = np.array(feats, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


# ── Defect detection heuristics (for explainability) ───────
def _detect_defects(feat_vec: np.ndarray) -> list[str]:
    """Use the feature vector to infer visible defects for the user."""
    defects = []
    # feat indices: 0-26=color stats, 27-31=GLCM, 32-33=dark spots, 34-36=shape, 37=coverage
    dark_ratio = float(feat_vec[32])
    very_dark = float(feat_vec[33])
    coverage = float(feat_vec[37])

    # GLCM contrast (index 27) indicates texture roughness
    glcm_contrast = float(feat_vec[27]) if len(feat_vec) > 27 else 0.0

    if very_dark > 0.04:
        defects.append("Surface Rot")
    elif dark_ratio > 0.12:
        defects.append("Dark Spots / Bruising")
    elif dark_ratio > 0.05:
        defects.append("Minor Blemishes")

    # Check green channel (unripe): HSV mean[1] is index 4, std[1] is index 5
    # Green in RGB: g_mean idx 1, g_std idx 2
    if len(feat_vec) > 2:
        g_mean = float(feat_vec[1])
        r_mean = float(feat_vec[0])
        if g_mean > r_mean * 1.15:
            defects.append("Unripe (Green Patches)")

    if glcm_contrast > 80:
        defects.append("Rough Surface Texture")

    if coverage < 0.06:
        defects.append("Tomato Not Clearly Detected")

    if not defects:
        defects = ["No major visible defects detected"]

    return defects


# ── Main prediction function ───────────────────────────────
def predict_tomato_quality(image_bytes: bytes) -> dict[str, Any]:
    """Run the full SVM prediction pipeline."""
    bundle = _get_bundle()
    model = bundle["model"]
    scaler = bundle["scaler"]
    le = bundle["label_encoder"]

    # Extract features
    feat_vec = extract_features(image_bytes)

    # Scale + predict
    X = scaler.transform(feat_vec.reshape(1, -1))

    # Get probabilities
    probas = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probas))
    quality = str(le.inverse_transform([pred_idx])[0])
    confidence = round(float(probas[pred_idx]) * 100, 2)
    quality_score = int(round(probas[pred_idx] * 100))

    # Detect defects from features
    defects = _detect_defects(feat_vec)

    # Build metrics for display
    metrics = {
        "dark_spot_ratio": round(float(feat_vec[32]), 4),
        "very_dark_ratio": round(float(feat_vec[33]), 4),
        "coverage": round(float(feat_vec[37]), 4),
        "glcm_contrast": round(float(feat_vec[27]), 4),
    }

    # Map confidence to quality_score (0-100 scale)
    if quality == "Good":
        quality_score = max(75, quality_score)
    elif quality == "Medium":
        quality_score = max(45, min(74, quality_score))
    else:
        quality_score = min(44, quality_score)

    return {
        "quality": quality,
        "confidence": confidence,
        "quality_score": quality_score,
        "detected_defects": defects,
        "metrics": metrics,
        "model_version": bundle.get("model_version", "svm-phase1-v1"),
    }
