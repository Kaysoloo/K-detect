"""
K-Detect tomato quality predictor — SVM Phase 1.
Validates input is actually a tomato before classifying.
"""

from __future__ import annotations

import os
from io import BytesIO
from typing import Any

import cv2
import joblib
import numpy as np
from PIL import Image

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "kdetect_svm.joblib",
)
_bundle = None


def _get_bundle():
    global _bundle
    if _bundle is None:
        if os.path.exists(_MODEL_PATH):
            _bundle = joblib.load(_MODEL_PATH)
        else:
            raise FileNotFoundError(
                f"Model not found at {_MODEL_PATH}. Run scripts/train_model.py first."
            )
    return _bundle


# ── Tomato detection ───────────────────────────────────────
def _is_tomato(img_bgr: np.ndarray) -> tuple[bool, float, str]:
    """Check if the image contains a tomato. Returns (is_tomato, confidence, reason)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Red ranges (two wraps around 0/180 in OpenCV HSV)
    red1 = cv2.inRange(hsv, (0, 30, 40), (18, 255, 255))
    red2 = cv2.inRange(hsv, (160, 30, 40), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    # Orange/yellow range (ripe tomatoes, some defects)
    orange = cv2.inRange(hsv, (10, 40, 50), (30, 255, 255))

    # Green range (unripe tomatoes)
    green = cv2.inRange(hsv, (30, 25, 30), (80, 255, 255))

    # Brown/dark range (rot spots on tomatoes)
    brown = cv2.inRange(hsv, (5, 20, 15), (25, 180, 120))

    # Combine all tomato-like pixels
    tomato_mask = cv2.bitwise_or(
        cv2.bitwise_or(red, orange), cv2.bitwise_or(green, brown)
    )

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tomato_mask = cv2.morphologyEx(tomato_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    h, w = tomato_mask.shape
    tomato_ratio = float(tomato_mask.sum()) / (h * w * 255)

    # Also check: do we have a large contiguous blob? (tomato-shaped)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tomato_mask, connectivity=8)
    if num_labels > 1:
        # Exclude background (label 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_blob_ratio = float(areas.max()) / (h * w) if len(areas) > 0 else 0.0
    else:
        max_blob_ratio = 0.0

    # Decision logic
    if tomato_ratio < 0.03:
        return False, 0.0, "No tomato-like colors detected in image."
    if max_blob_ratio < 0.015:
        return False, 0.0, "No distinct tomato shape found — colors are too scattered."
    if tomato_ratio < 0.06:
        return False, round(tomato_ratio * 100, 1), "Tomato region too small in the image."

    # Confidence: how tomato-like is this image?
    tomato_conf = min(99.0, tomato_ratio * 350 + max_blob_ratio * 200)
    return True, round(tomato_conf, 1), ""


# ── Feature extraction ─────────────────────────────────────
def extract_features(image_bytes: bytes) -> np.ndarray:
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

    for sp in [rgb, hsv, lab]:
        for ch in range(3):
            c = sp[:, :, ch].astype(np.float32) / 255.0
            feats.append(float(np.mean(c)))
            feats.append(float(np.std(c)))
            feats.append(float(np.median(c)))

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

    gf = gray.astype(np.float32)
    feats.append(float((gf < 40).mean()))
    feats.append(float((gf < 20).mean()))

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

    feats.append(float(mask.mean()))
    arr = np.array(feats, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _detect_defects(feat_vec: np.ndarray) -> list[str]:
    defects = []
    dark_ratio = float(feat_vec[32])
    very_dark = float(feat_vec[33])
    coverage = float(feat_vec[37])
    glcm_contrast = float(feat_vec[27]) if len(feat_vec) > 27 else 0.0

    if very_dark > 0.04:
        defects.append("Surface Rot")
    elif dark_ratio > 0.12:
        defects.append("Dark Spots / Bruising")
    elif dark_ratio > 0.05:
        defects.append("Minor Blemishes")

    if len(feat_vec) > 2:
        g_mean = float(feat_vec[1])
        r_mean = float(feat_vec[0])
        if g_mean > r_mean * 1.15:
            defects.append("Unripe (Green Patches)")

    if glcm_contrast > 80:
        defects.append("Rough Surface Texture")

    if coverage < 0.06:
        defects.append("Low Visibility")

    if not defects:
        defects = ["No visible defects detected"]

    return defects


def predict_tomato_quality(image_bytes: bytes) -> dict[str, Any]:
    """Full prediction pipeline with tomato validation."""
    # ── 1. Decode image ──
    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ── 2. Tomato check ──
    is_tomato, tomato_conf, rejection_reason = _is_tomato(img_bgr)

    if not is_tomato:
        return {
            "quality": "Not a Tomato",
            "confidence": tomato_conf,
            "quality_score": 0,
            "detected_defects": [rejection_reason],
            "metrics": {},
            "model_version": "svm-phase1-v1",
            "rejected": True,
        }

    # ── 3. Normal prediction ──
    bundle = _get_bundle()
    model = bundle["model"]
    scaler = bundle["scaler"]
    le = bundle["label_encoder"]

    # Resize for feature extraction
    img_bgr = cv2.resize(img_bgr, (224, 224))
    img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)

    # We already have the image decoded — rebuild features directly
    # (avoid double-decode, but keep the existing extract_features for consistency)
    feat_vec = extract_features(image_bytes)

    X = scaler.transform(feat_vec.reshape(1, -1))
    probas = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probas))
    quality = str(le.inverse_transform([pred_idx])[0])
    confidence = round(float(probas[pred_idx]) * 100, 2)
    quality_score = int(round(probas[pred_idx] * 100))

    defects = _detect_defects(feat_vec)

    metrics = {
        "dark_spot_ratio": round(float(feat_vec[32]), 4),
        "very_dark_ratio": round(float(feat_vec[33]), 4),
        "coverage": round(float(feat_vec[37]), 4),
        "glcm_contrast": round(float(feat_vec[27]), 4),
    }

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
        "rejected": False,
    }
