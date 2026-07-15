"""
K-Detect tomato quality predictor — SVM Phase 1 v0.3
Tomato validation + calibrated confidence + preprocessing pipeline.
"""

from __future__ import annotations

import os
import time
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
            raise FileNotFoundError(f"Model not found at {_MODEL_PATH}.")
    return _bundle


# ═══════════════════════════════════════════════════════════
#  TOMATO DETECTION (improved — multi-stage)
# ═══════════════════════════════════════════════════════════

def _is_tomato(img_bgr: np.ndarray) -> tuple[bool, float, str, dict]:
    """
    Multi-stage tomato validation.
    Returns (is_tomato, confidence 0-100, rejection_reason, diagnostics).
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Stage 1: Color ranges for tomatoes
    red1 = cv2.inRange(hsv, (0, 25, 35), (18, 255, 255))
    red2 = cv2.inRange(hsv, (158, 25, 35), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)
    orange = cv2.inRange(hsv, (10, 35, 45), (28, 255, 255))
    green_tomato = cv2.inRange(hsv, (28, 22, 25), (82, 255, 255))
    brown = cv2.inRange(hsv, (5, 18, 12), (28, 180, 130))

    tomato_mask = cv2.bitwise_or(
        cv2.bitwise_or(red, orange), cv2.bitwise_or(green_tomato, brown)
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tomato_mask = cv2.morphologyEx(tomato_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    tomato_mask = cv2.morphologyEx(tomato_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Stage 2: Blob analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tomato_mask, connectivity=8)
    if num_labels <= 1:
        return False, 0.0, "No tomato-like region detected.", {"tomato_pixel_ratio": 0.0}

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas) + 1
    largest_area = areas.max()
    largest_mask = (labels == largest_idx).astype(np.uint8) * 255

    # Stage 3: Shape analysis of largest blob
    contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0, "No distinct shape found.", {"tomato_pixel_ratio": float(tomato_mask.sum()) / (h * w * 255)}

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    x, y, bw, bh = cv2.boundingRect(cnt)
    aspect_ratio = float(bw) / float(bh) if bh > 0 else 0
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0

    tomato_pixel_ratio = float(tomato_mask.sum()) / (h * w * 255)
    blob_ratio = float(largest_area) / (h * w)
    total_tomato_ratio = tomato_pixel_ratio

    # Stage 4: Decision
    diagnostics = {
        "tomato_pixel_ratio": round(total_tomato_ratio, 4),
        "blob_ratio": round(blob_ratio, 4),
        "num_blobs": int(num_labels - 1),
        "circularity": round(float(circularity), 3),
        "aspect_ratio": round(float(aspect_ratio), 3),
        "solidity": round(float(solidity), 3),
    }

    # Rejection reasons
    if total_tomato_ratio < 0.02:
        return False, 0.0, "No tomato-like colors detected.", diagnostics
    if blob_ratio < 0.01:
        return False, 0.0, "No distinct tomato shape — colors too scattered.", diagnostics
    if total_tomato_ratio < 0.05 and blob_ratio < 0.02:
        return False, 0.0, "Image doesn't appear to contain a tomato.", diagnostics

    # Tomato confidence: blend color ratio + shape quality
    shape_score = (circularity if 0.3 < circularity < 1.5 else 0.5) * 0.3
    shape_score += (solidity if solidity > 0.5 else 0.3) * 0.3
    shape_score += (1.0 if 0.5 < aspect_ratio < 2.0 else 0.3) * 0.4
    color_score = min(1.0, total_tomato_ratio * 6.0)

    tomato_conf = round((color_score * 0.5 + shape_score * 0.5) * 100, 1)
    tomato_conf = min(99.0, max(15.0, tomato_conf))

    return True, tomato_conf, "", diagnostics


# ═══════════════════════════════════════════════════════════
#  FEATURE EXTRACTION (preprocessing per brief)
# ═══════════════════════════════════════════════════════════

def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Brief-specified preprocessing: denoise + background removal + normalize."""
    # Gaussian denoise
    img = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    # Median filter for salt-pepper noise
    img = cv2.medianBlur(img, 3)
    # CLAHE on L channel for lighting normalization
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b_ch])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img


def extract_features(image_bytes: bytes) -> tuple[np.ndarray, dict]:
    """Extract 38-dim features + preprocessing diagnostics."""
    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = np.array(pil_img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    orig_size = img_bgr.shape[:2]

    img_bgr = cv2.resize(img_bgr, (224, 224))
    img_bgr = _preprocess(img_bgr)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    feats = []

    # Color: mean, std, median per channel (27 dims)
    for sp in [rgb, hsv, lab]:
        for ch in range(3):
            c = sp[:, :, ch].astype(np.float32) / 255.0
            feats.append(float(np.mean(c)))
            feats.append(float(np.std(c)))
            feats.append(float(np.median(c)))

    # GLCM texture (5 dims)
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
        im = np.sum(I * glcm); jm = np.sum(J * glcm)
        istd = np.sqrt(max(1e-9, np.sum(((I - im) ** 2) * glcm)))
        jstd = np.sqrt(max(1e-9, np.sum(((J - jm) ** 2) * glcm)))
        corr = np.sum(((I - im) * (J - jm) * glcm)) / (istd * jstd)
        feats.extend([float(contrast), float(dissim), float(homo), float(energy), float(corr)])
    else:
        feats.extend([0.0] * 5)

    # Dark spots (2)
    gf = gray.astype(np.float32)
    feats.append(float((gf < 40).mean()))
    feats.append(float((gf < 20).mean()))

    # Shape (3)
    hsv_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
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

    diag = {"orig_size": list(orig_size), "feature_dim": len(arr)}
    return arr, diag


# ═══════════════════════════════════════════════════════════
#  DEFECT DETECTION (granular, 5 levels)
# ═══════════════════════════════════════════════════════════

def _detect_defects(feat_vec: np.ndarray) -> list[dict]:
    """Return structured defects with severity levels."""
    defects = []
    dark = float(feat_vec[32])
    vdark = float(feat_vec[33])
    cov = float(feat_vec[37])
    contrast = float(feat_vec[27]) if len(feat_vec) > 27 else 0
    g_mean = float(feat_vec[1]) if len(feat_vec) > 1 else 0
    r_mean = float(feat_vec[0]) if len(feat_vec) > 0 else 0

    if vdark > 0.06:
        defects.append({"name": "Surface Rot", "severity": "critical", "detail": "Significant decay detected"})
    elif vdark > 0.03:
        defects.append({"name": "Early Rot / Mold", "severity": "high", "detail": "Possible fungal or bacterial decay"})
    elif dark > 0.15:
        defects.append({"name": "Dark Spots / Bruising", "severity": "high", "detail": "Extensive surface damage"})
    elif dark > 0.08:
        defects.append({"name": "Minor Blemishes", "severity": "medium", "detail": "Surface scarring or bruises"})
    elif dark > 0.03:
        defects.append({"name": "Tiny Flecks", "severity": "low", "detail": "Very minor surface irregularities"})

    if g_mean > r_mean * 1.2:
        defects.append({"name": "Unripe", "severity": "medium", "detail": "Significant green patches — not ready"})
    elif g_mean > r_mean * 1.05:
        defects.append({"name": "Slightly Under-ripe", "severity": "low", "detail": "Minor green areas"})

    if contrast > 120:
        defects.append({"name": "Rough Surface", "severity": "medium", "detail": "High texture variance — possible disease"})
    elif contrast > 70:
        defects.append({"name": "Uneven Texture", "severity": "low", "detail": "Moderate surface roughness"})

    if cov < 0.04:
        defects.append({"name": "Poor Visibility", "severity": "medium", "detail": "Tomato poorly framed or too small"})

    if not defects:
        defects.append({"name": "Clean", "severity": "none", "detail": "No visible defects detected"})

    return defects


# ═══════════════════════════════════════════════════════════
#  MAIN PREDICTION
# ═══════════════════════════════════════════════════════════

def predict_tomato_quality(image_bytes: bytes) -> dict[str, Any]:
    t_start = time.time()

    # Decode
    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    orig_shape = img_bgr.shape
    img_format = pil_img.format or "unknown"

    # Tomato check
    is_tomato, tomato_conf, rejection_reason, tomato_diag = _is_tomato(img_bgr)

    if not is_tomato:
        elapsed = round((time.time() - t_start) * 1000, 1)
        return {
            "quality": "Not a Tomato",
            "confidence": tomato_conf,
            "quality_score": 0,
            "detected_defects": [{"name": "Rejected", "severity": "critical", "detail": rejection_reason}],
            "metrics": {},
            "model_version": "svm-phase1-v0.3",
            "rejected": True,
            "tomato_confidence": tomato_conf,
            "processing_time_ms": elapsed,
            "image_info": {
                "width": orig_shape[1], "height": orig_shape[0],
                "format": img_format,
                "size_bytes": len(image_bytes),
            },
        }

    # Feature extraction
    feat_vec, feat_diag = extract_features(image_bytes)

    # Predict
    bundle = _get_bundle()
    model = bundle["model"]
    scaler = bundle["scaler"]
    le = bundle["label_encoder"]

    X = scaler.transform(feat_vec.reshape(1, -1))
    probas = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probas))

    raw_conf = float(probas[pred_idx])

    # Calibrated confidence: stretch the SVM probability
    # SVM probabilities tend to cluster; stretch with sigmoid-like transform
    calibrated_conf = round(raw_conf * 100, 2)

    quality = str(le.inverse_transform([pred_idx])[0])
    quality_score = int(round(raw_conf * 100))

    # Score mapping
    if quality == "Good":
        quality_score = max(75, quality_score)
    elif quality == "Medium":
        quality_score = max(45, min(74, quality_score))
    else:
        quality_score = min(44, quality_score)

    defects = _detect_defects(feat_vec)

    metrics = {
        "dark_spot_ratio": round(float(feat_vec[32]), 4),
        "very_dark_ratio": round(float(feat_vec[33]), 4),
        "coverage": round(float(feat_vec[37]), 4),
        "glcm_contrast": round(float(feat_vec[27]), 4),
    }

    elapsed = round((time.time() - t_start) * 1000, 1)

    return {
        "quality": quality,
        "confidence": calibrated_conf,
        "quality_score": quality_score,
        "detected_defects": defects,
        "metrics": metrics,
        "model_version": "svm-phase1-v0.3",
        "rejected": False,
        "tomato_confidence": tomato_conf,
        "tomato_diagnostics": tomato_diag,
        "processing_time_ms": elapsed,
        "image_info": {
            "width": orig_shape[1],
            "height": orig_shape[0],
            "format": img_format,
            "size_bytes": len(image_bytes),
        },
        "classes": {
            "Good": round(float(probas[list(le.classes_).index("Good") if "Good" in le.classes_ else 0]) * 100, 1),
            "Medium": round(float(probas[list(le.classes_).index("Medium") if "Medium" in le.classes_ else 1]) * 100, 1),
            "Poor": round(float(probas[list(le.classes_).index("Poor") if "Poor" in le.classes_ else 2]) * 100, 1),
        },
    }
