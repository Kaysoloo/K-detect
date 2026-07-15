"""
train_model.py - K-Detect Phase 1 ML Pipeline
Extracts handcrafted CV features and trains SVM/RF/KNN.
Output: models/kdetect_svm.joblib
"""

import argparse, time
from pathlib import Path
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import joblib


def extract_features(img_path, target_size=(224, 224)):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.resize(img, target_size)
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

    a = np.array(feats, dtype=np.float32)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def load_dataset(dataset_path, split="train"):
    X, y, paths = [], [], []
    split_dir = Path(dataset_path) / split
    if not split_dir.exists():
        return [], [], []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            feat = extract_features(img_path)
            if feat is not None:
                X.append(feat)
                y.append(class_dir.name)
                paths.append(str(img_path))
    return X, y, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/tomato_dataset"))
    parser.add_argument("--output", type=Path, default=Path("models/kdetect_svm.joblib"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 55)
    print("  K-Detect: Phase 1 ML Training")
    print("=" * 55)
    print(f"  Dataset: {args.dataset}")

    print("\n[1/3] Loading images & extracting features...")
    t0 = time.time()
    X_train, y_train, _ = load_dataset(args.dataset, "train")
    X_test, y_test, _ = load_dataset(args.dataset, "test")
    elapsed = time.time() - t0
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}  Dim: {len(X_train[0])}  ({elapsed:.1f}s)")

    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    print(f"  Classes: {list(le.classes_)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("\n[2/3] Training models...")
    models = {
        "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=args.seed),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=args.seed),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    }

    best_model, best_name, best_score = None, "", 0.0
    results = {}

    for name, clf in models.items():
        t1 = time.time()
        clf.fit(X_train_s, y_train_enc)
        y_pred = clf.predict(X_test_s)
        acc = accuracy_score(y_test_enc, y_pred)
        cv = cross_val_score(clf, X_train_s, y_train_enc, cv=5)
        print(f"  {name:>16s}  test={acc:.4f}  cv={cv.mean():.4f}+/-{cv.std():.4f}  ({time.time()-t1:.1f}s)")
        results[name] = {
            "test_accuracy": float(acc),
            "cv_mean": float(cv.mean()),
            "cv_std": float(cv.std()),
        }
        if acc > best_score:
            best_score, best_model, best_name = acc, clf, name

    print(f"\n  Best Model: {best_name} ({best_score:.4f})")
    y_pred_best = best_model.predict(X_test_s)
    print("\n" + classification_report(
        y_test_enc, y_pred_best,
        target_names=[str(c) for c in le.classes_],
        digits=4,
    ))
    cm = confusion_matrix(y_test_enc, y_pred_best)
    print("Confusion Matrix:")
    for i, row in enumerate(cm):
        print(f"  {str(le.classes_[i]):>8s}  {row.tolist()}")

    print(f"\n[3/3] Saving -> {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": best_model,
        "scaler": scaler,
        "label_encoder": le,
        "feature_dim": X_train.shape[1],
        "results": results,
        "model_version": "svm-phase1-v1",
    }
    joblib.dump(bundle, args.output)
    size_kb = args.output.stat().st_size / 1024
    print(f"  Size: {size_kb:.1f} KB")
    print("\nDone!")


if __name__ == "__main__":
    main()
