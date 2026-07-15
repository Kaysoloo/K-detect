# 🍅 K-Detect — Smart Tomato Quality Detection

**AI-powered tomato quality classification** using computer vision and SVM machine learning.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build the dataset (one-time)
bash scripts/setup_dataset.sh

# 3. Train the model (one-time)
python3 scripts/train_model.py

# 4. Run the app
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** — upload a tomato image, get instant quality prediction.

---

## Project Structure

```
k-detect/
├── api/
│   └── index.py              # Vercel serverless entry point
├── app/
│   ├── main.py               # FastAPI application
│   ├── database.py           # SQLite prediction history
│   └── ml/
│       └── predictor.py      # SVM inference pipeline (38-dim features)
├── frontend/
│   └── index.html            # Dashboard UI (drag & drop, score ring, history)
├── models/
│   └── kdetect_svm.joblib    # Trained SVM model (committed to repo)
├── scripts/
│   ├── setup_dataset.sh      # Clone FGrade + build compact dataset
│   ├── build_compact_dataset.py  # 10-class → 3-class mapper
│   └── train_model.py        # Feature extraction + SVM/RF/KNN training
├── vercel.json               # Vercel deployment config
├── requirements.txt          # Python dependencies
└── README.md
```

---

## ML Pipeline (Phase 1)

| Step | Detail |
|------|--------|
| Dataset | [FGrade](https://github.com/skarifahmed/FGrade) (MIT) → compact 600 images |
| Classes | Good / Medium / Poor |
| Features | 38-dim: color stats (RGB/HSV/LAB), GLCM texture, shape descriptors |
| Model | SVM (RBF kernel, C=10) |
| Accuracy | 62.5% test / 62.7% CV (5-fold) |

### Feature Vector (38 dimensions)

| Group | Dims | Description |
|-------|------|-------------|
| Color | 27 | Mean, std, median per channel (RGB × HSV × LAB) |
| Texture | 5 | GLCM: contrast, dissimilarity, homogeneity, energy, correlation |
| Defect | 2 | Dark spot ratio, very dark pixel ratio |
| Shape | 3 | Circularity, aspect ratio, solidity |
| Coverage | 1 | Tomato foreground mask ratio |

---

## Deployment

### Option 1: Vercel (frontend + API on one domain) ★ Recommended

```bash
npm i -g vercel
vercel --prod
```

- `vercel.json` routes `/api/*` to the Python serverless function
- Static files from `frontend/` served at `/*`
- SQLite writes to `/tmp/` (ephemeral — history resets on cold start)
- Model (114 KB) is bundled in the deployment

### Option 2: Vercel (frontend) + Render (backend)

1. **Deploy backend to Render:**
   - Create a Web Service from your GitHub repo
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

2. **Deploy frontend to Vercel** (set `frontend/` as root directory)

3. **Access:** `https://your-app.vercel.app/?api=https://your-backend.onrender.com`

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/predict` | Upload image → quality prediction |
| `GET` | `/api/history?limit=25` | Recent predictions |

### Example Response

```json
{
  "id": 1,
  "quality": "Good",
  "confidence": 84.66,
  "quality_score": 85,
  "detected_defects": ["No major visible defects detected"],
  "metrics": {
    "dark_spot_ratio": 0.012,
    "very_dark_ratio": 0.003,
    "coverage": 0.78,
    "glcm_contrast": 45.2
  },
  "model_version": "svm-phase1-v1"
}
```

---

## Dataset Citation

Das S., Kar S., Sekh A.A. (2021) **FGrade: A Large Volume Dataset for Grading Tomato Freshness Quality.** CVIP 2020, Communications in Computer and Information Science, vol 1377. Springer, Singapore. [DOI](https://doi.org/10.1007/978-981-16-1092-9_38)

---

## Tech Stack

- **Backend:** FastAPI + Uvicorn
- **ML:** scikit-learn (SVM), OpenCV, NumPy
- **Frontend:** Vanilla HTML/CSS/JS (zero dependencies, works in sandbox)
- **Database:** SQLite
- **Deploy:** Vercel / Render / Railway

---

## License

MIT — [FGrade dataset](https://github.com/skarifahmed/FGrade) is also MIT-licensed.
