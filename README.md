# DualForensics: Deepfake Detection with Explainability

Dual-branch spatiotemporal attention network for deepfake detection using the FaceForensics++ dataset. Combines spatial artifact analysis (CNN + CBAM) with temporal inconsistency detection (Transformer encoder) through cross-attention fusion. Includes Grad-CAM explainability.

## Project Structure

```
project/
├── configs/
│   └── settings.py           # All hyperparameters and paths
├── src/
│   ├── data/                  # [Member 1] Data pipeline
│   │   ├── video_processor.py #   Frame extraction, sampling
│   │   ├── face_detector.py   #   MTCNN + Haar cascade
│   │   ├── preprocessing.py   #   Discovery, extraction, splitting
│   │   └── dataset.py         #   PyTorch Dataset + DataLoaders
│   ├── models/                # [Member 2] Architecture
│   │   ├── backbone.py        #   EfficientNet-B0
│   │   ├── attention.py       #   CBAM (channel + spatial)
│   │   ├── temporal.py        #   Transformer encoder
│   │   ├── fusion.py          #   Cross-attention fusion
│   │   └── dualforensics.py   #   Full model + baselines
│   ├── training/              # [Member 3] Training
│   │   ├── trainer.py         #   Training loop, evaluation
│   │   └── metrics.py         #   Accuracy, F1, AUC, etc.
│   └── explainability/        # [Member 3] XAI
│       ├── gradcam.py         #   Grad-CAM implementation
│       └── visualize.py       #   Dashboards, text explanations
├── outputs/
│   ├── plots/                 #   All saved figures
│   ├── models/                #   Checkpoints
│   └── logs/                  #   Training history, results
├── main.py                    #   Entry point
└── requirements.txt
```

## Team Contributions

| Member | Responsibility | Files |
|--------|---------------|-------|
| Member 1 | Data pipeline, face detection, dataset class | `src/data/*` |
| Member 2 | Model architecture, attention, transformer | `src/models/*` |
| Member 3 | Training, evaluation, explainability | `src/training/*`, `src/explainability/*` |

## Quick Start

```bash
pip install -r requirements.txt

# Full pipeline
python main.py --model dualforensics --epochs 15

# Baselines
python main.py --model cnn_only --epochs 10
python main.py --model cnn_lstm --epochs 10

# Skip training, just evaluate + explain
python main.py --model dualforensics --skip-train --explain 10
```

## Dataset

FaceForensics++ (c23): https://www.kaggle.com/datasets/xdxd003/ff-c23

~7000 videos: 1000 real + 6000 fake across 6 manipulation types.

---

## Web application (FastAPI + Next.js)

Full-stack demo: `backend/` serves the **exact** `DualForensics` checkpoint (`dualforensics_best.pth`); `frontend/` is a Next.js UI with upload, GradCAM grid, temporal and region charts, and demo mode.

### Layout

```
backend/   # FastAPI, model.py, inference.py, Dockerfile, weights
frontend/  # Next.js 14 (App Router), Tailwind, Recharts
```

Copy your trained file to `backend/dualforensics_best.pth` (or set `MODEL_PATH`).

### Backend (local)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- `GET /api/health` — `{"status":"ok","model_loaded":...}`
- `POST /api/predict` — multipart field `file` (video), max ~50 MB
- `GET /api/demo` — runs the pipeline on a small synthetic clip (no upload)

Environment:

| Variable | Purpose |
|----------|---------|
| `MODEL_PATH` | Path to `dualforensics_best.pth` (default: `./dualforensics_best.pth` under `backend/`) |
| `CORS_ORIGINS` | Comma-separated allowed origins (default includes `http://localhost:3000`) |

### Frontend (local)

```bash
cd frontend
cp .env.example .env.local   # set NEXT_PUBLIC_API_URL to your API origin
npm install
npm run dev
```

Open http://localhost:3000

### Deploy

- **Backend (Railway / Render):** build from `backend/` using the included `Dockerfile`. Set `PORT` is automatic on Railway; keep `CMD` as provided. Upload `dualforensics_best.pth` with the service (or volume mount). Set `CORS_ORIGINS` to your Vercel URL, e.g. `https://your-app.vercel.app`.
- **Frontend (Vercel):** root directory `frontend`, build `npm run build`, output `.next` handled by Vercel. Set `NEXT_PUBLIC_API_URL` to the public Railway/Render API URL (no trailing slash required).

### Model metrics (FaceForensics++ c23)

Accuracy 85.24%, Precision 99.73%, Recall 83.00%, F1 0.9060, AUC 0.9787.
