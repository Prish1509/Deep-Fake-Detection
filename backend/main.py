import os
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference import (
    MAX_UPLOAD_BYTES,
    GradCAM,
    ensure_demo_video,
    load_model,
    run_inference,
)

BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = BACKEND_DIR / "dualforensics_best.pth"
DEMO_VIDEO_PATH = BACKEND_DIR / "demo_video.mp4"

WEIGHTS_PATH = Path(os.environ.get("MODEL_PATH", str(DEFAULT_WEIGHTS)))
FRONTEND_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")

print(f"[startup] MODEL_PATH env: {os.environ.get('MODEL_PATH')}")
print(f"[startup] Resolved weights path: {WEIGHTS_PATH}")
print(f"[startup] Weights exists: {WEIGHTS_PATH.is_file()}")
if WEIGHTS_PATH.exists():
    print(f"[startup] Weights size bytes: {WEIGHTS_PATH.stat().st_size}")

device = torch.device("cpu")
model = None
gradcam: GradCAM | None = None
model_load_error: str | None = None

try:
    if WEIGHTS_PATH.is_file():
        model = load_model(WEIGHTS_PATH, device)
        gradcam = GradCAM(model)
    else:
        model_load_error = f"Weights not found at {WEIGHTS_PATH}"
except Exception as e:
    model_load_error = str(e)
    model = None
    gradcam = None

app = FastAPI(title="DualForensics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in FRONTEND_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_model():
    if model is None or gradcam is None:
        raise HTTPException(
            status_code=503,
            detail=model_load_error or "Model not loaded",
        )


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None and gradcam is not None,
        "model_path": str(WEIGHTS_PATH),
        "model_exists": WEIGHTS_PATH.is_file(),
        "model_error": model_load_error,
    }


@app.get("/api/demo")
def demo():
    _require_model()
    ensure_demo_video(DEMO_VIDEO_PATH)
    return run_inference(model, device, str(DEMO_VIDEO_PATH), gradcam)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    _require_model()
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing video file")

    suffix = Path(file.filename).suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".webm", ".avi", ".mkv"}:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)",
        )
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return run_inference(model, device, tmp_path, gradcam)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
