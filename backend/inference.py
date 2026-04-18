import base64
import io
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import DualForensics

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

REGIONS = {
    "forehead": (0, 56, 45, 179),
    "eyes": (56, 100, 25, 199),
    "nose": (100, 145, 67, 157),
    "mouth": (145, 190, 45, 179),
    "jaw/chin": (190, 224, 25, 199),
}

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
DISPLAY_FRAMES = 8


def extract_frames(video_path: str, max_frames: int = 300):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def sample_uniform(frames, n=16):
    total = len(frames)
    if total == 0:
        return []
    if total <= n:
        return list(frames) + [frames[-1]] * (n - total)
    idx = np.linspace(0, total - 1, n, dtype=int)
    return [frames[i] for i in idx]


def detect_face(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = haar.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        margin = 0.1
        mx, my = int(w * margin), int(h * margin)
        crop = rgb[max(0, y-my):min(rgb.shape[0], y+h+my),
                   max(0, x-mx):min(rgb.shape[1], x+w+mx)]
        return Image.fromarray(crop).resize((224, 224), Image.BILINEAR)
    h, w = rgb.shape[:2]
    s = min(h, w)
    cy, cx = h // 2, w // 2
    crop = rgb[cy-s//2:cy+s//2, cx-s//2:cx+s//2]
    return Image.fromarray(crop).resize((224, 224), Image.BILINEAR)


def region_analysis(hmap):
    return sorted([(n, float(hmap[y1:y2, x1:x2].mean()))
                   for n, (y1, y2, x1, x2) in REGIONS.items()],
                  key=lambda x: -x[1])


def _api_explanation_text(pred: float, hmap: np.ndarray, temporal_importance: np.ndarray | None) -> str:
    if pred <= 0.5:
        return "No significant manipulation artifacts detected."
    parts = []
    regions = region_analysis(hmap)
    top = [r for r in regions if r[1] > 0.2][:3]
    if top:
        parts.append("Artifacts detected in " + ", ".join(r[0] for r in top) + ".")
    if temporal_importance is not None and len(temporal_importance):
        thresh = float(np.mean(temporal_importance) + np.std(temporal_importance))
        anomalous = np.where(temporal_importance > thresh)[0]
        if len(anomalous) > 0:
            parts.append(f"Temporal anomalies at frames {anomalous.tolist()}.")
    if not parts:
        parts.append("Elevated deepfake score; review heatmaps for localized cues.")
    return " ".join(parts)


def region_scores_dict(hmap: np.ndarray) -> dict[str, float]:
    return {n: float(hmap[y1:y2, x1:x2].mean()) for n, (y1, y2, x1, x2) in REGIONS.items()}


def temporal_importance_from_attn(t_attn: torch.Tensor) -> list[float]:
    """Per-frame scores from self-attention mass (B, N, N)."""
    a = t_attn[0].sum(dim=0).detach().cpu().numpy().astype(np.float64)
    if a.max() > 1e-8:
        a = a / a.max()
    return [float(x) for x in a.tolist()]


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.grads = None
        self.acts = None
        model.backbone_feat[-1].register_forward_hook(
            lambda m, i, o: setattr(self, 'acts', o.detach()))
        model.backbone_feat[-1].register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'grads', go[0].detach()))

    def run(self, video_tensor):
        self.model.eval()
        inp = video_tensor.clone().detach().requires_grad_(True)
        lg, exp = self.model(inp)
        pred = torch.sigmoid(lg[0, 0]).item()
        self.model.zero_grad()
        lg[0, 0].backward(retain_graph=True)
        N = video_tensor.shape[1]
        hmaps = []
        if self.grads is not None and self.acts is not None:
            for i in range(min(N, self.grads.shape[0])):
                w = self.grads[i].mean(dim=[1, 2])
                h = (w[:, None, None] * self.acts[i]).sum(0)
                h = F.relu(h)
                h = h - h.min()
                h = h / (h.max() + 1e-8)
                hmaps.append(cv2.resize(h.cpu().numpy(), (224, 224)))
        else:
            hmaps = [np.zeros((224, 224))] * N
        return hmaps, pred, exp


def _pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _heatmap_overlay(rgb_uint8: np.ndarray, hmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    hmap_u8 = (np.clip(hmap, 0, 1) * 255).astype(np.uint8)
    color = cv2.applyColorMap(hmap_u8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    blended = (alpha * color + (1 - alpha) * rgb_uint8.astype(np.float32)).astype(np.uint8)
    return Image.fromarray(blended)


def _build_gradcam_dashboard(face_rgbs: list[np.ndarray], hmaps: list[np.ndarray], cols: int = 4) -> str:
    rows = int(np.ceil(len(face_rgbs) / cols))
    cell = 224
    canvas = np.zeros((rows * cell, cols * cell * 2, 3), dtype=np.uint8)
    for idx, (fr, hm) in enumerate(zip(face_rgbs, hmaps)):
        r, c = divmod(idx, cols)
        y0, x0 = r * cell, c * cell * 2
        face_img = Image.fromarray(fr).resize((cell, cell), Image.BILINEAR)
        canvas[y0:y0 + cell, x0:x0 + cell] = np.array(face_img)
        ov = _heatmap_overlay(np.array(face_img), hm)
        canvas[y0:y0 + cell, x0 + cell:x0 + 2 * cell] = np.array(ov)
    pil = Image.fromarray(canvas)
    return _pil_to_b64_png(pil)


def _display_indices(n_total: int, k: int) -> list[int]:
    if n_total <= k:
        return list(range(n_total))
    return [int(x) for x in np.linspace(0, n_total - 1, k, dtype=int)]


def load_model(weights_path: Path, device: torch.device) -> DualForensics:
    model = DualForensics()
    ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def ensure_demo_video(path: Path) -> None:
    if path.exists() and path.stat().st_size > 2000:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 12.0, (640, 480))
    for i in range(48):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (40 + (i * 3) % 80, 50, 60 + i % 40)
        cv2.rectangle(frame, (180 + i * 2, 140), (460, 380), (200, 200, 220), -1)
        cv2.putText(frame, "DUALFORENSICS DEMO", (120, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        out.write(frame)
    out.release()


def run_inference(
    model: DualForensics,
    device: torch.device,
    video_path: str,
    gradcam: GradCAM,
) -> dict[str, Any]:
    frames = extract_frames(video_path)
    sampled = sample_uniform(frames, n=16)
    face_pils = [detect_face(f) for f in sampled]
    tensors = torch.stack([val_tf(im) for im in face_pils], dim=0).unsqueeze(0).to(device)

    hmaps, pred, exp = gradcam.run(tensors)
    t_attn = exp["temporal_attn"]
    temporal_importance = temporal_importance_from_attn(t_attn)

    hmap_mean = np.mean(np.stack(hmaps, axis=0), axis=0)
    regions = region_scores_dict(hmap_mean)

    label = "DEEPFAKE" if pred > 0.5 else "REAL"
    confidence = float(pred if pred > 0.5 else 1.0 - pred)

    face_rgbs = [np.array(im.convert("RGB")) for im in face_pils]
    idxs = _display_indices(len(face_rgbs), DISPLAY_FRAMES)
    face_b64 = [_pil_to_b64_png(Image.fromarray(face_rgbs[i])) for i in idxs]
    heat_b64 = [
        _pil_to_b64_png(_heatmap_overlay(face_rgbs[i], hmaps[i]))
        for i in idxs
    ]
    dashboard_b64 = _build_gradcam_dashboard(face_rgbs, hmaps, cols=4)

    expl_api = _api_explanation_text(pred, hmap_mean, np.array(temporal_importance))

    return {
        "prediction": label,
        "confidence": confidence,
        "explanation": expl_api,
        "gradcam_image": dashboard_b64,
        "face_frames": face_b64,
        "heatmap_frames": heat_b64,
        "temporal_importance": temporal_importance,
        "region_scores": regions,
    }
