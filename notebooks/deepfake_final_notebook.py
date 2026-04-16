


# DualForensics: Deepfake Detection with Explainability
# Course: Introduction to Deep Learning
# Dataset: FaceForensics++ (c23)
#
# Kaggle notebook - copy each SECTION into a separate cell.
# Enable GPU: Session Options > Accelerator > GPU T4 x2


# ================================================================
# SECTION 1 - Dependencies
# ================================================================

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "facenet-pytorch"])
print("Dependencies ready.")


# ================================================================
# SECTION 2 - Imports and Configuration
# ================================================================

import os, cv2, glob, json, math, time, random, warnings
import numpy as np
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

DATASET_ROOT = "/kaggle/input/ff-c23"
OUTPUT_DIR = "/kaggle/working"
FACES_DIR = "/kaggle/working/faces"
PLOTS_DIR = "/kaggle/working/plots"
CKPT_DIR = "/kaggle/working/checkpoints"

REAL_FOLDER = "original"
FAKE_FOLDERS = [
    "DeepFakeDetection", "Deepfakes", "Face2Face",
    "FaceShifter", "FaceSwap", "NeuralTextures",
]

NUM_FRAMES = 16
FACE_SIZE = 224
FACE_MARGIN = 0.1
MAX_FRAMES = 300

FEATURE_DIM = 1280
SPATIAL_ATT_R = 16
T_HEADS = 4
T_LAYERS = 2
T_FF = 512
FUSION_DIM = 256
DROPOUT = 0.3
T_DROPOUT = 0.1

BATCH_SIZE = 4
NUM_WORKERS = 2
LR = 1e-4
NUM_EPOCHS = 15
PATIENCE = 5
REAL_W = 6.0
FAKE_W = 1.0
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for d in [FACES_DIR, PLOTS_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ================================================================
# SECTION 3 - Dataset Discovery
# ================================================================

def discover_videos():
    videos = []
    exts = (".mp4", ".avi", ".mov", ".mkv")
    real_dir = os.path.join(DATASET_ROOT, REAL_FOLDER)
    if os.path.isdir(real_dir):
        for root, _, files in os.walk(real_dir):
            for f in sorted(files):
                if f.lower().endswith(exts):
                    videos.append({"path": os.path.join(root, f), "label": 0, "type": "original"})
    for folder in FAKE_FOLDERS:
        fdir = os.path.join(DATASET_ROOT, folder)
        if os.path.isdir(fdir):
            for root, _, files in os.walk(fdir):
                for f in sorted(files):
                    if f.lower().endswith(exts):
                        videos.append({"path": os.path.join(root, f), "label": 1, "type": folder})
    return videos

all_videos = discover_videos()
real_n = sum(1 for v in all_videos if v["label"] == 0)
fake_n = sum(1 for v in all_videos if v["label"] == 1)
print(f"Dataset: {len(all_videos)} videos (real={real_n}, fake={fake_n})")


# ================================================================
# SECTION 4 - Frame Extraction
# ================================================================

def extract_frames(path, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def sample_uniform(frames, n=NUM_FRAMES):
    total = len(frames)
    if total == 0:
        return []
    if total <= n:
        return list(frames) + [frames[-1]] * (n - total)
    idx = np.linspace(0, total - 1, n, dtype=int)
    return [frames[i] for i in idx]


# ================================================================
# SECTION 5 - Face Detection
# ================================================================

class FaceDetector:
    def __init__(self):
        self.mtcnn = None
        try:
            from facenet_pytorch import MTCNN
            self.mtcnn = MTCNN(image_size=FACE_SIZE, margin=int(FACE_SIZE * FACE_MARGIN),
                               keep_all=False, device=DEVICE, post_process=False)
        except Exception:
            pass
        self.haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.mtcnn:
            try:
                t = self.mtcnn(Image.fromarray(rgb))
                if t is not None:
                    return Image.fromarray(t.permute(1,2,0).byte().cpu().numpy())
            except Exception:
                pass
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self.haar.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            mx, my = int(w*FACE_MARGIN), int(h*FACE_MARGIN)
            crop = rgb[max(0,y-my):min(rgb.shape[0],y+h+my), max(0,x-mx):min(rgb.shape[1],x+w+mx)]
            return Image.fromarray(crop).resize((FACE_SIZE, FACE_SIZE), Image.BILINEAR)
        h, w = rgb.shape[:2]
        s = min(h, w)
        cy, cx = h//2, w//2
        crop = rgb[cy-s//2:cy+s//2, cx-s//2:cx+s//2]
        return Image.fromarray(crop).resize((FACE_SIZE, FACE_SIZE), Image.BILINEAR)

face_detector = FaceDetector()
print("Face detector initialized.")


# ================================================================
# SECTION 6 - Preprocess Videos
# ================================================================

def preprocess_all(video_list, detector):
    processed = []
    t0 = time.time()
    for i, v in enumerate(video_list):
        if (i+1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(video_list)}] {(i+1)/elapsed:.1f} vid/s")
        vname = Path(v["path"]).stem
        save_dir = os.path.join(FACES_DIR, v["type"], vname)
        if os.path.isdir(save_dir) and len(glob.glob(os.path.join(save_dir, "*.jpg"))) >= NUM_FRAMES:
            processed.append({**v, "faces_dir": save_dir})
            continue
        frames = extract_frames(v["path"])
        if not frames:
            continue
        sampled = sample_uniform(frames)
        os.makedirs(save_dir, exist_ok=True)
        for j, fr in enumerate(sampled):
            face = detector.detect(fr)
            face.save(os.path.join(save_dir, f"frame_{j:02d}.jpg"))
        processed.append({**v, "faces_dir": save_dir})
    print(f"Preprocessed {len(processed)} videos in {(time.time()-t0)/60:.1f}m")
    return processed

meta_path = os.path.join(FACES_DIR, "metadata.json")
if os.path.exists(meta_path):
    with open(meta_path) as f:
        processed_videos = json.load(f)
    print(f"Loaded {len(processed_videos)} preprocessed videos")
else:
    processed_videos = preprocess_all(all_videos, face_detector)
    with open(meta_path, "w") as f:
        json.dump(processed_videos, f)


# ================================================================
# SECTION 7 - Dataset and DataLoaders
# ================================================================

train_tf = transforms.Compose([
    transforms.Resize((FACE_SIZE, FACE_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
    transforms.GaussianBlur(3, (0.1, 1.0)),
    transforms.ToTensor(),
    transforms.RandomErasing(0.3, (0.02, 0.15)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((FACE_SIZE, FACE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
inv_norm = transforms.Normalize(
    [-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])

class VideoDataset(Dataset):
    def __init__(self, vids, transform=None):
        self.videos = vids
        self.tf = transform or val_tf
    def __len__(self):
        return len(self.videos)
    def __getitem__(self, idx):
        info = self.videos[idx]
        files = sorted(glob.glob(os.path.join(info.get("faces_dir",""), "frame_*.jpg")))
        faces = []
        for fp in files[:NUM_FRAMES]:
            try: faces.append(Image.open(fp).convert("RGB"))
            except: faces.append(Image.new("RGB", (FACE_SIZE, FACE_SIZE)))
        while len(faces) < NUM_FRAMES:
            faces.append(faces[-1] if faces else Image.new("RGB", (FACE_SIZE, FACE_SIZE)))
        frames = torch.stack([self.tf(f) for f in faces])
        return {"frames": frames, "label": torch.tensor(info["label"], dtype=torch.float32),
                "type": info["type"], "path": info["path"]}

strat = [f"{v['label']}_{v['type']}" for v in processed_videos]
train_v, temp_v, _, temp_k = train_test_split(processed_videos, strat, test_size=0.30,
                                               stratify=strat, random_state=SEED)
val_v, test_v = train_test_split(temp_v, test_size=0.5, stratify=temp_k, random_state=SEED)

train_ds = VideoDataset(train_v, train_tf)
val_ds = VideoDataset(val_v, val_tf)
test_ds = VideoDataset(test_v, val_tf)

labels = [v["label"] for v in train_v]
cnts = np.bincount(labels)
sw = torch.DoubleTensor([1.0/cnts[l] for l in labels])
sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS,
                          pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train: {len(train_v)} vids, Val: {len(val_v)}, Test: {len(test_v)}")
print(f"Batches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
# ================================================================
# SECTION 8 - Model Architecture
# ================================================================

class ChannelAttention(nn.Module):
    def __init__(self, ch, r=SPATIAL_ATT_R):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Linear(ch, ch//r, bias=False), nn.ReLU(True),
                                nn.Linear(ch//r, ch, bias=False))
        self.sig = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        a = self.fc(self.ap(x).view(b,c)) + self.fc(self.mp(x).view(b,c))
        return x * self.sig(a).view(b,c,1,1)

class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k//2, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        att = self.sig(self.conv(torch.cat([avg, mx], 1)))
        return x * att, att

class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = self.ca(x)
        x, sm = self.sa(x)
        return x, sm

class PosEnc(nn.Module):
    def __init__(self, d, maxlen=100, drop=0.1):
        super().__init__()
        self.drop = nn.Dropout(drop)
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(0, maxlen, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])

class TemporalTransformer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.pe = PosEnc(d, dropout=T_DROPOUT)
        layer = nn.TransformerEncoderLayer(d, T_HEADS, T_FF, T_DROPOUT,
                                           batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, T_LAYERS)
    def forward(self, x):
        xp = self.pe(x)
        out = self.enc(xp)
        with torch.no_grad():
            dk = x.size(-1) // T_HEADS
            s = torch.bmm(xp, xp.transpose(1,2)) / math.sqrt(dk)
            attn = F.softmax(s, dim=-1)
        return out, attn

class CrossFusion(nn.Module):
    def __init__(self, sd, td, fd=FUSION_DIM):
        super().__init__()
        self.sp = nn.Linear(sd, fd)
        self.tp = nn.Linear(td, fd)
        self.ca = nn.MultiheadAttention(fd, 4, 0.1, batch_first=True)
        self.ln = nn.LayerNorm(fd)
        self.out_dim = fd
    def forward(self, sf, tf):
        s, t = self.sp(sf), self.tp(tf)
        ao, aw = self.ca(s, t, t)
        return self.ln(ao + s).mean(1), aw

class DualForensics(nn.Module):
    def __init__(self):
        super().__init__()
        D = FEATURE_DIM
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone_feat = base.features
        self.backbone_pool = nn.AdaptiveAvgPool2d(1)
        self.cbam = CBAM(D)
        self.s_pool = nn.AdaptiveAvgPool2d(1)
        self.temporal = TemporalTransformer(D)
        self.fusion = CrossFusion(D, D)
        self.head = nn.Sequential(nn.Linear(self.fusion.out_dim, 256), nn.ReLU(True),
                                  nn.Dropout(DROPOUT), nn.Linear(256, 1))
    def forward(self, x):
        B, N, C, H, W = x.shape
        flat = x.view(B*N, C, H, W)
        fmaps = self.backbone_feat(flat)
        fvecs = self.backbone_pool(fmaps).flatten(1)
        D = fvecs.size(1)
        h, w = fmaps.shape[2], fmaps.shape[3]
        att_maps, s_maps = self.cbam(fmaps)
        sf = self.s_pool(att_maps).flatten(1).view(B, N, D)
        tf_in = fvecs.view(B, N, D)
        tf_out, t_attn = self.temporal(tf_in)
        fused, c_attn = self.fusion(sf, tf_out)
        logits = self.head(fused)
        return logits, {"spatial_maps": s_maps.view(B,N,1,h,w).detach(),
                        "temporal_attn": t_attn.detach(), "cross_attn": c_attn.detach()}

class CNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.feat = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Linear(FEATURE_DIM, 256), nn.ReLU(True),
                                  nn.Dropout(DROPOUT), nn.Linear(256, 1))
    def forward(self, x):
        B, N, C, H, W = x.shape
        v = self.pool(self.feat(x.view(B*N,C,H,W))).flatten(1)
        return self.head(v).view(B,N,1).mean(1), {}

class CNNLSTMBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.feat = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(FEATURE_DIM, 256, 2, batch_first=True, dropout=0.3, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True),
                                  nn.Dropout(DROPOUT), nn.Linear(256, 1))
    def forward(self, x):
        B, N, C, H, W = x.shape
        v = self.pool(self.feat(x.view(B*N,C,H,W))).flatten(1).view(B,N,-1)
        o, _ = self.lstm(v)
        return self.head(o[:,-1,:]), {}

def build_model(name="dualforensics"):
    m = {"dualforensics": DualForensics, "cnn_only": CNNBaseline, "cnn_lstm": CNNLSTMBaseline}
    model = m[name]().to(DEVICE)
    p = sum(x.numel() for x in model.parameters())
    print(f"Model '{name}': {p:,} params")
    return model

model = build_model("dualforensics")

with torch.no_grad():
    dummy = torch.randn(2, NUM_FRAMES, 3, FACE_SIZE, FACE_SIZE).to(DEVICE)
    out, _ = model(dummy)
    print(f"Forward pass OK: {out.shape}")
del dummy; torch.cuda.empty_cache()


# ================================================================
# SECTION 9 - Training
# ================================================================

def w_bce(logits, targets):
    p = torch.sigmoid(logits)
    w = torch.where(targets == 0, torch.tensor(REAL_W, device=DEVICE),
                    torch.tensor(FAKE_W, device=DEVICE))
    return (w * -(targets*torch.log(p+1e-8) + (1-targets)*torch.log(1-p+1e-8))).mean()

optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, NUM_EPOCHS, eta_min=1e-7)
history = {"tl":[], "ta":[], "vl":[], "va":[], "vf":[], "vu":[]}
best_auc = 0.0

print(f"\nTraining DualForensics for {NUM_EPOCHS} epochs...")

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    model.train()
    rl, tp, tl = 0.0, [], []
    for batch in train_loader:
        fr = batch["frames"].to(DEVICE)
        lb = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        lg, _ = model(fr)
        lg = lg.squeeze(1)
        loss = w_bce(lg, lb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        rl += loss.item()
        tp.extend((torch.sigmoid(lg)>0.5).float().cpu().numpy())
        tl.extend(lb.cpu().numpy())
    scheduler.step()
    tr_loss = rl / len(train_loader)
    tr_acc = accuracy_score(tl, tp)

    model.eval()
    vl_, vp_, vpb_, vlb_ = 0.0, [], [], []
    with torch.no_grad():
        for batch in val_loader:
            fr = batch["frames"].to(DEVICE)
            lb = batch["label"].to(DEVICE)
            lg, _ = model(fr); lg = lg.squeeze(1)
            vl_ += w_bce(lg, lb).item()
            pb = torch.sigmoid(lg)
            vp_.extend((pb>0.5).float().cpu().numpy())
            vpb_.extend(pb.cpu().numpy())
            vlb_.extend(lb.cpu().numpy())
    v_loss = vl_ / max(len(val_loader),1)
    v_acc = accuracy_score(vlb_, vp_)
    v_f1 = f1_score(vlb_, vp_, zero_division=0)
    v_auc = roc_auc_score(vlb_, vpb_) if len(set(vlb_))>1 else 0.5

    history["tl"].append(tr_loss); history["ta"].append(tr_acc)
    history["vl"].append(v_loss); history["va"].append(v_acc)
    history["vf"].append(v_f1); history["vu"].append(v_auc)

    dt = time.time()-t0
    print(f"Ep {epoch}/{NUM_EPOCHS} [{dt:.0f}s] tr_acc={tr_acc:.4f} "
          f"v_acc={v_acc:.4f} v_f1={v_f1:.4f} v_auc={v_auc:.4f}")

    if v_auc > best_auc:
        best_auc = v_auc
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best.pth"))
        print(f"  * Best (AUC={best_auc:.4f})")
    elif epoch - history["vu"].index(max(history["vu"])) - 1 >= PATIENCE:
        print(f"  Early stop at epoch {epoch}")
        break

print(f"Training done. Best AUC: {best_auc:.4f}")


# ================================================================
# SECTION 10 - Training Curves
# ================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].plot(history["tl"], label="Train"); axes[0].plot(history["vl"], label="Val")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")
axes[1].plot(history["ta"], label="Train"); axes[1].plot(history["va"], label="Val")
axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].set_xlabel("Epoch")
axes[2].plot(history["vf"], label="F1"); axes[2].plot(history["vu"], label="AUC")
axes[2].set_title("Val F1/AUC"); axes[2].legend(); axes[2].set_xlabel("Epoch")
plt.suptitle("DualForensics Training")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "training_curves.png"), dpi=150)
plt.show()


# ================================================================
# SECTION 11 - Test Evaluation
# ================================================================

bp = os.path.join(CKPT_DIR, "best.pth")
if os.path.exists(bp):
    model.load_state_dict(torch.load(bp, map_location=DEVICE))
    print("Loaded best model.")

model.eval()
all_pb, all_pd, all_lb, all_tp = [], [], [], []
with torch.no_grad():
    for batch in test_loader:
        fr = batch["frames"].to(DEVICE)
        lg, _ = model(fr)
        pb = torch.sigmoid(lg.squeeze(1))
        all_pb.extend(pb.cpu().numpy())
        all_pd.extend((pb>0.5).float().cpu().numpy())
        all_lb.extend(batch["label"].numpy())
        all_tp.extend(batch["type"])

lb_arr = np.array(all_lb); pd_arr = np.array(all_pd); pb_arr = np.array(all_pb)
t_acc = accuracy_score(lb_arr, pd_arr)
t_prec = precision_score(lb_arr, pd_arr, zero_division=0)
t_rec = recall_score(lb_arr, pd_arr, zero_division=0)
t_f1 = f1_score(lb_arr, pd_arr, zero_division=0)
t_auc = roc_auc_score(lb_arr, pb_arr) if len(np.unique(lb_arr))>1 else 0.5

print(f"\nTest Results:")
print(f"  Accuracy:  {t_acc:.4f}")
print(f"  Precision: {t_prec:.4f}")
print(f"  Recall:    {t_rec:.4f}")
print(f"  F1:        {t_f1:.4f}")
print(f"  AUC:       {t_auc:.4f}")

cm = confusion_matrix(lb_arr, pd_arr)
print(f"  Confusion: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

# ROC
fpr, tpr, _ = roc_curve(lb_arr, pb_arr)
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, linewidth=2, label=f"AUC={t_auc:.3f}")
plt.plot([0,1],[0,1],"r--"); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve"); plt.legend(); plt.grid(alpha=0.3)
plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"), dpi=150)
plt.show()

# Per-type
type_acc = defaultdict(lambda: {"c":0,"t":0})
for l, p, t in zip(lb_arr, pd_arr, all_tp):
    type_acc[t]["t"] += 1
    if l == p: type_acc[t]["c"] += 1
print("\nPer-type accuracy:")
for t in sorted(type_acc):
    a = type_acc[t]["c"] / max(type_acc[t]["t"], 1)
    print(f"  {t:25s}: {a:.4f}")


# ================================================================
# SECTION 12 - GradCAM Explainability
# ================================================================

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.grads = None
        self.acts = None
        model.backbone_feat[-1].register_forward_hook(lambda m,i,o: setattr(self, 'acts', o.detach()))
        model.backbone_feat[-1].register_full_backward_hook(lambda m,gi,go: setattr(self, 'grads', go[0].detach()))

    def run(self, video_t):
        self.model.eval()
        inp = video_t.clone().detach().requires_grad_(True)
        lg, exp = self.model(inp)
        pred = torch.sigmoid(lg[0,0]).item()
        self.model.zero_grad()
        lg[0,0].backward(retain_graph=True)
        N = video_t.shape[1]
        hmaps = []
        if self.grads is not None and self.acts is not None:
            for i in range(min(N, self.grads.shape[0])):
                w = self.grads[i].mean(dim=[1,2])
                h = (w[:,None,None] * self.acts[i]).sum(0)
                h = F.relu(h); h = h - h.min(); h = h / (h.max()+1e-8)
                hmaps.append(cv2.resize(h.cpu().numpy(), (FACE_SIZE, FACE_SIZE)))
        else:
            hmaps = [np.zeros((FACE_SIZE, FACE_SIZE))]*N
        return hmaps, pred, exp

REGIONS = {"forehead":(0,56,45,179),"eyes":(56,100,25,199),"nose":(100,145,67,157),
           "mouth":(145,190,45,179),"jaw/chin":(190,224,25,199)}

def region_analysis(hmap):
    return sorted([(n, float(hmap[y1:y2,x1:x2].mean())) for n,(y1,y2,x1,x2) in REGIONS.items()],
                  key=lambda x: -x[1])

def explain_text(pred, hmap, t_imp):
    lbl = "DEEPFAKE" if pred > 0.5 else "REAL"
    lines = [f"Prediction: {lbl} ({pred:.1%} confidence)"]
    if pred > 0.5:
        top = [r for r in region_analysis(hmap) if r[1] > 0.2][:3]
        if top:
            lines.append(f"Spatial: artifacts in {', '.join(r[0] for r in top)}")
        if t_imp is not None:
            th = np.mean(t_imp) + np.std(t_imp)
            anom = np.where(t_imp > th)[0]
            if len(anom): lines.append(f"Temporal: anomalous frames {anom.tolist()}")
    return "\n".join(lines)

gcam = GradCAM(model)


# ================================================================
# SECTION 13 - Explainability Dashboards
# ================================================================

num_explain = 6
explained = 0
for batch in test_loader:
    if explained >= num_explain:
        break
    B = batch["frames"].shape[0]
    for i in range(min(B, num_explain - explained)):
        fr = batch["frames"][i:i+1].to(DEVICE)
        true_l = batch["label"][i].item()
        mtype = batch["type"][i]
        vpath = batch["path"][i]

        try:
            hmaps, pred, exp = gcam.run(fr)
            t_attn = exp["temporal_attn"][0].cpu().numpy()
            t_imp = t_attn.mean(0)
            t_imp = (t_imp - t_imp.min()) / (t_imp.max() - t_imp.min() + 1e-8)
            avg_h = np.mean(hmaps, axis=0)
            N = fr.shape[1]
            show = min(8, N)

            fig = plt.figure(figsize=(20, 12))
            gs = gridspec.GridSpec(3, show, figure=fig, hspace=0.35)

            for j in range(show):
                ax = fig.add_subplot(gs[0, j])
                face = inv_norm(fr[0,j].cpu()).clamp(0,1).permute(1,2,0).numpy()
                ax.imshow(face); ax.imshow(hmaps[j], cmap="jet", alpha=0.4)
                ax.axis("off"); ax.set_title(f"F{j}", fontsize=9)

            ax_t = fig.add_subplot(gs[1, :show//2])
            ax_t.bar(range(N), t_imp, color=plt.cm.RdYlGn_r(t_imp))
            ax_t.set_xlabel("Frame"); ax_t.set_ylabel("Importance")
            ax_t.set_title("Temporal Importance")

            ax_r = fig.add_subplot(gs[1, show//2:])
            regs = region_analysis(avg_h)
            ax_r.barh([r[0] for r in regs], [r[1] for r in regs],
                      color=plt.cm.Reds([r[1]/max(r[1] for r in regs) for r in regs]))
            ax_r.set_title("Region Analysis"); ax_r.invert_yaxis()

            ax_txt = fig.add_subplot(gs[2, :])
            ax_txt.axis("off")
            txt = explain_text(pred, avg_h, t_imp)
            ax_txt.text(0.02, 0.9, txt, transform=ax_txt.transAxes, fontsize=11,
                        va="top", fontfamily="monospace",
                        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

            p_str = "FAKE" if pred > 0.5 else "REAL"
            t_str = "FAKE" if true_l == 1 else "REAL"
            ok = "OK" if (pred>0.5)==(true_l==1) else "WRONG"
            plt.suptitle(f"[{ok}] True={t_str} Pred={p_str}({pred:.0%}) Type={mtype}", fontsize=13)
            plt.tight_layout()
            vn = Path(vpath).stem
            plt.savefig(os.path.join(PLOTS_DIR, f"gradcam_{vn}.png"), dpi=150, bbox_inches="tight")
            plt.show(); plt.close()
            print(f"\nSample {explained+1}: {txt}")
        except Exception as e:
            print(f"  Failed: {e}")
        explained += 1

print(f"\nDashboards saved to {PLOTS_DIR}")


# ================================================================
# SECTION 14 - Save Model
# ================================================================

final_path = os.path.join(OUTPUT_DIR, "dualforensics_final.pth")
torch.save({"model_state_dict": model.state_dict(),
            "test_metrics": {"acc": t_acc, "f1": t_f1, "auc": t_auc}}, final_path)

with open(os.path.join(OUTPUT_DIR, "test_results.json"), "w") as f:
    json.dump({"accuracy": t_acc, "precision": t_prec, "recall": t_rec,
               "f1": t_f1, "auc": t_auc, "cm": cm.tolist()}, f, indent=2)

print(f"Model saved: {final_path}")
print("To download: Save Version > Output tab > download .pth file")


# ================================================================
# SECTION 15 - Inference on New Video
# ================================================================

def predict_video(model, video_path, detector=None):
    if detector is None:
        detector = FaceDetector()
    frames = extract_frames(video_path)
    if not frames:
        return None, "Could not read video."
    sampled = sample_uniform(frames)
    faces = [detector.detect(fr) for fr in sampled]
    tensor = torch.stack([val_tf(f) for f in faces]).unsqueeze(0).to(DEVICE)
    gcam_eng = GradCAM(model)
    hmaps, pred, exp = gcam_eng.run(tensor)
    t_attn = exp["temporal_attn"][0].cpu().numpy()
    t_imp = t_attn.mean(0)
    t_imp = (t_imp - t_imp.min()) / (t_imp.max() - t_imp.min() + 1e-8)
    txt = explain_text(pred, np.mean(hmaps, 0), t_imp)
    print(txt)
    return pred, txt

sample_vid = random.choice(test_v)
print(f"\nInference on: {sample_vid['path']}")
print(f"True label: {'FAKE' if sample_vid['label']==1 else 'REAL'} ({sample_vid['type']})")
predict_video(model, sample_vid["path"], face_detector)

print("\nNotebook complete.")
