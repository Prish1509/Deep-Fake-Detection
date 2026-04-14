"""
PyTorch dataset and dataloader utilities.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

from configs.settings import FACE_SIZE, NUM_FRAMES, BATCH_SIZE, NUM_WORKERS


train_transform = transforms.Compose([
    transforms.Resize((FACE_SIZE, FACE_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((FACE_SIZE, FACE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225],
)


class VideoDataset(Dataset):
    """Loads a sequence of N face images per video."""

    def __init__(self, video_list, transform=None):
        self.videos = video_list
        self.transform = transform or val_transform

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        info = self.videos[idx]
        faces_dir = info.get("faces_dir", "")

        face_files = sorted(glob.glob(os.path.join(faces_dir, "frame_*.jpg")))
        faces = []
        for fp in face_files[:NUM_FRAMES]:
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                img = Image.new("RGB", (FACE_SIZE, FACE_SIZE))
            faces.append(img)

        while len(faces) < NUM_FRAMES:
            faces.append(faces[-1] if faces else Image.new("RGB", (FACE_SIZE, FACE_SIZE)))

        frames = torch.stack([self.transform(f) for f in faces])

        return {
            "frames": frames,
            "label": torch.tensor(info["label"], dtype=torch.float32),
            "type": info["type"],
            "path": info["path"],
        }


def build_dataloaders(train_vids, val_vids, test_vids):
    train_ds = VideoDataset(train_vids, train_transform)
    val_ds = VideoDataset(val_vids, val_transform)
    test_ds = VideoDataset(test_vids, val_transform)

    labels = [v["label"] for v in train_vids]
    counts = np.bincount(labels)
    weights = 1.0 / counts
    sample_w = torch.DoubleTensor([weights[l] for l in labels])
    sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
