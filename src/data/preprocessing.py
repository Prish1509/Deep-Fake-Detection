"""
Dataset discovery and face extraction pipeline.
"""

import os
import glob
import json
import time
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

from configs.settings import (
    DATASET_ROOT, FACES_DIR, REAL_FOLDER, FAKE_FOLDERS,
    NUM_FRAMES, SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
)
from src.data.video_processor import extract_frames, sample_uniform
from src.data.face_detector import FaceDetector
from sklearn.model_selection import train_test_split


def discover_videos(dataset_root=DATASET_ROOT):
    videos = []
    exts = (".mp4", ".avi", ".mov", ".mkv")

    real_dir = os.path.join(dataset_root, REAL_FOLDER)
    if os.path.isdir(real_dir):
        for root, _, files in os.walk(real_dir):
            for f in sorted(files):
                if f.lower().endswith(exts):
                    videos.append({
                        "path": os.path.join(root, f),
                        "label": 0,
                        "type": "original",
                    })

    for folder in FAKE_FOLDERS:
        fdir = os.path.join(dataset_root, folder)
        if os.path.isdir(fdir):
            for root, _, files in os.walk(fdir):
                for f in sorted(files):
                    if f.lower().endswith(exts):
                        videos.append({
                            "path": os.path.join(root, f),
                            "label": 1,
                            "type": folder,
                        })

    return videos


def preprocess_videos(video_list, detector, output_dir=FACES_DIR):
    processed = []
    failed = 0
    t0 = time.time()

    for i, vinfo in enumerate(video_list):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(video_list) - i - 1) / rate / 60
            print(f"  [{i+1}/{len(video_list)}] {rate:.1f} vid/s, ETA {eta:.0f}m")

        vname = Path(vinfo["path"]).stem
        save_dir = os.path.join(output_dir, vinfo["type"], vname)

        if os.path.isdir(save_dir):
            existing = glob.glob(os.path.join(save_dir, "*.jpg"))
            if len(existing) >= NUM_FRAMES:
                processed.append({**vinfo, "faces_dir": save_dir})
                continue

        frames = extract_frames(vinfo["path"])
        if not frames:
            failed += 1
            continue

        sampled = sample_uniform(frames)
        os.makedirs(save_dir, exist_ok=True)

        for j, frame in enumerate(sampled):
            face = detector.detect(frame)
            face.save(os.path.join(save_dir, f"frame_{j:02d}.jpg"))

        processed.append({**vinfo, "faces_dir": save_dir})

    elapsed = time.time() - t0
    print(f"Preprocessing done: {len(processed)} ok, {failed} failed ({elapsed/60:.1f}m)")

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(processed, f, indent=2)

    return processed


def split_dataset(videos, seed=SEED):
    strat = [f"{v['label']}_{v['type']}" for v in videos]

    train, temp, _, temp_keys = train_test_split(
        videos, strat,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=strat,
        random_state=seed,
    )
    val, test = train_test_split(
        temp,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_keys,
        random_state=seed,
    )
    return train, val, test


def load_or_preprocess(dataset_root=DATASET_ROOT, faces_dir=FACES_DIR):
    meta_path = os.path.join(faces_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)

    videos = discover_videos(dataset_root)
    detector = FaceDetector()
    return preprocess_videos(videos, detector, faces_dir)
