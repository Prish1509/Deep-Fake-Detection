"""
Video frame extraction and temporal sampling.
"""

import cv2
import numpy as np
from configs.settings import NUM_FRAMES, MAX_FRAMES_TO_READ


def extract_frames(video_path, max_frames=MAX_FRAMES_TO_READ):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1
    cap.release()
    return frames


def sample_uniform(frames, n=NUM_FRAMES):
    total = len(frames)
    if total == 0:
        return []
    if total <= n:
        return list(frames) + [frames[-1]] * (n - total)
    indices = np.linspace(0, total - 1, n, dtype=int)
    return [frames[i] for i in indices]


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
    }
    cap.release()
    return info
