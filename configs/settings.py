"""
Project configuration.
All hyperparameters and paths in one place.
"""

import os
import torch

# -- Paths --
DATASET_ROOT = "/kaggle/input/faceforensics-dataset-c23/FaceForensics++_C23"
FACES_DIR = "./data/faces"
CHECKPOINT_DIR = "./outputs/models"
PLOTS_DIR = "./outputs/plots"
LOGS_DIR = "./outputs/logs"

REAL_FOLDER = "original"
FAKE_FOLDERS = [
    "DeepFakeDetection", "Deepfakes", "Face2Face",
    "FaceShifter", "FaceSwap", "NeuralTextures",
]

# -- Video Processing --
NUM_FRAMES = 16
FACE_SIZE = 224
FACE_MARGIN = 0.1
MAX_FRAMES_TO_READ = 300

# -- Model --
BACKBONE = "efficientnet_b0"
FEATURE_DIM = 1280
SPATIAL_ATT_REDUCTION = 16
TEMPORAL_HEADS = 4
TEMPORAL_LAYERS = 2
TEMPORAL_FF_DIM = 512
FUSION_DIM = 256
DROPOUT = 0.3
TRANSFORMER_DROPOUT = 0.1

# -- Training --
BATCH_SIZE = 4
NUM_WORKERS = 2
LR = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 15
PATIENCE = 5
REAL_WEIGHT = 6.0
FAKE_WEIGHT = 1.0

# -- Split --
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

# -- Device --
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Create dirs --
for d in [FACES_DIR, CHECKPOINT_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)
