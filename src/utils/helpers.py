
"""Misc utility functions."""

import random
import numpy as np
import torch
from configs.settings import SEED

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_gpu_info():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {name} ({mem:.1f} GB)")
    else:
        print("No GPU available. Using CPU.")
