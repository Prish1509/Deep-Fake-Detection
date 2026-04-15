
"""
Grad-CAM for spatial explainability.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from configs.settings import FACE_SIZE


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        target = model.backbone.features[-1]
        target.register_forward_hook(self._fwd_hook)
        target.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, video_tensor):
        """
        Args:
            video_tensor: [1, N, 3, H, W]
        Returns:
            heatmaps: list of N arrays [H, W] in [0, 1]
            prediction: float
            explain: dict with attention data
        """
        self.model.eval()
        inp = video_tensor.clone().detach().requires_grad_(True)
        logits, explain = self.model(inp)
        pred = torch.sigmoid(logits[0, 0]).item()

        self.model.zero_grad()
        logits[0, 0].backward(retain_graph=True)

        N = video_tensor.shape[1]
        heatmaps = []

        if self.gradients is not None and self.activations is not None:
            for i in range(min(N, self.gradients.shape[0])):
                g = self.gradients[i]
                a = self.activations[i]
                w = g.mean(dim=[1, 2])
                hmap = (w[:, None, None] * a).sum(0)
                hmap = F.relu(hmap)
                hmap = hmap - hmap.min()
                hmap = hmap / (hmap.max() + 1e-8)
                hmap = cv2.resize(hmap.cpu().numpy(), (FACE_SIZE, FACE_SIZE))
                heatmaps.append(hmap)
        else:
            heatmaps = [np.zeros((FACE_SIZE, FACE_SIZE))] * N

        return heatmaps, pred, explain
