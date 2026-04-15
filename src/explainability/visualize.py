
"""
Visualization of explainability outputs: heatmaps, temporal importance,
face region analysis, textual explanations, and dashboards.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from configs.settings import FACE_SIZE, PLOTS_DIR
from src.data.dataset import inv_normalize
from src.explainability.gradcam import GradCAM


FACE_REGIONS = {
    "forehead":    (0, 56, 45, 179),
    "eyes":        (56, 100, 25, 199),
    "nose":        (100, 145, 67, 157),
    "mouth":       (145, 190, 45, 179),
    "jaw/chin":    (190, 224, 25, 199),
    "left cheek":  (80, 170, 0, 45),
    "right cheek": (80, 170, 179, 224),
}


def analyze_regions(heatmap):
    scores = []
    for name, (y1, y2, x1, x2) in FACE_REGIONS.items():
        region = heatmap[y1:y2, x1:x2]
        scores.append((name, float(region.mean())))
    scores.sort(key=lambda x: -x[1])
    return scores


def generate_explanation(prediction, heatmap, temporal_imp):
    label = "DEEPFAKE" if prediction > 0.5 else "REAL"
    conf = prediction if prediction > 0.5 else 1 - prediction

    lines = [f"Prediction: {label} (confidence: {conf:.1%})"]

    if prediction > 0.5:
        regions = analyze_regions(heatmap)
        top = [r for r in regions if r[1] > 0.2][:3]
        if top:
            names = [r[0] for r in top]
            lines.append(f"Spatial evidence: artifacts in {', '.join(names)} region(s).")
            lines.append(f"Strongest signal: {top[0][0]} ({top[0][1]:.1%} activation)")

            for name, _ in top:
                if "jaw" in name or "cheek" in name:
                    lines.append("  - Blending boundary artifacts along face edges")
                elif "mouth" in name:
                    lines.append("  - Lip sync or expression transfer artifacts")
                elif "eye" in name:
                    lines.append("  - Unnatural eye reflections or gaze")

        if temporal_imp is not None:
            thresh = np.mean(temporal_imp) + np.std(temporal_imp)
            anomalous = np.where(temporal_imp > thresh)[0]
            if len(anomalous) > 0:
                lines.append(f"Temporal evidence: anomalous frames {anomalous.tolist()}")
    else:
        lines.append("No significant manipulation artifacts detected.")

    return "\n".join(lines)


def create_dashboard(model, batch, sample_idx, device, save_dir=PLOTS_DIR):
    frames = batch["frames"][sample_idx:sample_idx + 1].to(device)
    true_label = batch["label"][sample_idx].item()
    manip_type = batch["type"][sample_idx]
    video_path = batch["path"][sample_idx]

    gradcam = GradCAM(model)
    heatmaps, prediction, explain = gradcam.generate(frames)

    t_attn = explain["temporal_attn"][0].cpu().numpy()
    t_imp = t_attn.mean(axis=0)
    t_imp = (t_imp - t_imp.min()) / (t_imp.max() - t_imp.min() + 1e-8)

    avg_heatmap = np.mean(heatmaps, axis=0)
    N = frames.shape[1]
    show_n = min(8, N)

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(4, show_n, figure=fig, hspace=0.4, wspace=0.2)

    for i in range(show_n):
        ax = fig.add_subplot(gs[0, i])
        face = inv_normalize(frames[0, i].cpu()).clamp(0, 1)
        ax.imshow(face.permute(1, 2, 0).numpy())
        ax.set_title(f"F{i}", fontsize=9)
        ax.axis("off")

    for i in range(show_n):
        ax = fig.add_subplot(gs[1, i])
        face = inv_normalize(frames[0, i].cpu()).clamp(0, 1)
        ax.imshow(face.permute(1, 2, 0).numpy())
        if i < len(heatmaps):
            ax.imshow(heatmaps[i], cmap="jet", alpha=0.4)
        ax.set_title("GradCAM", fontsize=9)
        ax.axis("off")

    ax_t = fig.add_subplot(gs[2, :show_n // 2])
    colors = plt.cm.RdYlGn_r(t_imp)
    ax_t.bar(range(N), t_imp, color=colors)
    ax_t.set_xlabel("Frame"); ax_t.set_ylabel("Importance")
    ax_t.set_title("Temporal Importance")

    ax_r = fig.add_subplot(gs[2, show_n // 2:])
    regions = analyze_regions(avg_heatmap)
    rnames = [r[0] for r in regions[:6]]
    rscores = [r[1] for r in regions[:6]]
    rcolors = plt.cm.Reds(np.array(rscores) / max(max(rscores), 0.01))
    ax_r.barh(rnames, rscores, color=rcolors)
    ax_r.set_xlabel("Activation")
    ax_r.set_title("Face Region Analysis")
    ax_r.invert_yaxis()

    ax_txt = fig.add_subplot(gs[3, :])
    ax_txt.axis("off")
    explanation = generate_explanation(prediction, avg_heatmap, t_imp)
    ax_txt.text(0.02, 0.95, explanation, transform=ax_txt.transAxes,
                fontsize=11, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    pred_str = "FAKE" if prediction > 0.5 else "REAL"
    true_str = "FAKE" if true_label == 1 else "REAL"
    correct = "CORRECT" if (prediction > 0.5) == (true_label == 1) else "WRONG"
    plt.suptitle(
        f"[{correct}] True: {true_str} | Pred: {pred_str} ({prediction:.1%}) | Type: {manip_type}",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout()
    from pathlib import Path
    vname = Path(video_path).stem
    save_path = os.path.join(save_dir, f"gradcam_{vname}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return prediction, explanation


def explain_batch(model, test_loader, device, num_samples=6):
    print(f"\nGenerating explainability dashboards...")
    count = 0
    for batch in test_loader:
        if count >= num_samples:
            break
        B = batch["frames"].shape[0]
        for i in range(min(B, num_samples - count)):
            try:
                pred, expl = create_dashboard(model, batch, i, device)
                print(f"\nSample {count + 1}:")
                print(expl)
            except Exception as e:
                print(f"  Failed: {e}")
            count += 1
    print(f"\nDashboards saved to {PLOTS_DIR}")
