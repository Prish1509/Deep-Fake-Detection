
"""
Training loop with validation, early stopping, and checkpointing.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.settings import (
    DEVICE, LR, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE,
    REAL_WEIGHT, FAKE_WEIGHT, CHECKPOINT_DIR, PLOTS_DIR, LOGS_DIR,
)
from src.training.metrics import compute_metrics, per_type_accuracy, get_roc_data


def weighted_bce(logits, targets):
    probs = torch.sigmoid(logits)
    w = torch.where(
        targets == 0,
        torch.tensor(REAL_WEIGHT, device=targets.device),
        torch.tensor(FAKE_WEIGHT, device=targets.device),
    )
    bce = -(targets * torch.log(probs + 1e-8) + (1 - targets) * torch.log(1 - probs + 1e-8))
    return (w * bce).mean()


class Trainer:
    def __init__(self, model, train_loader, val_loader, model_name="dualforensics"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.name = model_name
        self.optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)
        self.best_auc = 0.0
        self.patience_ctr = 0
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "val_f1": [], "val_auc": [],
        }

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in self.train_loader:
            frames = batch["frames"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            self.optimizer.zero_grad()
            logits, _ = self.model(frames)
            logits = logits.squeeze(1)
            loss = weighted_bce(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return total_loss / len(self.train_loader), accuracy_score(all_labels, all_preds)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_probs, all_preds, all_labels, all_types = [], [], [], []

        for batch in self.val_loader:
            frames = batch["frames"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits, _ = self.model(frames)
            logits = logits.squeeze(1)
            total_loss += weighted_bce(logits, labels).item()

            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend((probs > 0.5).float().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_types.extend(batch["type"])

        metrics = compute_metrics(all_labels, all_preds, all_probs)
        metrics["loss"] = total_loss / max(len(self.val_loader), 1)
        metrics["per_type"] = per_type_accuracy(all_labels, all_preds, all_types)
        return metrics

    def train(self, epochs=NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Training: {self.name} | Device: {DEVICE} | Epochs: {epochs}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self.train_epoch()
            val = self.validate()
            lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(val["loss"])
            self.history["val_acc"].append(val["accuracy"])
            self.history["val_f1"].append(val["f1"])
            self.history["val_auc"].append(val["auc"])

            dt = time.time() - t0
            print(f"Epoch {epoch}/{epochs} [{dt:.0f}s] lr={lr:.1e}")
            print(f"  Train: loss={tr_loss:.4f} acc={tr_acc:.4f}")
            print(f"  Val:   loss={val['loss']:.4f} acc={val['accuracy']:.4f} "
                  f"f1={val['f1']:.4f} auc={val['auc']:.4f}")

            if val["auc"] > self.best_auc:
                self.best_auc = val["auc"]
                self.patience_ctr = 0
                ckpt = os.path.join(CHECKPOINT_DIR, f"{self.name}_best.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "best_auc": self.best_auc,
                }, ckpt)
                print(f"  * Best model saved (AUC={self.best_auc:.4f})")
            else:
                self.patience_ctr += 1
                if self.patience_ctr >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        self._save_history()
        self._plot_curves()
        return self.history

    def load_best(self):
        path = os.path.join(CHECKPOINT_DIR, f"{self.name}_best.pth")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded best model from epoch {ckpt['epoch']}")

    def _save_history(self):
        path = os.path.join(LOGS_DIR, f"{self.name}_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def _plot_curves(self):
        h = self.history
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        axes[0].plot(h["train_loss"], label="Train")
        axes[0].plot(h["val_loss"], label="Val")
        axes[0].set_title("Loss"); axes[0].legend()
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")

        axes[1].plot(h["train_acc"], label="Train")
        axes[1].plot(h["val_acc"], label="Val")
        axes[1].set_title("Accuracy"); axes[1].legend()
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")

        axes[2].plot(h["val_f1"], label="F1")
        axes[2].plot(h["val_auc"], label="AUC")
        axes[2].set_title("Val F1 / AUC"); axes[2].legend()
        axes[2].set_xlabel("Epoch")

        plt.suptitle(f"{self.name} Training Curves")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{self.name}_training_curves.png"), dpi=150)
        plt.close()


@torch.no_grad()
def evaluate_test(model, test_loader, model_name="model"):
    model.eval()
    all_probs, all_preds, all_labels, all_types, all_paths = [], [], [], [], []

    for batch in test_loader:
        frames = batch["frames"].to(DEVICE)
        logits, _ = model(frames)
        probs = torch.sigmoid(logits.squeeze(1))
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend((probs > 0.5).float().cpu().numpy())
        all_labels.extend(batch["label"].numpy())
        all_types.extend(batch["type"])
        all_paths.extend(batch["path"])

    labels = np.array(all_labels)
    preds = np.array(all_preds)
    probs = np.array(all_probs)

    m = compute_metrics(labels, preds, probs)
    pt = per_type_accuracy(labels, preds, all_types)

    print(f"\n{'='*60}")
    print(f"Test Results: {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy:  {m['accuracy']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall:    {m['recall']:.4f}")
    print(f"  F1:        {m['f1']:.4f}")
    print(f"  AUC:       {m['auc']:.4f}")
    print(f"\n  Per-type accuracy:")
    for t, a in sorted(pt.items()):
        print(f"    {t:25s}: {a:.4f}")

    # ROC curve
    fpr, tpr, auc = get_roc_data(labels, probs)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "r--", linewidth=1)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_roc_curve.png"), dpi=150)
    plt.close()

    # confusion matrix
    cm = np.array(m["cm"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"]); ax.set_yticklabels(["Real", "Fake"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_confusion_matrix.png"), dpi=150)
    plt.close()

    # save results
    results = {k: v for k, v in m.items() if k != "cm"}
    results["confusion_matrix"] = m["cm"]
    results["per_type"] = pt
    with open(os.path.join(LOGS_DIR, f"{model_name}_test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return m, labels, preds, probs, all_types, all_paths
