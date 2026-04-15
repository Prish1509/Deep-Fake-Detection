
"""
Evaluation metrics and result analysis.
"""

import numpy as np
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)


def compute_metrics(labels, preds, probs):
    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5,
        "cm": confusion_matrix(labels, preds).tolist(),
    }


def per_type_accuracy(labels, preds, types):
    buckets = defaultdict(lambda: {"correct": 0, "total": 0})
    for l, p, t in zip(labels, preds, types):
        buckets[t]["total"] += 1
        if l == p:
            buckets[t]["correct"] += 1
    return {t: d["correct"] / max(d["total"], 1) for t, d in buckets.items()}


def get_roc_data(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    return fpr, tpr, auc

