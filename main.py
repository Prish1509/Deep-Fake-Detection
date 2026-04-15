"""
DualForensics - Deepfake Detection
Main entry point: preprocess -> train -> evaluate -> explain
"""

import argparse
import torch

from configs.settings import DEVICE, DATASET_ROOT, FACES_DIR
from src.utils.helpers import set_seed, print_gpu_info
from src.data.preprocessing import load_or_preprocess, split_dataset
from src.data.dataset import build_dataloaders
from src.models.dualforensics import build_model
from src.training.trainer import Trainer, evaluate_test
from src.explainability.visualize import explain_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dualforensics",
                        choices=["dualforensics", "cnn_only", "cnn_lstm"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--explain", type=int, default=6,
                        help="Number of samples to generate explanations for")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    set_seed()
    print_gpu_info()

    # Data pipeline
    print("\n--- Loading dataset ---")
    videos = load_or_preprocess(DATASET_ROOT, FACES_DIR)
    train_vids, val_vids, test_vids = split_dataset(videos)
    print(f"Split: train={len(train_vids)} val={len(val_vids)} test={len(test_vids)}")

    train_loader, val_loader, test_loader = build_dataloaders(
        train_vids, val_vids, test_vids
    )

    # Model
    print(f"\n--- Building model: {args.model} ---")
    model = build_model(args.model)

    # Training
    if not args.skip_train:
        trainer = Trainer(model, train_loader, val_loader, args.model)
        trainer.train(epochs=args.epochs)
        trainer.load_best()

    # Evaluation
    print("\n--- Evaluating on test set ---")
    evaluate_test(model, test_loader, args.model)

    # Explainability
    if args.model == "dualforensics" and args.explain > 0:
        print("\n--- Generating explainability outputs ---")
        explain_batch(model, test_loader, DEVICE, args.explain)

    print("\nDone.")


if __name__ == "__main__":
    main()
