# DualForensics: Deepfake Detection with Explainability

Dual-branch spatiotemporal attention network for deepfake detection using the FaceForensics++ dataset. Combines spatial artifact analysis (CNN + CBAM) with temporal inconsistency detection (Transformer encoder) through cross-attention fusion. Includes Grad-CAM explainability.

## Project Structure

```
project/
├── configs/
│   └── settings.py           # All hyperparameters and paths
├── src/
│   ├── data/                  # [Member 1] Data pipeline
│   │   ├── video_processor.py #   Frame extraction, sampling
│   │   ├── face_detector.py   #   MTCNN + Haar cascade
│   │   ├── preprocessing.py   #   Discovery, extraction, splitting
│   │   └── dataset.py         #   PyTorch Dataset + DataLoaders
│   ├── models/                # [Member 2] Architecture
│   │   ├── backbone.py        #   EfficientNet-B0
│   │   ├── attention.py       #   CBAM (channel + spatial)
│   │   ├── temporal.py        #   Transformer encoder
│   │   ├── fusion.py          #   Cross-attention fusion
│   │   └── dualforensics.py   #   Full model + baselines
│   ├── training/              # [Member 3] Training
│   │   ├── trainer.py         #   Training loop, evaluation
│   │   └── metrics.py         #   Accuracy, F1, AUC, etc.
│   └── explainability/        # [Member 3] XAI
│       ├── gradcam.py         #   Grad-CAM implementation
│       └── visualize.py       #   Dashboards, text explanations
├── outputs/
│   ├── plots/                 #   All saved figures
│   ├── models/                #   Checkpoints
│   └── logs/                  #   Training history, results
├── main.py                    #   Entry point
└── requirements.txt
```

## Team Contributions

| Member | Responsibility | Files |
|--------|---------------|-------|
| Member 1 | Data pipeline, face detection, dataset class | `src/data/*` |
| Member 2 | Model architecture, attention, transformer | `src/models/*` |
| Member 3 | Training, evaluation, explainability | `src/training/*`, `src/explainability/*` |

## Quick Start

```bash
pip install -r requirements.txt

# Full pipeline
python main.py --model dualforensics --epochs 15

# Baselines
python main.py --model cnn_only --epochs 10
python main.py --model cnn_lstm --epochs 10

# Skip training, just evaluate + explain
python main.py --model dualforensics --skip-train --explain 10
```

## Dataset

FaceForensics++ (c23): https://www.kaggle.com/datasets/xdxd003/ff-c23

~7000 videos: 1000 real + 6000 fake across 6 manipulation types.
