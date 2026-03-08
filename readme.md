# Deepfake Detection using EfficientNet

## Project Overview

This project focuses on detecting deepfake videos using deep learning. The system processes video frames, extracts faces, and uses a convolutional neural network to classify whether the input is **real or fake**.
The backbone model used for feature extraction is **EfficientNet**, which is known for high accuracy and computational efficiency.

---

# EfficientNet Architecture

EfficientNet is a convolutional neural network architecture designed to achieve high accuracy while using fewer parameters and less computation. It was introduced by researchers at Google and uses a technique called **compound scaling** to balance network depth, width, and input resolution.

## Key Components of EfficientNet

### 1. MBConv Blocks

EfficientNet is built using **Mobile Inverted Bottleneck Convolution (MBConv)** blocks.
Each MBConv block contains the following steps:

1. **Expansion Layer (1×1 Convolution)**
   Expands the number of channels to increase feature representation.

2. **Depthwise Convolution**
   Applies convolution separately to each channel to reduce computational cost.

3. **Squeeze-and-Excitation (SE) Block**
   Reweights channels so the network focuses on important features.

4. **Projection Layer (1×1 Convolution)**
   Reduces the number of channels back to the required size.

5. **Skip Connection**
   Helps preserve information and improves gradient flow during training.

---

### EfficientNet-B0 Layer Structure

| Stage | Operator                            | Resolution | Channels | Layers |
| ----- | ----------------------------------- | ---------- | -------- | ------ |
| 1     | Conv3×3                             | 224×224    | 32       | 1      |
| 2     | MBConv1, k3×3                       | 112×112    | 16       | 1      |
| 3     | MBConv6, k3×3                       | 112×112    | 24       | 2      |
| 4     | MBConv6, k5×5                       | 56×56      | 40       | 2      |
| 5     | MBConv6, k3×3                       | 28×28      | 80       | 3      |
| 6     | MBConv6, k5×5                       | 14×14      | 112      | 3      |
| 7     | MBConv6, k5×5                       | 14×14      | 192      | 4      |
| 8     | MBConv6, k3×3                       | 7×7        | 320      | 1      |
| 9     | Conv1×1 + Pooling + Fully Connected | 7×7        | 1280     | 1      |

As the network goes deeper:

* Spatial resolution decreases
* Number of channels increases
* Features become more abstract

---

# Why EfficientNet is Used for Deepfake Detection

Deepfake detection requires identifying **very subtle visual artifacts** in manipulated faces. EfficientNet is well suited for this task for several reasons.

## 1. Strong Feature Extraction

EfficientNet can capture fine details such as:

* skin texture inconsistencies
* blending artifacts around face boundaries
* unnatural lighting patterns

These details are important for detecting deepfake manipulations.

---

## 2. Efficient Architecture

EfficientNet achieves high performance with fewer parameters compared to traditional CNN models like ResNet or VGG.
This makes training faster and reduces computational cost.

---

## 3. Compound Scaling

EfficientNet scales the model by balancing:

* network depth
* network width
* input resolution

This allows the network to learn richer representations while maintaining efficiency.

---

## 4. High Accuracy for Image Classification

EfficientNet has achieved state-of-the-art performance on image classification benchmarks such as ImageNet.
Because deepfake detection is also an image classification task, EfficientNet provides strong performance.

---

## 5. Good for High-Resolution Face Features

Deepfake artifacts are often very small. EfficientNet processes high-resolution images effectively, allowing the model to capture these subtle manipulations.

---

# Deepfake Detection Pipeline

The full detection pipeline used in this project is:

1. Video input
2. Frame extraction
3. Face detection (MTCNN)
4. Face cropping and resizing (224×224)
5. Feature extraction using EfficientNet
6. Classification layer (Real vs Fake)

---

# Future Improvements

Possible improvements to the system include:

* Temporal modeling using Transformers or LSTM
* Attention mechanisms to focus on manipulated regions
* Explainable AI techniques such as Grad-CAM
* Multi-frame deepfake detection models

---
