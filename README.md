# 🩻 Chest X-Ray Pathology Detection

**Status**: 🚧 Baseline Development Phase  
**Framework**: PyTorch  
**Domain**: Computer Vision / Medical Imaging

## 📋 Project Overview

This project aims to develop a Deep Learning model capable of automatically identifying 14 distinct thoracic pathologies from Chest X-Ray images (PA/AP views).
The dataset is derived from the **NIH Chest X-ray Dataset**, utilizing a subset of 11,219 images for training, validation, and testing.

## 🎯 Objective

To perform **Multi-Label Image Classification**. unlike standard classification (cat vs dog), a single X-ray can present multiple pathologies simultaneously (e.g., both "Infiltration" and "Effusion").

### The 14 Target Classes:

Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax.

## 🏗️ Project Structure

```
X-Ray-Chest/
├── images-224/             # Pre-processed X-ray images (resized to 224x224)
├── src/
│   ├── dataset.py          # Data Engineering: Split (Patient-wise), Clean, One-Hot Encoding
│   ├── dataloader.py       # PyTorch Pipeline: Transforms, Batches, Augmentation
│   ├── model.py            # Architecture: ResNet50 (Transfer Learning)
│   ├── visual.py           # Visualization Utilities for inspection
│   └── main.py             # Training Loop & Validation Engine
├── Data_Entry_2017.csv     # Ground Truth Metadata
└── requirements.txt        # Dependencies
```

## 🧠 Model Architecture (Baseline)

We are establishing a strong baseline using **Transfer Learning**, a standard technique in medical imaging where we leverage knowledge from natural images (ImageNet) to solve medical tasks.

- **Backbone**: **ResNet-50**
  - _Why?_ A deep Residual Network that balances performance and computational efficiency. It solves the vanishing gradient problem effectively.
- **Pre-training**: **ImageNet Weights**
  - _Why?_ Allows the model to recognize edge, texture, and shape patterns immediately, requiring less data and training time.
- **Adjustments**:
  - The final Fully Connected (FC) layer is replaced to output **14 scores** (one per disease).
  - **Activation**: We use raw logits. No Softmax is applied because classes are not mutually exclusive.
- **Loss Function**: **BCEWithLogitsLoss** (Binary Cross Entropy)
  - _Why?_ It treats each of the 14 classes as an independent binary classification problem (Disease Present vs Not Present).

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

### 2. Run Training

To start the baseline training loop:

```bash
python src/main.py
```

## 📉 Current Progress

- [x] **Data Parsing**: CSV loaded, cleaned, and labels One-Hot Encoded.
- [x] **Leakage Prevention**: Validated `patient_split` to ensure individual patients do not cross between Train/Val sets.
- [x] **Baseline Model**: Implemented `MultiLabelResNet` in `model.py`.
- [ ] **Training Evaluation**: Training loop ready for initial execution and metric logging.

---
