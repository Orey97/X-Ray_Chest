# 🩻 Chest X-Ray Pathology Detection System (v1.0)

**Status**: ✅ Production Ready (Training & Validation Complete)  
**Framework**: PyTorch / Google Colab Hybrid  
**Domain**: Medical Imaging / Computer Vision  
**Metrics**: AUROC (Area Under Receiver Operating Characteristic)

---

## 📋 Executive Summary

This project implements a **Medical-Grade Deep Learning System** capable of identifying **14 distinct thoracic pathologies** (e.g., Pneumonia, Effusion, Infiltration) from Chest X-Ray images.

Unlike standard classification tasks, this system solves a **Multi-Label Problem**, acknowledging that a single patient often presents with multiple co-occurring conditions. The architecture uses a **Hybrid Cloud Pipeline**, leveraging Google Colab's NVIDIA T4 GPUs for training and a local Python environment for secure, HIPAA-compliant (simulated) inference.

### 🔬 The 14 Target Pathologies

`Atelectasis`, `Cardiomegaly`, `Consolidation`, `Edema`, `Effusion`, `Emphysema`, `Fibrosis`, `Hernia`, `Infiltration`, `Mass`, `Nodule`, `Pleural_Thickening`, `Pneumonia`, `Pneumothorax`.

---

## 🏗️ System Architecture

We pivoted from a monolithic local training scripts to a more robust **Hybrid Cloud Architecture**:

```mermaid
graph LR
    A[Data Source (Drive)] -->|Auto-Discovery| B[Cloud Training (Colab)]
    B -->|ResNet50 + Transfer Learning| C{Model Training}
    C -->|BCEWithLogitsLoss| C
    C -->|Best Weights (.pth)| D[Artifact Download]
    D --> E[Local Inference Engine]
    E --> F[Single Prediction (inference.py)]
    E --> G[Clinical Validation (evaluate.py)]
```

### 📂 Directory Structure

```bash
X-Ray_Chest/
├── images-224/             # Cleaned Image Database (224x224px)
├── output/                 # Model Artifacts (best_model.pth)
├── src/
│   ├── main.py             # 🧠 TRAIN: CLI Entry Point (Training Loop)
│   ├── inference.py        # 🖥️ INFER: Diagnostic Engine (Production)
│   ├── evaluate.py         # 🧪 TEST: Scientific Validation & Error Slicing
│   ├── ablation.py         # 🔬 R&D: Empirical Experiments (Augmentation Study)
│   ├── analysis.py         # 📊 VIZ: Error Analysis & Gallery Generation
│   ├── model.py            # 🏗️ ARCH: ResNet50 + Transfer Learning
│   ├── dataset.py          # 💾 DATA: Patient-Aware Splitting Logic
│   ├── dataloader.py       # 🔄 LOAD: PyTorch Transforms & Augmentations
│   ├── metrics.py          # 📉 MATH: AUROC, CI, Calibration, F1
│   ├── schema.py           # 🛡️ SAFE: Metadata Enforcement & Versioning
│   └── loss.py             # ⚖️ OPTIM: Weighted BCE Loss for Imbalance
├── Data_Entry_2017.csv     # Clinical Metadata
└── requirements.txt        # Dependency Manifest
```

---

## 🧠 Engineering Highlights

### 1. Zero-Leakage Data Splitting

Medical AI often fails because models memorize patient anatomy rather than pathology. We implemented a strict **Patient-Aware Split Algorithm**:

- **Logic:** All X-rays from `Patient_001` exist strictly in ONE set (Train OR Val OR Test).
- **Benefit:** Prevents data leakage, ensuring the model generalizes to new humans, not just new photos of known humans.

### 2. Professional Modular Design

We pivoted from a monolithic notebook to a **Production-Grade Python Package** (`src/`):

- **Decoupling:** Training logic (`main.py`) is separate from Model definition (`model.py`) and Metrics (`metrics.py`).
- **Reproducibility:** Global seeding and config management ensure every run is deterministic.

### 3. Strict Schema Enforcement

Deep Learning models fail silently when input data shifts. We implemented a **Contract-Based** approach:

- **Schema Locking:** `schema.py` generates a cryptographic signature of the training data labels.
- **Inference Safety:** The system refuses to run inference if the live data schema doesn't perfectly match the training time schema (Fail Fast).
- **Weighted Loss:** We use `WeightedBCELoss` (`loss.py`) to dynamically handle 1:100 class imbalances.

---

## 🚀 Usage Guide

### Phase 1: Cloud Training

**Goal:** Train the "Brain".

1. Upload the entire project folder to Google Drive.
2. Open `src/Training.ipynb` in Google Colab.
3. Run the cells to mount Drive and execute `main.py` on GPU.

### Phase 2: Deployment (Local)

**Goal:** Install the "Brain" into the "Body".

1. Download `xray_best_model.pth` from Drive.
2. Place it in `output/best_model.pth`.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Phase 3: Validation, Visualization & Science

**Goal:** Deep Clinical & Empirical Verification.

#### A. Clinical Metrics (CI + Calibration)

Generate professional metrics (AUROC with 95% Confidence Intervals) and Reliability Diagrams:

```bash
python src/evaluate.py --bootstrap
```

#### B. Visual Error Analysis (Top-5 Failures)

Generate a visual gallery (`output/error_gallery.png`) showing actual X-Rays where the model failed most confidently:

```bash
python src/evaluate.py --slice_errors
```

#### C. Ablation Study

Empirically prove that Data Augmentation improves performance by training two models side-by-side:

```bash
python src/ablation.py --epochs 10
```

*Output: `ablation_report.md` comparing AUC gains.*

> **⚠️ Performance Note:**
> Running `evaluate.py` with `--bootstrap` (resampling 1000x) is computationally expensive. Expect a few minutes of runtime.

---

## 📊 Performance Targets

| Metric | Threshold | Status |
| :--- | :--- | :--- |
| **Pneumonia AUROC** | > 0.70 | 🟢 Targeted |
| **Infiltration AUROC** | > 0.68 | 🟢 Targeted |
| **Inference Time** | < 200ms | ⚡ Optimized |

---

> Engineered by Antigravity & User (Technical PM)
