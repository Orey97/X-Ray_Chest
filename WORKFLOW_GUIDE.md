# 🚀 Complete Workflow Guide: End-to-End Project Execution

## From Zero to Production in 30 Minutes

**Target Audience:** Developers, Data Scientists, ML Engineers
**Prerequisites:** Basic Python knowledge, Google Account
**IDE Recommendation:** VS Code, PyCharm, or any modern IDE

---

## 📋 Table of Contents

1. [Overview: What You'll Build](#1-overview-what-youll-build)
2. [Prerequisites & Environment Setup](#2-prerequisites--environment-setup)
3. [Phase 1: Data Preparation](#3-phase-1-data-preparation)
4. [Phase 2: Cloud Training (Google Colab)](#4-phase-2-cloud-training-google-colab)
5. [Phase 3: Local Deployment](#5-phase-3-local-deployment)
6. [Phase 4: Running Inference](#6-phase-4-running-inference)
7. [Phase 5: Evaluation & Metrics](#7-phase-5-evaluation--metrics)
8. [Phase 6: Advanced Experiments](#8-phase-6-advanced-experiments)
9. [Troubleshooting](#9-troubleshooting)
10. [Quick Reference Commands](#10-quick-reference-commands)

---

## 1. Overview: What You'll Build

This guide walks you through the complete pipeline for building a **medical AI system** that detects 14 thoracic pathologies from chest X-rays.

```
┌──────────────────────────────────────────────────────────────────┐
│                        YOUR JOURNEY                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   📦 Setup        →    🧠 Train      →    🖥️ Deploy    →    📊 Use │
│   (15 min)             (1-2 hrs)          (5 min)          (∞)   │
│                                                                   │
│   Install deps         GPU training      Download           Run   │
│   Get dataset          in Colab          artifacts          CLI   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**End Result:** A working CLI tool that takes any chest X-ray image and outputs pathology predictions with confidence scores.

---

## 2. Prerequisites & Environment Setup

### 2.1 Required Software

| Software | Version | Purpose | Download |
|----------|---------|---------|----------|
| **Python** | 3.9+ | Runtime | [python.org](https://python.org) |
| **VS Code** | Latest | IDE | [code.visualstudio.com](https://code.visualstudio.com) |
| **Git** | Latest | Version control | [git-scm.com](https://git-scm.com) |
| **Google Account** | - | Colab access | [accounts.google.com](https://accounts.google.com) |

### 2.2 Clone the Repository

**Option A: Using VS Code**

1. Open VS Code
2. Press `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (Mac)
3. Type "Git: Clone" and press Enter
4. Paste the repository URL:
   ```
   https://github.com/Orey97/X-Ray_Chest.git
   ```
5. Choose a local folder (e.g., `Desktop` or `Projects`)
6. Click "Open" when prompted

**Option B: Using Terminal**

```bash
# Navigate to your preferred directory
cd ~/Desktop

# Clone the repository
git clone https://github.com/Orey97/X-Ray_Chest.git

# Enter the project folder
cd X-Ray_Chest

# Open in VS Code
code .
```

### 2.3 Create Python Virtual Environment

**Why?** Isolates project dependencies from your system Python.

**In VS Code Terminal** (`` Ctrl+` `` to open):

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

### 2.4 Install Dependencies

```bash
# With virtual environment activated
pip install -r requirements.txt
```

**Expected Output:**
```
Successfully installed torch torchvision pandas numpy scikit-learn Pillow tqdm
```

### 2.5 Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
python -c "import tqdm; print(f'tqdm: {tqdm.__version__}')"
```

**Expected:** Version numbers printed without errors.

---

## 3. Phase 1: Data Preparation

### 3.1 Dataset Options

You have two options for obtaining the dataset:

**Option A: Use Sample Dataset (Quick Start)**

The repository includes a small sample in `images-224/` for testing. This is sufficient for verifying the pipeline works.

> ⚠️ **Note:** When training on sample data, the model and schema will only contain the classes present in your sample (e.g., 5 classes instead of 14). This is expected behavior. The full 14-class model requires the complete NIH dataset.

**Option B: Full NIH Dataset (Production)**

1. Download from [Kaggle NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data)
2. Extract to `images-224/` folder
3. Ensure `Data_Entry_2017.csv` is in `src/` folder

### 3.2 Verify Data Structure

Your project should look like this:

```
X-Ray_Chest/
├── images-224/
│   ├── 00000001_000.png
│   ├── 00000001_001.png
│   ├── 00000002_000.png
│   └── ... (more images)
├── src/
│   ├── Data_Entry_2017.csv    ← Metadata file
│   ├── main.py
│   ├── inference.py
│   └── ... (other modules)
└── output/                     ← Will contain trained model
```

**Quick Verification:**

```bash
# Check image count (should be > 0)
ls images-224 | wc -l

# Check CSV exists
ls src/Data_Entry_2017.csv
```

---

## 4. Phase 2: Cloud Training (Google Colab)

### 4.1 Why Colab?

Training requires a GPU. Google Colab provides **free NVIDIA T4 GPUs**. Local training on CPU would take days instead of hours.

### 4.2 Upload Project to Google Drive

1. **Open Google Drive:** [drive.google.com](https://drive.google.com)

2. **Create folder structure:**
   ```
   My Drive/
   └── X-Ray_Chest/           ← Create this folder
       ├── images-224/        ← Upload your images here
       ├── src/               ← Upload entire src folder
       └── output/            ← Create empty folder
   ```

3. **Upload files:**
   - Drag and drop `images-224/` folder
   - Drag and drop `src/` folder
   - Right-click → New Folder → name it `output`

> ⏱️ **Time Estimate:** Uploading 100K images may take 30-60 minutes depending on connection.

### 4.3 Training Options

You have **two training methods**:

**Option A: Standalone Notebook (Recommended for Beginners)**

1. In Google Drive, navigate to `X-Ray_Chest/src/`
2. Right-click on `Training.ipynb`
3. Select **Open with → Google Colaboratory**
4. The notebook is self-contained with all code inline

**Option B: Modular Python Script (Recommended for Production)**

1. Create a new Colab notebook
2. Mount Drive and run `!python src/main.py` with arguments
3. Uses the full modular codebase (`dataset.py`, `model.py`, etc.)

If you don't see Colaboratory:
- Click "Connect more apps"
- Search "Colaboratory"
- Install it

### 4.4 Configure GPU Runtime

1. In Colab, go to **Runtime → Change runtime type**
2. Select:
   - **Hardware accelerator:** T4 GPU
   - **Runtime shape:** High-RAM (if available)
3. Click **Save**

### 4.5 Run Training (Option A: Notebook)

Execute each cell in order by pressing `Shift+Enter`:

**Cell 1: Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```
- Click the authorization link
- Select your Google account
- Copy the authorization code
- Paste it in Colab

**Cell 2: Navigate to Project**
```python
%cd /content/drive/MyDrive/X-Ray_Chest
```

**Cell 3: Install Dependencies**
```python
!pip install torch torchvision pandas scikit-learn Pillow
```

**Cell 4: Start Training**
```python
!python src/main.py \
    --csv_file src/Data_Entry_2017.csv \
    --data_dir images-224 \
    --output_dir output \
    --num_epochs 10
```

### 4.6 Monitor Training Progress

You'll see output like:

```
[CONFIG] Device: cuda (NVIDIA T4)
[MANIFEST] Loaded 14 classes from schema
--- Splitting dataset by PatientID ---
Train: 70,000 | Val: 10,000 | Test: 20,000

Epoch 1/10
━━━━━━━━━━━━━━━━━━━━ 100% | Train Loss: 0.4523 | Val AUROC: 0.65
⚡ Best Model Saved!

Epoch 2/10
━━━━━━━━━━━━━━━━━━━━ 100% | Train Loss: 0.3891 | Val AUROC: 0.68
⚡ Best Model Saved!

...

Training Complete! Best Val AUROC: 0.74
Model saved to: output/best_model.pth
Schema saved to: output/schema.json
```

> ⏱️ **Training Time:** ~1-2 hours for 10 epochs on full dataset

### 4.7 Download Trained Artifacts

After training completes, download these files from Google Drive:

1. `output/best_model.pth` (~94 MB)
2. `output/schema.json` (~300 bytes)

**Download Method:**
- In Google Drive, right-click each file
- Select "Download"
- Save to your local `X-Ray_Chest/output/` folder

---

## 5. Phase 3: Local Deployment

### 5.1 Verify Artifacts

In VS Code terminal:

```bash
# Check model file exists
ls -lh output/best_model.pth

# Check schema file exists
cat output/schema.json
```

**Expected schema.json content:**
```json
{
    "label_list": ["Atelectasis", "Cardiomegaly", ..., "Pneumothorax"],
    "num_classes": 14,
    "timestamp": "2026-01-27T...",
    "version": "1.0"
}
```

> **Note:** If you trained on sample data, `num_classes` may be less than 14 (e.g., 5). The schema always reflects the actual training data composition.

### 5.2 Test Model Loading

```bash
python -c "
import torch
from src.model import MultiLabelResNet

model = MultiLabelResNet(num_classes=14)
model.load_state_dict(torch.load('output/best_model.pth', map_location='cpu'))
print('✅ Model loaded successfully!')
"
```

**Expected:** `✅ Model loaded successfully!`

---

## 6. Phase 4: Running Inference

### 6.1 Single Image Prediction

**Command:**
```bash
python src/inference.py --image images-224/00000001_000.png
```

**Expected Output:**
```
========================================
 🩺 DIAGNOSTIC REPORT: 00000001_000.png
========================================
  🔴 INFILTRATION         72.3%  (High Confidence)
  🔴 EFFUSION             58.1%  (High Confidence)
  🟡 ATELECTASIS          23.4%  (Low Confidence)
  🟡 PNEUMONIA            12.8%  (Low Confidence)
----------------------------------------
Schema Version: 2026-01-27T10:30:00
```

### 6.2 Understanding the Output

| Symbol | Meaning | Threshold |
|--------|---------|-----------|
| 🔴 | High Confidence Finding | ≥ 50% probability |
| 🟡 | Low Confidence Finding | 5% - 50% probability |
| 🟢 | Clean Scan | All findings < 5% |

### 6.3 Batch Processing (Multiple Images)

```bash
# Process all images in a folder
for img in images-224/*.png; do
    python src/inference.py --image "$img"
done
```

### 6.4 Common Inference Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError: schema.json` | Missing schema | Download from Drive → `output/` |
| `RuntimeError: SCHEMA MISMATCH` | Wrong model version | Re-download matching model + schema |
| `FileNotFoundError: Image not found` | Wrong image path | Use absolute path or check spelling |

---

## 7. Phase 5: Evaluation & Metrics

### 7.1 Full Test Set Evaluation

Run comprehensive evaluation on the test split:

```bash
python src/evaluate.py
```

**Expected Output:**
```
============================
📊 EVALUATION RESULTS
============================

Per-Class AUROC:
  Atelectasis:       0.7234
  Cardiomegaly:      0.8102
  Consolidation:     0.6891
  Edema:             0.7456
  ...
  Pneumothorax:      0.7823

Mean AUROC: 0.7412

F1 Score (@ 0.5 threshold): 0.4231
```

### 7.2 Bootstrap Confidence Intervals

Get statistically rigorous uncertainty estimates:

```bash
python src/evaluate.py --bootstrap
```

**Expected Output:**
```
Computing 95% Confidence Interval (1000 bootstrap rounds)...
━━━━━━━━━━━━━━━━━━━━ 100%

Mean AUROC: 0.7412
95% CI: [0.7289, 0.7534]
```

> ⏱️ **Note:** Bootstrap takes 2-5 minutes (resampling 1000×)

### 7.3 Error Analysis (Visual Gallery)

Generate images of the model's worst mistakes:

```bash
python src/evaluate.py --slice_errors
```

**Output Files:**
- `output/error_gallery.png` - Visual grid of failures
- `output/error_analysis_report.md` - Detailed failure breakdown

### 7.4 Calibration Analysis

Check if predicted probabilities match true frequencies:

```bash
python src/evaluate.py --calibration
```

**Output:**
- `output/calibration_data.csv` - Raw calibration data
- Calibration curves (if matplotlib installed)

---

## 8. Phase 6: Advanced Experiments

### 8.1 Ablation Study: Test Augmentation Value

Prove that data augmentation improves performance:

```bash
python src/ablation.py --epochs 5
```

**What Happens:**
1. Trains BASELINE model (no augmentation)
2. Trains AUGMENTED model (full augmentation)
3. Compares both on identical test set
4. Generates `ablation_report.md`

**Expected Output:**
```
================================
🔬 ABLATION STUDY RESULTS
================================

BASELINE (no augmentation):
  Mean AUROC: 0.6891

AUGMENTED (full pipeline):
  Mean AUROC: 0.7234

Delta: +0.0343 (+3.4%)
Conclusion: Augmentation improves generalization ✅
```

### 8.2 Custom Training Configuration

Modify training parameters:

```bash
python src/main.py \
    --csv_file src/Data_Entry_2017.csv \
    --data_dir images-224 \
    --output_dir output \
    --num_epochs 20 \
    --no_aug  # Disable augmentation
```

**Available CLI Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--csv_file` | *required* | Path to CSV metadata file |
| `--data_dir` | *required* | Path to images directory |
| `--output_dir` | `output` | Directory for model artifacts |
| `--num_epochs` | 10 | Training epochs |
| `--no_aug` | False | Disable augmentation (for ablation) |
| `--use_weighted_loss` | False | Enable class imbalance weighting |

**Hardcoded Configuration (in `main.py`):**

To modify these, edit the `CONFIG` dictionary in `src/main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Images per batch |
| `learning_rate` | 1e-4 | Initial learning rate |
| `patience` | 3 | Early stopping patience |
| `num_workers` | 0 | DataLoader workers (0 for Windows) |

---

## 9. Troubleshooting

### 9.1 Common Issues

**Issue: `ModuleNotFoundError: No module named 'torch'`**
```bash
# Solution: Activate virtual environment first
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Then reinstall
pip install -r requirements.txt
```

**Issue: `CUDA out of memory`**
```bash
# Solution: Reduce batch size
python src/main.py --batch_size 16
```

**Issue: `FileNotFoundError: Data_Entry_2017.csv`**
```bash
# Solution: Verify CSV is in correct location
ls src/Data_Entry_2017.csv

# Or specify absolute path
python src/main.py --csv_file /full/path/to/Data_Entry_2017.csv
```

**Issue: Training is extremely slow**
- Ensure you're using Colab GPU (not CPU)
- Check: Runtime → Change runtime type → T4 GPU

**Issue: Model predictions all similar (~0.5)**
- Model may not have trained properly
- Check training logs for decreasing loss
- Ensure sufficient training epochs (≥5)

### 9.2 Getting Help

1. **Check Logs:** Training outputs detailed progress
2. **Verify Data:** Ensure images exist in correct paths
3. **Test Components:** Use the verification commands above
4. **GPU Issues:** Restart Colab runtime

---

## 10. Quick Reference Commands

### Setup
```bash
# Clone repo
git clone https://github.com/Orey97/X-Ray_Chest.git
cd X-Ray_Chest

# Create environment
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Training (Colab)
```python
!python src/main.py --csv_file src/Data_Entry_2017.csv --data_dir images-224 --output_dir output
```

### Inference
```bash
# Single image
python src/inference.py --image path/to/image.png
```

### Evaluation
```bash
# Basic evaluation
python src/evaluate.py

# With confidence intervals
python src/evaluate.py --bootstrap

# Error visualization
python src/evaluate.py --slice_errors
```

### Experiments
```bash
# Ablation study
python src/ablation.py --epochs 5
```

---

## 🎉 Congratulations!

You've successfully:
- ✅ Set up a complete ML development environment
- ✅ Trained a production-grade medical AI model
- ✅ Deployed it for local inference
- ✅ Evaluated performance with statistical rigor
- ✅ Conducted scientific ablation experiments

**Next Steps:**
1. Try with your own chest X-ray images
2. Experiment with different hyperparameters
3. Extend to additional pathologies
4. Deploy as a web service (Flask/FastAPI)

---

## Appendix: VS Code Recommended Extensions

For the best development experience, install these VS Code extensions:

| Extension | Purpose |
|-----------|---------|
| **Python** (Microsoft) | Python language support |
| **Pylance** | Fast Python IntelliSense |
| **Jupyter** | Notebook support |
| **GitLens** | Enhanced Git integration |
| **Markdown Preview Enhanced** | Better markdown rendering |

**Install via VS Code:**
1. Press `Ctrl+Shift+X`
2. Search each extension name
3. Click "Install"

---

**Document Version:** 1.0 | **Last Updated:** January 2026
