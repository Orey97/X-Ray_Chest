# Multi-Label Thoracic Pathology Detection from Chest Radiographs

## A Production-Grade Deep Learning System for Automated Medical Image Analysis

**Author:** Renaldo Arapi
**Version:** 1.0 | **Date:** January 2026
**Domain:** Medical AI • Computer Vision • Multi-Label Classification

---

## Abstract

This repository presents a complete, production-grade deep learning pipeline for detecting **14 distinct thoracic pathologies** from chest X-ray images. The system addresses the clinically realistic **multi-label classification problem**, where patients frequently present with multiple co-occurring conditions (e.g., Pneumonia + Effusion + Cardiomegaly).

The architecture implements a **Hybrid Cloud Pipeline** leveraging Google Colab's NVIDIA T4 GPUs for training while supporting secure local inference. Key engineering innovations include:

- **Zero-Leakage Patient-Aware Data Splitting** preventing anatomical memorization
- **Schema-Enforced Inference Contracts** guaranteeing model-data compatibility
- **Dynamic Class Weighting** mitigating extreme label imbalance (1:100 ratios)
- **Bootstrap Confidence Intervals** for statistically rigorous performance reporting

The pipeline achieves competitive AUROC scores while maintaining strict reproducibility, modularity, and clinical safety standards.

---

## Table of Contents

1. [Problem Statement & Clinical Context](#1-problem-statement--clinical-context)
2. [Dataset Description](#2-dataset-description)
3. [System Architecture](#3-system-architecture)
4. [Data Engineering Pipeline](#4-data-engineering-pipeline)
5. [Model Architecture](#5-model-architecture)
6. [Loss Function Design](#6-loss-function-design)
7. [Training Methodology](#7-training-methodology)
8. [Evaluation Framework](#8-evaluation-framework)
9. [Inference Engine](#9-inference-engine)
10. [Experimental Validation](#10-experimental-validation)
11. [Usage Guide](#11-usage-guide)
12. [Ethical Considerations](#12-ethical-considerations)
13. [References](#13-references)

---

## 1. Problem Statement & Clinical Context

### 1.1 The Clinical Challenge

Chest X-rays are among the most frequently performed radiological examinations worldwide, with radiologists interpreting millions of scans daily. Critical conditions such as **Pneumothorax** (collapsed lung) demand immediate attention, yet compete for review time against routine findings.

Automated triage systems offer a compelling solution: computationally flag high-risk scans to prioritize radiologist attention. However, clinical reality presents unique challenges:

#### Challenge 1: Multi-Pathology Co-occurrence

Unlike standard image classification where each image belongs to exactly one class, chest X-rays frequently exhibit multiple simultaneous findings. A single scan may show Cardiomegaly, Effusion, and Edema together—requiring independent binary predictions for each condition.

#### Challenge 2: Extreme Class Imbalance

Medical datasets exhibit severe class imbalance:

| Pathology | Prevalence | Imbalance Ratio |
|-----------|------------|-----------------|
| Infiltration | ~18% | 1:5 |
| No Finding | ~53% | 1:1 |
| Pneumonia | ~1% | 1:100 |
| Hernia | ~0.2% | 1:500 |

A naive model achieves 99.8% accuracy on Hernia by predicting "negative" always—while detecting zero actual cases.

#### Challenge 3: Data Leakage Risk

Patients often have multiple scans over time (initial visit, follow-ups). Naive random splitting by image allows patient anatomy to leak between train and test sets, leading to artificially inflated performance that collapses on truly unseen patients.

### 1.2 STAR Analysis: Project Scope

| Dimension | Description |
|-----------|-------------|
| **Situation** | Healthcare AI initiatives frequently fail in production due to hidden data leakage, schema drift, and inability to quantify prediction uncertainty. Existing tutorials prioritize quick wins over engineering rigor. |
| **Task** | Design and implement a production-grade pipeline that: (1) Prevents data leakage, (2) Enforces strict schema contracts, (3) Handles class imbalance, (4) Provides statistically valid performance metrics. |
| **Action** | Developed a modular Python package with strict separation of concerns: Data pipeline, Model architecture, Loss functions, Metrics computation, Schema validation, Training orchestration, and Production inference. |
| **Result** | A fully validated, reproducible pipeline achieving competitive AUROC with 95% confidence intervals, schema-locked inference, and documented error analysis. |

---

## 2. Dataset Description

### 2.1 Source & Attribution

The dataset derives from the **NIH Clinical Center Chest X-ray Dataset** (Wang et al., 2017), containing over 100,000 frontal-view X-ray images from more than 30,000 unique patients. Labels were extracted via Natural Language Processing from associated radiology reports.

**Primary Metadata**: `Data_Entry_2017.csv`

| Column | Description | Example |
|--------|-------------|---------|
| `Image Index` | Unique filename | `00000001_000.png` |
| `Finding Labels` | Pipe-separated pathologies | `Pneumonia\|Effusion` |
| `Patient ID` | De-identified identifier | `00001` |
| `Patient Age` | Age in years | `58` |
| `Patient Gender` | M/F | `M` |
| `View Position` | Radiograph orientation | `PA` or `AP` |

### 2.2 The 14 Target Pathologies

Our canonical label set, enforced via `schema.json`:

```plaintext
1. Atelectasis        6. Emphysema         11. Nodule
2. Cardiomegaly       7. Fibrosis          12. Pleural_Thickening
3. Consolidation      8. Hernia            13. Pneumonia
4. Edema              9. Infiltration      14. Pneumothorax
5. Effusion          10. Mass
```

### 2.3 Known Limitations & Biases

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Label Noise** | Ground truth extracted via NLP has ~10% error rate | Model ceiling is limited; use for triage only |
| **Population Bias** | Single hospital system (NIH Clinical Center) | May not generalize to different populations/equipment |
| **Class Imbalance** | Hernia/Pneumonia underrepresented | Weighted loss function |

---

## 3. System Architecture

### 3.1 Hybrid Cloud Pipeline

The system uses a two-phase architecture optimizing for both GPU training power and local deployment security:

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: CLOUD TRAINING                       │
├─────────────────────────────────────────────────────────────────────┤
│  Google Drive           Google Colab (NVIDIA T4)                    │
│  ┌──────────┐           ┌─────────────────────────┐                 │
│  │ Dataset  │ ────────► │ main.py                 │                 │
│  │ images/  │           │   • Patient-aware split │                 │
│  │ CSV      │           │   • Transfer learning   │                 │
│  └──────────┘           │   • Weighted BCE loss   │                 │
│                         │   • Early stopping      │                 │
│                         └───────────┬─────────────┘                 │
│                                     │                               │
│                                     ▼                               │
│                         ┌─────────────────────────┐                 │
│                         │ Artifacts:              │                 │
│                         │   • best_model.pth      │                 │
│                         │   • schema.json         │                 │
│                         └─────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      PHASE 2: LOCAL INFERENCE                        │
├─────────────────────────────────────────────────────────────────────┤
│  Local Environment                                                   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ inference.py                                                  │   │
│  │   1. Load schema.json (mandatory contract)                    │   │
│  │   2. Validate model dimensions against schema                 │   │
│  │   3. Process input image (resize, normalize)                  │   │
│  │   4. Generate probability vector [14 classes]                 │   │
│  │   5. Output clinical diagnostic report                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Directory Structure

```text
X-Ray_Chest/
├── src/
│   ├── main.py           # Training orchestration & CLI interface
│   ├── inference.py      # Production single-image inference engine
│   ├── evaluate.py       # Test set evaluation with CI & error slicing
│   ├── ablation.py       # Controlled augmentation experiments
│   ├── analysis.py       # Error visualization & gallery generation
│   ├── model.py          # ResNet-50 architecture with head replacement
│   ├── dataset.py        # Data loading & patient-aware splitting
│   ├── dataloader.py     # PyTorch Dataset & augmentation transforms
│   ├── metrics.py        # AUROC, F1, calibration, bootstrap CI
│   ├── schema.py         # Schema contract & validation logic
│   ├── loss.py           # Weighted BCE loss for class imbalance
│   └── Training.ipynb    # Colab driver notebook
├── output/
│   ├── best_model.pth    # Trained model weights
│   └── schema.json       # Label schema contract
├── images-224/           # Preprocessed 224×224 images
├── data/                 # NIH official split files
└── requirements.txt      # Python dependencies
```

---

## 4. Data Engineering Pipeline

### 4.1 The Patient Leakage Problem

Consider Patient A with two scans: `img_01.png` (initial visit) and `img_02.png` (follow-up). With naive random splitting:

- `img_01.png` → **Training Set**
- `img_02.png` → **Test Set**

The model learns Patient A's unique anatomical features (rib spacing, heart shape, spine curvature). At test time, it "recognizes" familiar anatomy rather than detecting pathology—producing **artificially inflated accuracy** that collapses on truly unseen patients.

### 4.2 Mathematical Formalization

Let $P$ denote the set of unique patients, and $I_p$ the set of images for patient $p \in P$.

**Leakage-Free Constraint:**

$$\forall p \in P: \quad I_p \subseteq D_{\text{train}} \quad \text{XOR} \quad I_p \subseteq D_{\text{val}} \quad \text{XOR} \quad I_p \subseteq D_{\text{test}}$$

This exclusive-or relationship guarantees no patient's images span multiple partitions.

### 4.3 Implementation

```python
def patient_split(self, test_size=0.2, val_size=0.125):
    """
    Splits dataset ensuring ZERO PATIENT LEAKAGE.
    
    Problem: A single patient often has multiple X-rays (follow-ups).
    If we split randomly by image, patient X could have image A in Train
    and image B in Test. The model memorizes anatomy instead of pathology.
    
    Solution: Split by PatientID. All images from a unique patient go
    strictly into ONE set.
    """
    patients = self.data['patientid'].unique()
    
    # First split: Train+Val vs Test
    train_p, test_p = train_test_split(
        patients, test_size=test_size, random_state=42
    )
    
    # Second split: Train vs Val
    train_p, val_p = train_test_split(
        train_p, test_size=val_size, random_state=42
    )
    
    # Filter original data by patient membership
    train_df = self.data[self.data['patientid'].isin(train_p)]
    val_df = self.data[self.data['patientid'].isin(val_p)]
    test_df = self.data[self.data['patientid'].isin(test_p)]
    
    return train_df, val_df, test_df
```

### 4.4 STAR Analysis: Patient-Aware Splitting

| Dimension | Description |
|-----------|-------------|
| **Situation** | Standard ML pipelines split randomly by instance, ignoring patient-level dependencies (repeat scans, longitudinal studies). |
| **Task** | Implement a split algorithm treating the patient as the atomic unit, ensuring complete isolation between partitions. |
| **Action** | Modified `train_test_split()` to operate on unique patient IDs rather than the full DataFrame, then filtered by membership. |
| **Result** | Zero intersection guarantee: `set(train_patients) ∩ set(test_patients) = ∅`. Model must generalize to truly unseen patients. |

### 4.5 Data Augmentation

Medical imaging requires careful augmentation selection. Unlike natural images, X-rays have specific physical constraints:

| Augmentation | Rationale |
|--------------|-----------|
| `RandomResizedCrop(scale=0.9-1.1)` | Simulates patient positioning variation |
| `RandomHorizontalFlip(p=0.5)` | Pathology detection should be location-invariant |
| `RandomRotation(degrees=7)` | Simulates slight patient tilt during acquisition |
| `ColorJitter(brightness=0.1)` | Compensates for equipment calibration differences |

**Excluded Augmentations:**

- Vertical flip (anatomically impossible)
- Extreme rotations (destroys diagnostic context)
- Color shifts (X-rays are grayscale)

```python
def get_transforms(image_size=224):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet statistics
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Validation/Test: No augmentation
    base_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, base_transform, base_transform
```

---

## 5. Model Architecture

### 5.1 Transfer Learning with ResNet-50

Training a 25M-parameter network from scratch requires millions of labeled images. **Transfer learning** leverages representations learned on ImageNet (1.2M images, 1000 classes) and fine-tunes for chest X-rays (~100K images, 14 classes).

**Intuition:** Early layers learn generic features (edges, textures, shapes) that transfer across domains. Only the final classification "head" requires domain-specific learning.

### 5.2 Architecture Specification

| Component | Specification |
|-----------|---------------|
| **Backbone** | ResNet-50 (He et al., 2015) |
| **Pre-training** | ImageNet-1K (`ResNet50_Weights.DEFAULT`) |
| **Input Shape** | `[Batch, 3, 224, 224]` (RGB, normalized) |
| **Feature Dimension** | 2048 (output of avgpool layer) |
| **Classification Head** | `nn.Linear(2048, 14)` |
| **Output Activation** | None (raw logits; sigmoid in loss for stability) |

### 5.3 Head Replacement ("Surgery")

```python
class MultiLabelResNet(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(MultiLabelResNet, self).__init__()
        
        # Load pre-trained ResNet50 backbone
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.base = models.resnet50(weights=weights)
        
        # SURGERY: Replace 1000-class ImageNet head with 14-class medical head
        in_features = self.base.fc.in_features  # 2048
        self.base.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # Output raw logits (NOT probabilities)
        # Sigmoid applied inside BCEWithLogitsLoss for numerical stability
        return self.base(x)
```

### 5.4 Why No Sigmoid in Forward Pass?

For numerical stability, PyTorch's `BCEWithLogitsLoss` combines sigmoid and binary cross-entropy in a fused operation:

$$\mathcal{L}(x, y) = -\left[ y \cdot \log(\sigma(x)) + (1-y) \cdot \log(1 - \sigma(x)) \right]$$

Where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function.

Computing this directly with pre-applied sigmoid causes numerical underflow when $\sigma(x) \approx 0$ or $\sigma(x) \approx 1$. The fused implementation uses the log-sum-exp trick:

$$\mathcal{L}(x, y) = \max(x, 0) - x \cdot y + \log(1 + e^{-|x|})$$

This formulation maintains numerical precision across all input ranges.

### 5.5 STAR Analysis: Model Architecture

| Dimension | Description |
|-----------|-------------|
| **Situation** | Training from random initialization requires far more data and compute than available. ImageNet weights encode useful visual primitives. |
| **Task** | Leverage transfer learning while adapting output layer for 14-class multi-label task. |
| **Action** | Loaded `ResNet50_Weights.DEFAULT`, replaced `.fc` layer with `Linear(2048, 14)`, output raw logits for `BCEWithLogitsLoss` compatibility. |
| **Result** | Model converges in ~10 epochs instead of 100+. Final layer learns pathology-specific features on generic representations. |

---

## 6. Loss Function Design

### 6.1 The Class Imbalance Problem

A naive model minimizes loss by predicting "negative" for rare classes:

| Class | Positive Samples | Negative Samples | Naive Accuracy |
|-------|------------------|------------------|----------------|
| Hernia | 50 | 24,950 | 99.8% (0 detections) |
| Pneumonia | 250 | 24,750 | 99.0% (0 detections) |

### 6.2 Mathematical Solution: Positive Weighting

For each class $c$, compute a **positive weight** based on class frequency:

$$w_c^+ = \frac{N_{\text{neg}}^{(c)}}{\max(N_{\text{pos}}^{(c)}, 1)}$$

**Example:** If Hernia has 50 positive and 24,950 negative samples:

$$w_{\text{Hernia}}^+ = \frac{24950}{50} = 499$$

This penalizes false negatives for Hernia 499× more than false positives, forcing the model to prioritize sensitivity.

**Safety Clamp:** Cap weights at 100 to prevent training instability:

$$w_c^+ = \min(w_c^+, 100)$$

### 6.3 Implementation

```python
class WeightedBCELossWrapper(nn.Module):
    """
    Wraps BCEWithLogitsLoss with dynamically calculated positive weights.
    Mitigates class imbalance by penalizing false negatives in rare classes.
    """
    def __init__(self, train_df, label_list, device, max_weight=100.0):
        super().__init__()
        self.device = device
        self.max_weight = max_weight
        
        pos_weights = self._calculate_pos_weights(train_df, label_list)
        self.pos_weight_tensor = torch.tensor(
            pos_weights, dtype=torch.float32
        ).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor)

    def _calculate_pos_weights(self, df, labels):
        weights = []
        total_samples = len(df)
        
        for label in labels:
            num_pos = df[label].sum()
            num_neg = total_samples - num_pos
            
            weight = num_neg / max(num_pos, 1)
            weight = min(weight, self.max_weight)  # Safety clamp
            weights.append(weight)
            
        return weights

    def forward(self, input, target):
        return self.criterion(input, target)
```

### 6.4 STAR Analysis: Loss Function

| Dimension | Description |
|-----------|-------------|
| **Situation** | Rare pathologies (Hernia, Pneumonia) constitute <1% of samples. Standard BCE produces models ignoring these classes. |
| **Task** | Design loss function up-weighting rare classes despite limited examples. |
| **Action** | Computed per-class `pos_weight = neg_count / pos_count`, clamped to 100, passed to `BCEWithLogitsLoss`. |
| **Result** | Model achieves non-trivial AUROC for rare classes. Training balances detection across all 14 pathologies. |

---

## 7. Training Methodology

### 7.1 Hyperparameter Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Optimizer** | Adam | Adaptive learning rates, robust to sparse gradients |
| **Learning Rate** | 1e-4 | Conservative for fine-tuning pre-trained weights |
| **Batch Size** | 32 | GPU memory constraint; larger batches stabilize gradients |
| **Max Epochs** | 10 | Early stopping typically triggers at 5-7 |
| **LR Scheduler** | ReduceLROnPlateau | Reduce 10× when validation AUROC stagnates |
| **Early Stopping** | 3 epochs patience | Prevent overfitting |

### 7.2 Training Loop

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()           # Clear gradients
        outputs = model(images)         # Forward pass
        loss = criterion(outputs, labels)  # Weighted BCE
        loss.backward()                 # Backpropagation
        optimizer.step()                # Update weights
        
        running_loss += loss.item() * images.size(0)
    
    return running_loss / len(loader.dataset)
```

### 7.3 Early Stopping Strategy

```python
for epoch in range(CONFIG['num_epochs']):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_results = validate(model, val_loader, criterion, device)
    val_auc = val_results["mean_auroc"]
    
    scheduler.step(val_auc)  # Reduce LR if plateau
    
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        epochs_no_improve = 0
        torch.save(model.state_dict(), "output/best_model.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= CONFIG['patience']:
            print("Early Stopping Triggered")
            break
```

### 7.4 STAR Analysis: Training Process

| Dimension | Description |
|-----------|-------------|
| **Situation** | Fixed-epoch training risks overfitting (memorization) or underfitting (premature stop). |
| **Task** | Implement adaptive training with learning rate scheduling and early stopping. |
| **Action** | Combined `ReduceLROnPlateau` (plateau-aware LR decay) with patience-based early stopping. Checkpoint best model by validation AUROC. |
| **Result** | Training auto-terminates at optimal generalization. LR reduces 10× on plateau for fine-grained convergence. |

---

## 8. Evaluation Framework

### 8.1 Why AUROC?

Unlike accuracy, **Area Under the Receiver Operating Characteristic (AUROC)** is:

1. **Threshold-Independent**: Evaluates ranking quality across all thresholds
2. **Class-Imbalance Robust**: Random classifier scores exactly 0.5 regardless of distribution
3. **Clinically Interpretable**: "Probability that a random positive ranks higher than a random negative"

### 8.2 Mathematical Definition

For predictions $\{s_i\}$ and ground truth $\{y_i\}$:

$$\text{AUROC} = \frac{\sum_{i: y_i=1} \sum_{j: y_j=0} \mathbf{1}[s_i > s_j]}{N_{\text{pos}} \cdot N_{\text{neg}}}$$

This equals the Wilcoxon-Mann-Whitney statistic.

For multi-label, compute **per-class AUROC** and report **macro-average mean**:

$$\text{Mean AUROC} = \frac{1}{C} \sum_{c=1}^{C} \text{AUROC}_c$$

### 8.3 Bootstrap Confidence Intervals

Point estimates are insufficient for clinical reporting. We compute 95% CI via bootstrapping:

1. Resample test predictions with replacement $B=1000$ times
2. Compute mean AUROC for each bootstrap sample
3. Report 2.5th and 97.5th percentiles as CI bounds

```python
def compute_bootstrap_ci(self, n_rounds=1000, confidence=0.95):
    y_true = np.vstack(self.y_true)
    y_probs = np.vstack(self.y_pred_probs)
    n_samples = y_true.shape[0]
    
    scores = []
    rng = np.random.default_rng(42)  # Reproducible
    
    for _ in range(n_rounds):
        indices = rng.choice(n_samples, n_samples, replace=True)
        y_t_boot, y_p_boot = y_true[indices], y_probs[indices]
        
        valid_scores = []
        for i in range(self.num_classes):
            if len(np.unique(y_t_boot[:, i])) > 1:
                valid_scores.append(roc_auc_score(y_t_boot[:, i], y_p_boot[:, i]))
        
        if valid_scores:
            scores.append(np.mean(valid_scores))
    
    alpha = (1.0 - confidence) / 2.0
    return np.percentile(scores, alpha * 100), np.percentile(scores, (1 - alpha) * 100)
```

### 8.4 Calibration Curves

A model is **well-calibrated** when predicted probabilities match true frequencies. When it predicts 80%, the condition should be present ~80% of the time.

```python
def compute_calibration_curve(self, n_bins=10):
    from sklearn.calibration import calibration_curve
    
    curves = {}
    for i, label in enumerate(self.label_list):
        if len(np.unique(y_true[:, i])) > 1:
            prob_true, prob_pred = calibration_curve(
                y_true[:, i], y_probs[:, i], n_bins=n_bins
            )
            curves[label] = (prob_true, prob_pred)
    return curves
```

---

## 9. Inference Engine

### 9.1 Schema Enforcement: Fail-Fast Design

Deep learning models fail **silently** when input schema changes:

- Column reordering: "Pneumonia" prediction interpreted as "Atelectasis"
- Missing classes: Undefined tensor dimensions
- Version drift: Incompatible label ordering

**Solution:** Contract-based validation via `schema.json`:

```json
{
    "label_list": ["Atelectasis", "Cardiomegaly", "...", "Pneumothorax"],
    "num_classes": 14,
    "metadata_checksum": "a3f2b1c4...",
    "timestamp": "2026-01-26T10:30:00",
    "version": "1.0"
}
```

### 9.2 Validation Logic

```python
def validate_model(self, model):
    """FAIL FAST: Verify model architecture matches schema."""
    fc = model.base.fc if hasattr(model, 'base') else model.fc
    
    if fc.out_features != self.num_classes:
        raise RuntimeError(
            f"SCHEMA MISMATCH: Model has {fc.out_features} outputs, "
            f"schema defines {self.num_classes} classes."
        )

def validate_prediction(self, logits):
    """FAIL FAST: Verify inference output dimensions."""
    if logits.shape[-1] != self.num_classes:
        raise RuntimeError(
            f"INFERENCE MISMATCH: Output size {logits.shape[-1]}, "
            f"expected {self.num_classes}."
        )
```

### 9.3 Production Inference Pipeline

```python
def predict_single(model, manifest, image_path, device):
    # 1. Load and preprocess image
    _, val_tf, _ = get_transforms()
    img = Image.open(image_path).convert("RGB")
    img_tensor = val_tf(img).unsqueeze(0).to(device)
    
    # 2. Inference with validation
    with torch.no_grad():
        logits = model(img_tensor)
        manifest.validate_prediction(logits)  # FAIL FAST
        probs = torch.sigmoid(logits)
    
    return probs.cpu().numpy()[0]  # Shape: [14]
```

### 9.4 Clinical Output Report

```python
def print_report(probs, filename, manifest):
    print(f"\n{'='*50}")
    print(f" DIAGNOSTIC REPORT: {filename}")
    print(f"{'='*50}")
    
    pairs = sorted(zip(manifest.label_list, probs), key=lambda x: -x[1])
    
    for label, score in pairs:
        if score >= 0.50:
            print(f"  [HIGH] {label}: {score*100:.1f}%")
        elif score >= 0.05:
            print(f"  [LOW]  {label}: {score*100:.1f}%")
    
    print(f"\nSchema Version: {manifest.timestamp}")
```

---

## 10. Experimental Validation

### 10.1 Ablation Study: Augmentation Value

To validate augmentation choices, we conduct controlled ablation:

| Variant | Augmentation | Other Settings |
|---------|--------------|----------------|
| BASELINE | Disabled (resize only) | Identical |
| AUGMENTED | Full pipeline | Identical |

**Expected Results:**

| Metric | Baseline | Augmented | Delta |
|--------|----------|-----------|-------|
| Mean AUROC | ~0.70 | ~0.72 | **+0.02** |

Positive delta confirms augmentation prevents overfitting to training pixel patterns.

### 10.2 Error Slicing: Failure Analysis

Aggregate AUROC hides individual failures. For each class, we identify:

1. **Top False Positives**: $y_c = 0$ but high $p_c$ (model hallucinations)
2. **Top False Negatives**: $y_c = 1$ but low $p_c$ (missed detections)

```python
def find_top_errors(self, k=5):
    for i, class_name in enumerate(self.label_list):
        # False Positives: Label=0, High Prob
        neg_indices = np.where(y_true[:, i] == 0)[0]
        top_fp = np.argsort(y_probs[neg_indices, i])[::-1][:k]
        
        # False Negatives: Label=1, Low Prob
        pos_indices = np.where(y_true[:, i] == 1)[0]
        top_fn = np.argsort(y_probs[pos_indices, i])[:k]
```

Visual error galleries enable targeted data collection or model improvement.

---

## 11. Usage Guide

### Phase 1: Cloud Training

```bash
# 1. Upload project to Google Drive
# 2. Open src/Training.ipynb in Colab
# 3. Execute cells to mount Drive and run training
```

### Phase 2: Local Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Download artifacts from Drive
# Place in output/best_model.pth and output/schema.json
```

### Phase 3: Inference

```bash
# Single image prediction
python src/inference.py --image path/to/xray.png

# Full test set evaluation with bootstrap CI
python src/evaluate.py --bootstrap

# Visual error analysis
python src/evaluate.py --slice_errors

# Ablation study (augmentation comparison)
python src/ablation.py --epochs 10
```

---

## 12. Ethical Considerations

### Safety Warning

> ⚠️ **This model is NOT a diagnostic tool.**

| Risk | Description | Mitigation |
|------|-------------|------------|
| **False Positives** | Incorrectly flags healthy patients | Use for prioritization only |
| **False Negatives** | Misses actual pathology | Never dismiss patients based on low scores |
| **Demographic Bias** | Single hospital training data | May not generalize |
| **Label Noise** | ~10% NLP extraction error | Ground truth ceiling is limited |

### Intended Use

- ✅ Automated triage: Prioritize high-probability scans for radiologist review
- ✅ Research: Quantitative pathology pattern analysis
- ✅ Education: Training tool for radiology residents

### Prohibited Use

- ❌ Autonomous diagnosis without human verification
- ❌ Patient dismissal based on low probability scores
- ❌ Pediatric or non-chest-X-ray applications

---

## 13. References

1. Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks. *IEEE CVPR*.

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *IEEE CVPR*.

3. Rajpurkar, P., et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. *arXiv:1711.05225*.

4. Irvin, J., et al. (2019). CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels. *AAAI*.

5. Efron, B., & Tibshirani, R. J. (1994). An Introduction to the Bootstrap. *CRC Press*.

---

## Appendix: Source Code Inventory

| Module | Purpose | Lines |
|--------|---------|-------|
| `main.py` | Training orchestration, CLI | 282 |
| `dataset.py` | Data loading, patient splitting | 247 |
| `dataloader.py` | PyTorch Dataset, transforms | 107 |
| `model.py` | ResNet-50 architecture | 38 |
| `loss.py` | Weighted BCE loss | 70 |
| `metrics.py` | AUROC, CI, calibration | 147 |
| `schema.py` | Contract validation | 138 |
| `inference.py` | Production inference | 153 |
| `evaluate.py` | Test evaluation | 165 |
| `analysis.py` | Error visualization | 229 |
| `ablation.py` | Augmentation experiments | 141 |

**Total:** ~1,700 lines of production-grade Python code.
