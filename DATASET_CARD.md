# Dataset Card: Chest X-Ray (NIH / Kaggle Adaptation)

## 1. Dataset Overview

**Name**: Chest X-Ray (NIH / Kaggle Adaptation)
**Version**: 1.0 (Schema-Enforced)
**Primary Modality**: Grayscale Radiography (224x224px)
**Task**: Multi-Label Classification (14 Pathologies)

## 2. Provenance & Rights

- **Source**: NIH Clinical Center (Wang et al., 2017)
- **License**: [Please verify specific Kaggle/NIH license terms]
- **Modifications**:
  - Resized to 224x224.
  - Standardized Metadata via `DatasetManifest`.
  - Strict Patient-Level Splitting.

### The Problem

A single patient often has multiple scans (e.g., initial visit + follow-up).

- If Patient A has scans `img_01.png` and `img_02.png`.
- And we split randomly by image.
- `img_01` goes to TRAIN, `img_02` goes to TEST.
- **Result**: The model memorizes Patient A's ribcage structure instead of learning to detect "Pneumonia". This leads to artificially inflated accuracy that collapses on new patients.

### The Contract

1. **Atomic Unit**: The Patient.
2. **Constraint**: All images associated with `PatientID_X` must reside EXCLUSIVELY in one partition (Train OR Val OR Test).
3. **Implementation**: Enforced in `src/dataset.py:patient_split()`.
4. **Verification**: The `schema.json` (future extension) or split logs must verify zero intersection of PatientIDs between sets.

## 4. Schema & Labels

**Schema Version Control**: Managed by `schema.json` (Required for Inference).

**Canonical Label Order**:

1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Effusion
6. Emphysema
7. Fibrosis
8. Hernia
9. Infiltration
10. Mass
11. Nodule
12. Pleural_Thickening
13. Pneumonia
14. Pneumothorax

## 5. Known Limitations & Biases

- **Class Imbalance**: "Hernia" and "Pneumonia" are significantly underrepresented compared to "Infiltration".
- **Label Noise**: Ground truth labels were NLP-extracted from radiology reports, not biopsy-confirmed. Estimated error rate ~10%.
- **Population Bias**: Data collected from a specific hospital system (NIH Clinical Center); may not generalize to populations with different equipment or demographic distributions.
