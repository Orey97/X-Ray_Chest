# Model Card: Multi-Label Chest X-Ray ResNet

## 1. Model Details

- **Architecture**: ResNet50 (ImageNet Pretrained Backbone)
- **Heads**: 14-Class Linear Output (Sigmoid Activation)
- **Input**: 224x224 RGB Images (Grayscale replicated to 3 channels)
- **Training Framework**: PyTorch / Google Colab Hybrid

## 2. Intended Use (Scope)

- **Primary Goal**: Automated Triage & Prioritization.
  - *Example*: Flagging scans with high probability of "Pneumothorax" to move them to the top of a Radiologist's worklist.
- **Target User**: Clinical Researchers, Healthcare IT Systems (Worklist Orchestration).

## 3. Non-Goals (Out of Scope)

> [!WARNING]
> This model is **NOT** a diagnostic tool.

- **NOT for Autonomous Diagnosis**: The model output should never be used to dispense treatment without human verification.
- **NOT for "Clean Bill of Health"**: A low probability score does not guarantee the absence of pathology (Risk of False Negatives).
- **NOT for Pediatric/Non-Standard use**: Validated only on adult chest radiographs from the NIH dataset.

## 4. Performance Metrics

### Optimization Metric

- **Mean AUROC**: The primary metric for model selection / early stopping. Chosen for its robustness to threshold variation.

### Diagnostic Metrics

- **Per-Class AUROC**: Reported individually to highlight performance disparities (e.g., Cardiomegaly vs Hernia).
- **F1 Score**: Monitored but not optimized directly.

## 5. Technical Constraints

- **Schema Lock**: Inference requires a matching `schema.json` artifact.
- **Input Specs**: Images must be pre-processed to 224x224. Aspect ratio distortion is currently tolerated but not optimal.

## 6. Ethical Considerations

- **False Positives**: May cause alarm fatigue in clinicians.
- **False Negatives**: May delay critical care if used as a filter. (Mitigation: Use only for prioritization, never for dismissal).
