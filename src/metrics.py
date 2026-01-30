"""
=============================================================================
                    METRICS.PY - Clinical Performance Measurement
=============================================================================

PURPOSE:
    Provides a unified container for computing clinical performance metrics.
    Decouples metric calculation from training/validation/inference loops.

WHY A SEPARATE METRICS MODULE?

    Without this:
        - Training loop computes AUROC one way
        - Validation computes AUROC slightly differently
        - Evaluation script computes AUROC yet another way
        - Bugs and inconsistencies creep in
    
    With ClinicalMetricContainer:
        - ONE implementation of each metric
        - Consistent results across all components
        - Easy to add new metrics in one place

KEY METRICS FOR MEDICAL IMAGING:

    1. AUROC (Area Under ROC Curve) - PRIMARY
       - Measures discrimination: Can model separate sick from healthy?
       - Threshold-independent (evaluates all possible thresholds)
       - Range: 0.5 (random) to 1.0 (perfect)
       - Medical standard: AUROC > 0.7 is "acceptable", > 0.8 is "good"
    
    2. F1 Score (Harmonic Mean of Precision & Recall)
       - Balance between precision (don't over-diagnose) and recall (don't miss cases)
       - Threshold-dependent (requires choosing a cutoff, typically 0.5)
       - Range: 0 to 1
    
    3. Bootstrap Confidence Intervals
       - Statistical rigor: "We're 95% confident AUROC is between 0.72-0.78"
       - Required for medical publications and regulatory submissions

=============================================================================
"""

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import torch


class ClinicalMetricContainer:
    """
    Unified Metric Calculation Module for multi-label classification.
    
    Design Pattern: Accumulator
        - Collects batch predictions during inference
        - Computes metrics once at the end (more efficient)
    
    Usage:
        metrics = ClinicalMetricContainer(label_list)
        
        for batch in dataloader:
            outputs = model(batch)
            metrics.update(labels, outputs)  # Accumulate
            
        results = metrics.compute()  # Final calculation
        print(results["mean_auroc"])
    """
    
    def __init__(self, label_list):
        """
        Initialize the metric container.
        
        Args:
            label_list (list): Ordered list of class names from schema.
                              Used for per-class metric reporting.
        """
        self.label_list = label_list
        self.reset()

    def reset(self):
        """Clear accumulated predictions. Call before new evaluation."""
        self.y_true = []       # Ground truth labels
        self.y_pred_probs = [] # Predicted probabilities (after sigmoid)

    def update(self, y_true_batch, y_logits_batch):
        """
        Accumulate a batch of predictions.
        
        Called once per batch during inference loop.
        
        Args:
            y_true_batch: Ground truth, shape [Batch, NumClasses]
                         Values are 0 or 1
                         
            y_logits_batch: Model outputs, shape [Batch, NumClasses]
                           RAW LOGITS (not probabilities!)
                           We apply sigmoid here to convert to probabilities
        
        Why accept logits instead of probabilities?
            Centralized sigmoid application ensures consistency.
            If caller applied sigmoid, we might double-apply it by accident.
        """
        # Convert tensors to numpy arrays (metrics computed on CPU)
        if isinstance(y_true_batch, torch.Tensor):
            y_true_batch = y_true_batch.cpu().detach().numpy()
        if isinstance(y_logits_batch, torch.Tensor):
            y_logits_batch = y_logits_batch.cpu().detach().numpy()
            
        # ═══════════════════════════════════════════════════════════════════
        # SIGMOID: Convert logits → probabilities
        # ═══════════════════════════════════════════════════════════════════
        # Logits range: (-∞, +∞)
        # Probabilities range: (0, 1)
        # Formula: p = 1 / (1 + exp(-logit))
        #
        y_probs = 1 / (1 + np.exp(-y_logits_batch))
        
        # Append to accumulator lists
        self.y_true.append(y_true_batch)
        self.y_pred_probs.append(y_probs)

    def compute(self):
        """
        Compute all metrics on accumulated predictions.
        
        Returns:
            dict: {
                "mean_auroc": float,      # Average AUROC across all classes
                "per_class_auroc": {...}, # AUROC for each pathology
                "f1_macro": float         # Macro-averaged F1 score
            }
            
        Note on handling missing classes:
            Some diseases may have 0 positive or 0 negative samples.
            AUROC is undefined in this case (can't compute ROC curve).
            We set these to NaN and exclude from mean calculation.
        """
        if not self.y_true:
            return {}  # No data accumulated

        # Stack all batches into single arrays
        y_true = np.vstack(self.y_true)    # Shape: [N, NumClasses]
        y_probs = np.vstack(self.y_pred_probs)  # Shape: [N, NumClasses]
        
        metrics = {}
        
        # ═══════════════════════════════════════════════════════════════════
        # METRIC 1: AUROC (Area Under ROC Curve)
        # ═══════════════════════════════════════════════════════════════════
        # Compute per-class AUROC, then average
        #
        class_aucs = {}
        valid_scores = []
        
        for i, label in enumerate(self.label_list):
            try:
                # AUROC requires at least one positive AND one negative sample
                # If all labels are 0 (or all are 1), AUROC is undefined
                if len(np.unique(y_true[:, i])) > 1:
                    score = roc_auc_score(y_true[:, i], y_probs[:, i])
                    class_aucs[label] = score
                    valid_scores.append(score)
                else:
                    class_aucs[label] = float('nan')  # Undefined
            except ValueError:
                class_aucs[label] = float('nan')  # Error during computation

        metrics["per_class_auroc"] = class_aucs
        metrics["mean_auroc"] = np.mean(valid_scores) if valid_scores else 0.0
        
        # ═══════════════════════════════════════════════════════════════════
        # METRIC 2: F1 Score (Macro Average)
        # ═══════════════════════════════════════════════════════════════════
        # F1 requires binary predictions (not probabilities)
        # We use threshold = 0.5 (standard default)
        #
        y_pred_binary = (y_probs > 0.5).astype(int)
        metrics["f1_macro"] = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
        
        return metrics

    def compute_bootstrap_ci(self, n_rounds=1000, confidence=0.95):
        """
        Compute Bootstrap Confidence Intervals for Mean AUROC.
        
        Bootstrap Method:
            1. Sample N predictions WITH REPLACEMENT from original data
            2. Compute AUROC on this bootstrap sample
            3. Repeat 1000 times to build a distribution
            4. Take 2.5th and 97.5th percentiles as 95% CI
        
        Args:
            n_rounds (int): Number of bootstrap iterations (default: 1000)
            confidence (float): Confidence level (default: 0.95 = 95%)
            
        Returns:
            tuple: (lower_bound, upper_bound) of the confidence interval
            
        Why Bootstrap?
            - No distributional assumptions (non-parametric)
            - Works for any metric, not just those with closed-form CIs
            - Standard in medical imaging literature
        """
        if not self.y_true:
            return (0.0, 0.0)

        y_true = np.vstack(self.y_true)
        y_probs = np.vstack(self.y_pred_probs)
        n_samples = y_true.shape[0]
        
        scores = []
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        
        print(f"[METRICS] Bootstrapping {n_rounds} rounds for CI calculation...")
        
        for _ in range(n_rounds):
            # Sample with replacement (same size as original)
            indices = rng.choice(n_samples, n_samples, replace=True)
            y_t_boot = y_true[indices]
            y_p_boot = y_probs[indices]
            
            # Compute mean AUROC on this bootstrap sample
            valid_scores = []
            for i in range(self.num_classes):
                 if len(np.unique(y_t_boot[:, i])) > 1:
                    try:
                        valid_scores.append(roc_auc_score(y_t_boot[:, i], y_p_boot[:, i]))
                    except ValueError:
                        pass
            
            if valid_scores:
                scores.append(np.mean(valid_scores))
        
        if not scores:
            return (0.0, 0.0)
        
        # ═══════════════════════════════════════════════════════════════════
        # PERCENTILE METHOD: Extract CI bounds
        # ═══════════════════════════════════════════════════════════════════
        # For 95% CI: lower = 2.5th percentile, upper = 97.5th percentile
        #
        alpha = (1.0 - confidence) / 2.0
        low = np.percentile(scores, alpha * 100)
        high = np.percentile(scores, (1.0 - alpha) * 100)
        
        return (low, high)

    def compute_calibration_curve(self, n_bins=10):
        """
        Compute reliability data for plotting calibration curves.
        
        Calibration answers: "When model says 70% probability, is it ACTUALLY
        70% chance of disease?"
        
        Returns:
            dict: {label: (prob_true, prob_pred)} for each class
            
        Interpretation:
            - Perfect calibration: diagonal line (predicted = actual)
            - Under-confidence: curve above diagonal
            - Over-confidence: curve below diagonal
        """
        from sklearn.calibration import calibration_curve
        
        if not self.y_true:
            return {}

        y_true = np.vstack(self.y_true)
        y_probs = np.vstack(self.y_pred_probs)
        
        curves = {}
        for i, label in enumerate(self.label_list):
            if len(np.unique(y_true[:, i])) > 1:
                prob_true, prob_pred = calibration_curve(y_true[:, i], y_probs[:, i], n_bins=n_bins)
                curves[label] = (prob_true, prob_pred)
                
        return curves

    @property
    def num_classes(self):
        """Number of classes (convenience property)."""
        return len(self.label_list)
