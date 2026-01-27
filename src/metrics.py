import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import torch

class ClinicalMetricContainer:
    """
    Unified Metric Calculation Module.
    Decouples metric logic from Training, Validation, and Inference.
    """
    def __init__(self, label_list):
        self.label_list = label_list
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred_probs = []

    def update(self, y_true_batch, y_logits_batch):
        """
        Accumulate batch results.
        y_true_batch: [Batch, NumClasses] (0 or 1)
        y_logits_batch: [Batch, NumClasses] (Raw Logits)
        """
        # Detach from graph if tensor
        if isinstance(y_true_batch, torch.Tensor):
            y_true_batch = y_true_batch.cpu().detach().numpy()
        if isinstance(y_logits_batch, torch.Tensor):
            y_logits_batch = y_logits_batch.cpu().detach().numpy()
            
        # Convert Logits -> Probabilities Here (Centralized Logic)
        y_probs = 1 / (1 + np.exp(-y_logits_batch))
        
        self.y_true.append(y_true_batch)
        self.y_pred_probs.append(y_probs)

    def compute(self):
        """
        Returns dictionary of metrics.
        - mean_auroc (Optim)
        - f1_macro (Diagnostic)
        - per_class_auroc (Diagnostic)
        """
        if not self.y_true:
            return {}

        y_true = np.vstack(self.y_true)
        y_probs = np.vstack(self.y_pred_probs)
        
        metrics = {}
        
        # 1. AUROC (Per Class & Mean)
        # Handle missing classes (e.g., batch/dataset where a disease never appears)
        class_aucs = {}
        valid_scores = []
        
        for i, label in enumerate(self.label_list):
            try:
                # Require at least one positive and one negative sample
                if len(np.unique(y_true[:, i])) > 1:
                    score = roc_auc_score(y_true[:, i], y_probs[:, i])
                    class_aucs[label] = score
                    valid_scores.append(score)
                else:
                    class_aucs[label] = float('nan')
            except ValueError:
                class_aucs[label] = float('nan')

        metrics["per_class_auroc"] = class_aucs
        metrics["mean_auroc"] = np.mean(valid_scores) if valid_scores else 0.0
        
        # 2. F1 Score (Macro) - purely for reference, using 0.5 threshold
        y_pred_binary = (y_probs > 0.5).astype(int)
        metrics["f1_macro"] = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
        
        return metrics

    def compute_bootstrap_ci(self, n_rounds=1000, confidence=0.95):
        """
        Computes Bootstrap Confidence Intervals for Mean AUROC.
        Returns: (low, high) tuple.
        """
        if not self.y_true:
            return (0.0, 0.0)

        y_true = np.vstack(self.y_true)
        y_probs = np.vstack(self.y_pred_probs)
        n_samples = y_true.shape[0]
        
        scores = []
        rng = np.random.default_rng(42) # Fixed seed for reproducibility of CI
        
        print(f"[METRICS] Bootstrapping {n_rounds} rounds for CI calculation...")
        
        # We only bootstrap the MEAN AUROC for now to be fast
        for _ in range(n_rounds):
            indices = rng.choice(n_samples, n_samples, replace=True)
            y_t_boot = y_true[indices]
            y_p_boot = y_probs[indices]
            
            # fast calculation of mean auroc
            valid_scores = []
            for i in range(self.num_classes): # Optimization: Pre-calc num_classes
                 if len(np.unique(y_t_boot[:, i])) > 1:
                    try:
                        valid_scores.append(roc_auc_score(y_t_boot[:, i], y_p_boot[:, i]))
                    except ValueError:
                        pass
            
            if valid_scores:
                scores.append(np.mean(valid_scores))
        
        if not scores:
            return (0.0, 0.0)
            
        alpha = (1.0 - confidence) / 2.0
        low = np.percentile(scores, alpha * 100)
        high = np.percentile(scores, (1.0 - alpha) * 100)
        
        return (low, high)

    def compute_calibration_curve(self, n_bins=10):
        """
        Computes reliability data for plotting calibration curves.
        Returns dict: {label: (prob_true, prob_pred)}
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
        return len(self.label_list)
