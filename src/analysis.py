"""
=============================================================================
                    ANALYSIS.PY - Error Analysis & Visualization
=============================================================================

PURPOSE:
    Provides tools for qualitative analysis of model failures.
    Helps identify systematic error patterns for model improvement.

WHY ERROR ANALYSIS MATTERS:

    High AUROC (e.g., 0.80) doesn't mean the model is perfect!
    
    Example: Model achieves 80% AUROC on Pneumonia detection.
    
    Questions that metrics DON'T answer:
    - WHICH 20% does it fail on?
    - Are failures RANDOM or SYSTEMATIC?
    - Are there patient subgroups where it fails more?
    - What do the failure cases look like?
    
    Error Slicing answers these by:
    1. Finding the WORST failures (highest confidence mistakes)
    2. Generating a visual gallery for human review
    3. Enabling qualitative pattern discovery

ERROR TYPES:

    FALSE POSITIVE (FP):
        - Ground truth: Healthy (label = 0)
        - Prediction: Sick (high probability)
        - Model HALLUCINATED a disease
        - Clinical risk: Unnecessary treatment
    
    FALSE NEGATIVE (FN):
        - Ground truth: Sick (label = 1)
        - Prediction: Healthy (low probability)
        - Model MISSED a disease
        - Clinical risk: Delayed treatment

=============================================================================
"""

import torch
import numpy as np
import os
from tqdm import tqdm


class ErrorSlicer:
    """
    Identifies specific samples where the model fails most confidently.
    
    "Most confidently wrong" = Highest confidence in wrong direction
    
    For False Positives: Find cases where label=0 but probability is highest
    For False Negatives: Find cases where label=1 but probability is lowest
    
    These are the samples where the model is MOST WRONG, making them
    valuable for understanding failure modes.
    """
    
    def __init__(self, model, dataloader, manifest, device, image_dir=None):
        """
        Initialize the error slicer.
        
        Args:
            model: Trained neural network
            dataloader: DataLoader with test data
            manifest: Schema with label information
            device: torch.device
            image_dir: Path to images (for visualization)
        """
        self.model = model
        self.dataloader = dataloader
        self.manifest = manifest
        self.device = device
        self.image_dir = image_dir  # Stored for visualization path resolution
        
    def find_top_errors(self, k=5):
        """
        Scan the entire dataset to find Top-K False Positives and 
        False Negatives for each class.
        
        Args:
            k (int): How many errors to find per class (default: 5)
            
        Returns:
            dict: {
                "Pneumonia": {
                    "top_fp": [{"path": "img.png", "prob": 0.95}, ...],
                    "top_fn": [{"path": "img.png", "prob": 0.02}, ...]
                },
                ...
            }
            
        Algorithm:
            1. Run inference on entire dataset
            2. For each class:
               a. Filter to samples with label=0, sort by prob descending → Top FP
               b. Filter to samples with label=1, sort by prob ascending → Top FN
        """
        self.model.eval()
        
        # Accumulate all predictions
        all_probs = []
        all_labels = []
        
        print("[INFO] Scanning dataset for error slicing...")
        with torch.no_grad():
            for imgs, labels in tqdm(self.dataloader, desc="Scanning"):
                imgs = imgs.to(self.device)
                logits = self.model(imgs)
                probs = torch.sigmoid(logits)  # Convert to probabilities
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
        # Stack into arrays
        y_probs = np.vstack(all_probs)  # [N, num_classes]
        y_true = np.vstack(all_labels)   # [N, num_classes]
        
        results = {}
        
        # Access the underlying dataframe for image paths
        dataset_ref = self.dataloader.dataset
        df = dataset_ref.dataframe
        
        # ══════════════════════════════════════════════════════════════════
        # FIND ERRORS FOR EACH CLASS
        # ══════════════════════════════════════════════════════════════════
        for i, class_name in enumerate(self.manifest.label_list):
            
            # -----------------------------------------------------------------
            # FALSE POSITIVES: Label=0, High Probability
            # -----------------------------------------------------------------
            # These are healthy samples that the model thinks are sick.
            # Sort by probability descending (highest first).
            
            neg_indices = np.where(y_true[:, i] == 0)[0]  # All negatives
            if len(neg_indices) > 0:
                neg_probs = y_probs[neg_indices, i]
                top_fp_idx_local = np.argsort(neg_probs)[::-1][:k]  # Top k by prob
                top_fp_idx_global = neg_indices[top_fp_idx_local]
                
                fp_list = []
                for idx in top_fp_idx_global:
                     row = df.iloc[idx]
                     path = row['image'] if 'image' in row else f"Idx_{idx}"
                     prob = y_probs[idx, i]
                     fp_list.append({"path": path, "prob": float(prob)})
            else:
                fp_list = []

            # -----------------------------------------------------------------
            # FALSE NEGATIVES: Label=1, Low Probability
            # -----------------------------------------------------------------
            # These are sick samples that the model thinks are healthy.
            # Sort by probability ascending (lowest first).
            
            pos_indices = np.where(y_true[:, i] == 1)[0]  # All positives
            if len(pos_indices) > 0:
                 pos_probs = y_probs[pos_indices, i]
                 top_fn_idx_local = np.argsort(pos_probs)[:k]  # Bottom k by prob
                 top_fn_idx_global = pos_indices[top_fn_idx_local]
                 
                 fn_list = []
                 for idx in top_fn_idx_global:
                     row = df.iloc[idx]
                     path = row['image'] if 'image' in row else f"Idx_{idx}"
                     prob = y_probs[idx, i]
                     fn_list.append({"path": path, "prob": float(prob)})
            else:
                fn_list = []
                
            results[class_name] = {
                "top_fp": fp_list,
                "top_fn": fn_list
            }
            
        return results

    def save_report(self, results, path):
        """
        Save error analysis as a markdown report.
        
        The report lists:
        - Top False Positives per class (model hallucinations)
        - Top False Negatives per class (missed detections)
        
        Each entry includes the image path and model confidence.
        
        Args:
            results: Output from find_top_errors()
            path: Output file path (.md)
        """
        with open(path, "w") as f:
            f.write("# Error Analysis Report\n")
            f.write("Systematic Failure Analysis (Top-5 Hardest Samples)\n\n")
            
            for label, data in results.items():
                f.write(f"## {label}\n")
                
                f.write("### Top False Positives (Model hallucinations)\n")
                for item in data["top_fp"]:
                    f.write(f"- {item['prob']:.4f} | {item['path']}\n")
                
                f.write("\n### Top False Negatives (Missed detections)\n")
                for item in data["top_fn"]:
                    f.write(f"- {item['prob']:.4f} | {item['path']}\n")
                f.write("\n" + "-"*40 + "\n")
                
        print(f"[ANALYSIS] Report saved to {path}")

    def plot_top_errors(self, results, save_path):
        """
        Generate a visual gallery of top errors.
        
        Creates a grid image showing:
        - Rows: One per class (up to 5 classes)
        - Columns: 2 FP + 2 FN per class
        
        Args:
            results: Output from find_top_errors()
            save_path: Output file path (.png)
        """
        try:
            import matplotlib.pyplot as plt
            from PIL import Image
        except ImportError:
            print("[WARNING] matplotlib or PIL not installed. Skipping visualization.")
            return

        print(f"[INFO] Generating Error Gallery: {save_path}")
        
        # Limit to 5 classes for readability
        selected_classes = list(results.keys())[:5]
        n_classes = len(selected_classes)
        cols = 4  # 2 FP + 2 FN
        
        fig, axes = plt.subplots(nrows=n_classes, ncols=cols, figsize=(15, 3*n_classes))
        
        # Handle single class case
        if n_classes == 1:
            axes = [axes]
            
        fig.suptitle("Error Analysis Gallery: Top False Positives & Negatives", fontsize=16)
        
        for i, class_name in enumerate(selected_classes):
            row_data = results[class_name]
            
            # Columns 0-1: False Positives
            for j in range(2):
                ax = axes[i][j] if n_classes > 1 else axes[j]
                if j < len(row_data["top_fp"]):
                    item = row_data["top_fp"][j]
                    self._show_image(ax, item["path"], f"FP: {class_name}\nProb: {item['prob']:.2f}")
                else:
                    ax.axis('off')
                    
            # Columns 2-3: False Negatives
            for j in range(2):
                ax = axes[i][j+2] if n_classes > 1 else axes[j+2]
                if j < len(row_data["top_fn"]):
                    item = row_data["top_fn"][j]
                    self._show_image(ax, item["path"], f"FN: {class_name}\nProb: {item['prob']:.2f}")
                else:
                    ax.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(save_path)
        plt.close()
        print(f"[INFO] Gallery saved.")

    def _show_image(self, ax, path, title):
        """
        Display an image in a matplotlib axis.
        
        Handles path resolution using stored image_dir.
        
        Args:
            ax: Matplotlib axis object
            path: Image filename or path
            title: Title to display above image
        """
        try:
            import matplotlib.pyplot as plt
            from PIL import Image
            
            # Resolve path: If file doesn't exist, try prepending image_dir
            display_path = path
            if not os.path.exists(display_path) and self.image_dir:
                display_path = os.path.join(self.image_dir, path)
            
            if os.path.exists(display_path):
                img = Image.open(display_path).convert("RGB")
                ax.imshow(img, cmap="gray")
            else:
                ax.text(0.5, 0.5, "Image Not Found", ha='center', va='center')
                
            ax.set_title(title, fontsize=8, color='red')
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, "Error Load", ha='center', va='center')
            ax.axis('off')
            print(f"[WARNING] Viz Error: {e}")
