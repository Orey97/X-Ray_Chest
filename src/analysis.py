
import torch
import numpy as np
import os
from tqdm import tqdm

class ErrorSlicer:
    """
    Identifies specific samples where the model fails most confidently.
    Useful for qualitative error analysis.
    """
    def __init__(self, model, dataloader, manifest, device):
        self.model = model
        self.dataloader = dataloader
        self.manifest = manifest
        self.device = device
        
    def find_top_errors(self, k=5):
        """
        Scan the entire dataset to find Top-K False Positives and False Negatives per class.
        Returns: Dict tree structure with error metadata.
        """
        self.model.eval()
        
        # We need to track: [Prob, Label, ImagePath]
        # Since standard dataloader might not yield paths, we will assume 
        # the dataset returns (image, label, path) OR we reconstruct indices.
        # Check dataloader structure: dataset.py returns (img, tokens, patientid, imagepath)
        # Wait, dataloader.py's ChestXRayDataset.__getitem__ returns (image, label). 
        # We need to Modify it or access dataset's internal list by index.
        # For Phase 4, let's assume we map by index.
        
        # Store all preds: List of (prob_vector, label_vector, batch_start_index)
        all_probs = []
        all_labels = []
        
        print("[INFO] Scanning dataset for error slicing...")
        with torch.no_grad():
            for imgs, labels in tqdm(self.dataloader, desc="Scanning"):
                imgs = imgs.to(self.device)
                logits = self.model(imgs)
                probs = torch.sigmoid(logits)
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
        # Concatenate
        y_probs = np.vstack(all_probs)
        y_true = np.vstack(all_labels)
        
        results = {}
        
        dataset_ref = self.dataloader.dataset
        # Depending on implementation, dataset_ref might be the subset or original.
        # We assume dataset_ref has a way to get path by index.
        # If it's a Subset, we need to map indices. 
        # Ideally, we used our custom Split returning DataFrames.
        # ChestXRayDataset wraps a DF.
        
        df = dataset_ref.dataframe # Access internal DF used in dataloader.py
        
        for i, class_name in enumerate(self.manifest.label_list):
            # 1. False Positives: Label=0, High Prob
            # Mask: where label is 0
            neg_indices = np.where(y_true[:, i] == 0)[0]
            if len(neg_indices) > 0:
                # Get probs for these negatives
                neg_probs = y_probs[neg_indices, i]
                # Sort descending (Highest prob first)
                top_fp_idx_local = np.argsort(neg_probs)[::-1][:k]
                top_fp_idx_global = neg_indices[top_fp_idx_local]
                
                fp_list = []
                for idx in top_fp_idx_global:
                     # Get Path
                     row = df.iloc[idx]
                     path = row['image'] if 'image' in row else f"Idx_{idx}"
                     prob = y_probs[idx, i]
                     fp_list.append({"path": path, "prob": float(prob)})
            else:
                fp_list = []

            # 2. False Negatives: Label=1, Low Prob
            pos_indices = np.where(y_true[:, i] == 1)[0]
            if len(pos_indices) > 0:
                 pos_probs = y_probs[pos_indices, i]
                 # Sort ascending (Lowest prob first)
                 top_fn_idx_local = np.argsort(pos_probs)[:k]
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
        Generates a visual gallery of top errors.
        Requires matplotlib and PIL/cv2.
        """
        try:
            import matplotlib.pyplot as plt
            from PIL import Image
        except ImportError:
            print("[WARNING] matplotlib or PIL not installed. Skipping visualization.")
            return

        print(f"[INFO] Generating Error Gallery: {save_path}")
        
        # Determine grid size. Let's show Top 3 FP and Top 3 FN for each class.
        # But that might be huge. Let's just create separate images per class or one big PDF?
        # For simplicity: One distinct image per class.
        # BETTER: Just plotting the first class with errors as a demo, or one row per class.
        
        # Config: Max 5 classes, Top 2 FP per class to keep it readable in one image.
        
        selected_classes = list(results.keys())[:5] # limit to 5 classes
        n_classes = len(selected_classes)
        cols = 4 # 2 FP, 2 FN
        
        fig, axes = plt.subplots(nrows=n_classes, ncols=cols, figsize=(15, 3*n_classes))
        
        # Handle single class case (axes is 1D array)
        if n_classes == 1:
            axes = [axes]
            
        fig.suptitle("Error Analysis Gallery: Top False Positives & Negatives", fontsize=16)
        
        for i, class_name in enumerate(selected_classes):
            row_data = results[class_name]
            
            # Columns 0,1: False Positives
            for j in range(2):
                ax = axes[i][j] if n_classes > 1 else axes[j]
                if j < len(row_data["top_fp"]):
                    item = row_data["top_fp"][j]
                    self._show_image(ax, item["path"], f"FP: {class_name}\nProb: {item['prob']:.2f}")
                else:
                    ax.axis('off')
                    
            # Columns 2,3: False Negatives
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
        try:
            import matplotlib.pyplot as plt
            from PIL import Image
            
            # Check if path is absolute or needs resolution
            # In our setup, 'path' from DataFrame might be just filename. 
            # We need the base image directory. But wait, datasets usually have abs path or we assumed it.
            # In dataset.py, we only stored filename? 
            # Let's check Main.py -> dataset_manager.data
            
            # NOTE: We don't have the image_dir here easily unless passed.
            # We will rely on absolute paths if possible, or try to guess.
            # Actually, `ChestXRayDataset` builds the path in `__getitem__`.
            # Let's assume the path in result is what we need or we try to find it.
            
            # Workaround: Use the root of the test_env if path is relative
            # But we don't know the root here. 
            # Let's hope the path is valid. If not, placeholder.
            
            # Simple heuristic: If not exists, try finding it in current dir/test_env/images
            display_path = path
            if not os.path.exists(display_path):
                 # Try typical locations
                 candidates = [
                     os.path.join("test_env", "images", path),
                     os.path.join("..", "test_env", "images", path)
                 ]
                 for c in candidates:
                     if os.path.exists(c):
                         display_path = c
                         break
            
            if os.path.exists(display_path):
                img = Image.open(display_path).convert("RGB")
                ax.imshow(img, cmap="gray")
            else:
                ax.text(0.5, 0.5, "Image Not Found", ha='center')
                
            ax.set_title(title, fontsize=8, color='red')
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, "Error Load", ha='center')
            ax.axis('off')
            print(f"[WARNING] Viz Error: {e}")
