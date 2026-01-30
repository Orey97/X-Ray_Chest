"""
=============================================================================
                    ABLATION.PY - Controlled Experiment Runner
=============================================================================

PURPOSE:
    Runs controlled experiments to measure the impact of specific features
    (like data augmentation) on model performance.

WHAT IS AN ABLATION STUDY?

    In medical research, "ablation" means removing something to see what happens.
    In machine learning, we:
    
    1. Train a FULL model (with all features)
    2. Train a VARIANT model (with one feature removed)
    3. Compare performance
    4. Quantify the contribution of that feature
    
    Example from this file:
    - Model A: Trained WITH augmentation
    - Model B: Trained WITHOUT augmentation (ablated)
    - Compare: How much does augmentation improve AUROC?

WHY ABLATION MATTERS:

    Without ablation studies, we can't answer:
    - "Is augmentation actually helping?"
    - "Would a simpler model work just as well?"
    - "Which components are essential vs. optional?"
    
    Ablation provides EVIDENCE-BASED answers to these questions.

USAGE:
    # Run ablation with 5 epochs per variant
    python src/ablation.py --csv_file src/Data_Entry_2017.csv --data_dir images-224 --epochs 5
    
    # Output: ablation_report.md with per-class and aggregate comparisons

=============================================================================
"""

import os
import subprocess
import sys
import torch
import argparse
from schema import DatasetManifest
from model import MultiLabelResNet
from metrics import ClinicalMetricContainer
from dataloader import get_transforms, ChestXRayDataset
from torch.utils.data import DataLoader
from dataset import Dataset as ProjectDataset


def train_variant(name, output_dir, use_aug, args):
    """
    Train a single experimental variant.
    
    Uses subprocess to call main.py with specific configuration.
    This ensures each variant trains in a clean environment.
    
    Args:
        name (str): Human-readable name for this variant (e.g., "BASELINE")
        output_dir (str): Where to save this variant's artifacts
        use_aug (bool): Whether to enable data augmentation
        args: CLI arguments containing csv_file, data_dir, epochs
        
    Returns:
        str: Path to the saved model (best_model.pth)
        
    Raises:
        RuntimeError: If training subprocess fails
    """
    print(f"\n" + "="*50)
    print(f"   RUNNING ABLATION VARIANT: {name}")
    print(f"   Augmentation: {'ENABLED' if use_aug else 'DISABLED'}")
    print("="*50)
    
    # ══════════════════════════════════════════════════════════════════════
    # BUILD COMMAND LINE
    # ══════════════════════════════════════════════════════════════════════
    # We call main.py as a subprocess. This ensures:
    # - Clean Python environment (no state pollution)
    # - Same training logic for all variants
    # - Easy to parallelize in the future
    
    cmd = [
        sys.executable, "src/main.py",  # Use same Python interpreter
        "--csv_file", args.csv_file,
        "--data_dir", args.data_dir,
        "--output_dir", output_dir,
        "--num_epochs", str(args.epochs)
    ]
    
    # Add --no_aug flag if augmentation is disabled
    if not use_aug:
        cmd.append("--no_aug")
        
    print(f"[CMD] {' '.join(cmd)}")
    
    # Run training
    result = subprocess.run(cmd, capture_output=False)  # Stream output to console
    
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for variant {name}. Exit Code: {result.returncode}")
        
    return os.path.join(output_dir, "best_model.pth")


def evaluate_model(model_path, test_loader, device, manifest):
    """
    Evaluate a trained model and return metrics.
    
    Args:
        model_path (str): Path to .pth model file
        test_loader: DataLoader with test data
        device: torch.device
        manifest: Schema for label information
        
    Returns:
        dict: Metrics including mean_auroc and per_class_auroc
              Returns None if model file doesn't exist
    """
    print(f"[EVAL] Loading {model_path}...")
    
    # Handle case where best_model wasn't saved (no improvement)
    if not os.path.exists(model_path):
        fallback = model_path.replace("best_model.pth", "final_model.pth")
        if os.path.exists(fallback):
            print(f"[WARN] Using final_model.pth fallback")
            model_path = fallback
        else:
            return None

    # Load model
    model = MultiLabelResNet(num_classes=manifest.num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Compute metrics
    metrics = ClinicalMetricContainer(manifest.label_list)
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            metrics.update(labels, logits)
            
    return metrics.compute()


def main():
    """
    Main ablation study entry point.
    
    Workflow:
    1. Parse arguments
    2. Train BASELINE variant (no augmentation)
    3. Train AUGMENTED variant (with augmentation)
    4. Load both models and evaluate on same test set
    5. Generate comparison report
    """
    parser = argparse.ArgumentParser(description="Run Ablation Study (Augmentation vs Baseline)")
    parser.add_argument("--csv_file", type=str, default="test_env/data.csv")
    parser.add_argument("--data_dir", type=str, default="test_env/images")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    
    # Create base directory for ablation results
    base_dir = "ablation_results"
    os.makedirs(base_dir, exist_ok=True)
    
    # ══════════════════════════════════════════════════════════════════════
    # TRAIN BOTH VARIANTS
    # ══════════════════════════════════════════════════════════════════════
    
    # Variant 1: BASELINE (no augmentation)
    # This establishes what the model can learn without synthetic variations
    path_baseline = train_variant("BASELINE", os.path.join(base_dir, "baseline"), False, args)
    
    # Variant 2: AUGMENTED (with augmentation)
    # This shows the benefit of data augmentation
    path_aug = train_variant("AUGMENTED", os.path.join(base_dir, "augmented"), True, args)
    
    # ══════════════════════════════════════════════════════════════════════
    # COMPARATIVE EVALUATION
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*50)
    print("   COMPARING RESULTS")
    print("="*50)
    
    # Load schema (should be identical for both variants)
    schema_path = os.path.join(base_dir, "baseline", "schema.json")
    manifest = DatasetManifest.load(schema_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare test data (same data for both evaluations)
    ds = ProjectDataset(args.csv_file)
    ds.clean_column_names()
    label_col = "finding_labels" if "finding_labels" in ds.data.columns else "label"
    ds.clean_labels(label_col)
    ds.one_hot_encode_labels(label_col, explicit_labels=manifest.label_list)
    ds.select_relevant_columns()
    _, _, test_df = ds.patient_split()
    
    _, _, test_tf = get_transforms()
    test_ds = ChestXRayDataset(test_df, args.data_dir, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # Evaluate both models
    res_base = evaluate_model(path_baseline, test_loader, device, manifest)
    res_aug = evaluate_model(path_aug, test_loader, device, manifest)
    
    if not res_base or not res_aug:
        print("[ERROR] Could not load models for comparison.")
        return

    # ══════════════════════════════════════════════════════════════════════
    # GENERATE COMPARISON REPORT
    # ══════════════════════════════════════════════════════════════════════
    
    print(f"\n{'Metric':<20} | {'Baseline':<10} | {'Augmented':<10} | {'Delta':<10}")
    print("-" * 60)
    
    base_auc = res_base["mean_auroc"]
    aug_auc = res_aug["mean_auroc"]
    delta = aug_auc - base_auc
    
    print(f"{'Mean AUROC':<20} | {base_auc:.4f}     | {aug_auc:.4f}     | {delta:+.4f}")
    print("-" * 60)
    
    # Save detailed report as markdown
    with open("ablation_report.md", "w") as f:
        f.write("# Ablation Study Report\n\n")
        f.write(f"**Mean AUROC Delta:** `{delta:+.4f}`\n\n")
        f.write("| Pathology | Baseline | Augmented | Delta |\n")
        f.write("|---|---|---|---|\n")
        
        for label in manifest.label_list:
            b = res_base["per_class_auroc"][label]
            a = res_aug["per_class_auroc"][label]
            d = a - b
            f.write(f"| {label} | {b:.4f} | {a:.4f} | {d:+.4f} |\n")
            
    print("[INFO] ablation_report.md saved.")


if __name__ == "__main__":
    main()
