"""
=============================================================================
                    EVALUATE.PY - Comprehensive Model Evaluation
=============================================================================

PURPOSE:
    Performs rigorous evaluation of a trained model on the test set.
    Implements clinical-grade metrics with statistical rigor.

FEATURES:

    1. CORE METRICS
       - Per-class AUROC (clinical standard)
       - Mean AUROC (aggregate performance)
       - F1 Score (precision/recall balance)
    
    2. BOOTSTRAP CONFIDENCE INTERVALS (--bootstrap)
       - Generates 95% CI for Mean AUROC
       - Required for medical publications
       - Example output: "AUROC: 0.74 (95% CI: 0.72-0.76)"
    
    3. ERROR SLICING (--slice_errors)
       - Finds Top-K False Positives (model hallucinations)
       - Finds Top-K False Negatives (missed detections)
       - Generates visual error gallery
    
    4. CALIBRATION ANALYSIS
       - Checks if probability 70% means actual 70% chance
       - Outputs calibration curves for model trustworthiness

USAGE:
    # Basic evaluation
    python src/evaluate.py
    
    # With confidence intervals (slow but rigorous)
    python src/evaluate.py --bootstrap
    
    # With error analysis
    python src/evaluate.py --slice_errors

=============================================================================
"""

import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

# Custom Modules
from dataset import Dataset as ProjectDataset
from dataloader import ChestXRayDataset, get_transforms
from model import MultiLabelResNet
from schema import DatasetManifest
from metrics import ClinicalMetricContainer
from analysis import ErrorSlicer


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Chest X-Ray Model (Phase 4 Standards).")
    parser.add_argument("--csv_file", type=str, default="Data_Entry_2017.csv",
                        help="Path to CSV metadata.")
    parser.add_argument("--data_dir", type=str, default="images-224",
                        help="Path to images directory.")
    parser.add_argument("--model_dir", type=str, default="output",
                        help="Directory containing best_model.pth and schema.json.")
    parser.add_argument("--bootstrap", action="store_true", 
                        help="Compute 95%% Confidence Intervals (Slow).")
    parser.add_argument("--slice_errors", action="store_true",
                        help="Generate Error Analysis Report.")
    return parser.parse_args()


def evaluate_pipeline():
    """
    Main evaluation pipeline.
    
    Workflow:
    1. Resolve paths and load schema
    2. Prepare test set (using same splits as training)
    3. Load trained model
    4. Run inference on entire test set
    5. Compute metrics
    6. (Optional) Bootstrap confidence intervals
    7. (Optional) Error slicing and visualization
    8. Calibration analysis
    """
    args = get_args()
    
    # ==========================================================================
    # STEP 1: PATH RESOLUTION
    # ==========================================================================
    # Handle both absolute and relative paths gracefully.
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Allow command line or default relative paths
    csv_path = args.csv_file if os.path.isabs(args.csv_file) else os.path.join(base_dir, "src", args.csv_file)
    if not os.path.exists(csv_path):  # Fallback to base dir
         csv_path = os.path.join(base_dir, args.csv_file)

    img_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(base_dir, args.data_dir)
    model_dir = args.model_dir if os.path.isabs(args.model_dir) else os.path.join(base_dir, args.model_dir)
    
    # ==========================================================================
    # STEP 2: LOCATE MODEL FILE
    # ==========================================================================
    # First try best_model.pth (saved when validation improved)
    # Fall back to final_model.pth (saved at end of training)
    
    model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"[MODEL] 'best_model.pth' not found. Checking for 'final_model.pth'...")
        final_path = os.path.join(model_dir, "final_model.pth")
        if os.path.exists(final_path):
            model_path = final_path
        else:
            raise FileNotFoundError(f"Neither best_model.pth nor final_model.pth found in {model_dir}")
            
    schema_path = os.path.join(model_dir, "schema.json")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Device: {device}")
    
    # ==========================================================================
    # STEP 3: SCHEMA ENFORCEMENT
    # ==========================================================================
    # The schema is REQUIRED for evaluation. It defines label order.
    
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"❌ Critical: schema.json not found in {model_dir}. Evaluation aborted.")
    
    manifest = DatasetManifest.load(schema_path)
    print(f"[SCHEMA] Loaded {manifest.num_classes} classes from {schema_path}")

    # ==========================================================================
    # STEP 4: DATA LOADING (Patient-Aware Split)
    # ==========================================================================
    # CRITICAL: Use the SAME splitting logic as training to get the same test set.
    # If splits differ, we'd be evaluating on training data → invalid results.
    
    print("[DATA] Preparing Test Set...")
    project_ds = ProjectDataset(csv_path)
    project_ds.clean_column_names()
    project_ds.clean_labels("finding_labels")  # Parse multi-label strings
    
    # Use manifest labels for consistent ordering
    project_ds.one_hot_encode_labels("finding_labels", explicit_labels=manifest.label_list)
    
    # Check image integrity (warns but proceeds)
    project_ds.check_image_files(img_dir)
    
    # Select only relevant columns
    project_ds.select_relevant_columns()
    
    # Get test split (same random seed = same split as training)
    _, _, test_df = project_ds.patient_split()
    print(f"[DATA] Test Set Size: {len(test_df)} patients/images")
    
    # Create PyTorch DataLoader
    _, val_tf, _ = get_transforms()
    test_ds = ChestXRayDataset(test_df, img_dir, val_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # ==========================================================================
    # STEP 5: MODEL LOADING
    # ==========================================================================
    print(f"[MODEL] Loading {model_path}...")
    model = MultiLabelResNet(num_classes=manifest.num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Validate model against schema
    manifest.validate_model(model)
    
    # ==========================================================================
    # STEP 6: CORE EVALUATION
    # ==========================================================================
    print("[INFO] Starting Inference...")
    metrics = ClinicalMetricContainer(manifest.label_list)
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            logits = model(imgs)
            metrics.update(labels, logits)  # Accumulate predictions
            
    # ==========================================================================
    # STEP 7: REPORT GENERATION
    # ==========================================================================
    results = metrics.compute()
    
    print("\n" + "="*50)
    print(" [INFO] FINAL CLINICAL METRICS (Test Set)")
    print("="*50)
    print(f"{'Pathology':<25} | {'AUROC':<10}")
    print("-" * 40)
    
    # If Bootstrap requested, compute confidence intervals
    ci_low, ci_high = (None, None)
    if args.bootstrap:
        ci_low, ci_high = metrics.compute_bootstrap_ci(n_rounds=500)  # 500 for speed
        
    for label, score in results["per_class_auroc"].items():
        print(f"{label:<25} | {score:.4f}")
        
    print("-" * 40)
    mean_auc = results["mean_auroc"]
    
    if args.bootstrap:
        print(f"MEAN AUROC:             | {mean_auc:.4f} (95% CI: {ci_low:.4f} - {ci_high:.4f})")
    else:
        print(f"MEAN AUROC:             | {mean_auc:.4f}")
    print("="*50 + "\n")
    
    # ==========================================================================
    # STEP 8: ERROR SLICING (Optional)
    # ==========================================================================
    # Identifies the model's worst mistakes for qualitative review.
    
    if args.slice_errors:
        print("[ANALYSIS] Running Error Slicing...")
        slicer = ErrorSlicer(model, test_loader, manifest, device, image_dir=img_dir)
        errors = slicer.find_top_errors(k=5)
        
        # Save markdown report
        report_path = os.path.join(base_dir, "error_analysis_report.md")
        slicer.save_report(errors, report_path)
        
        # Generate visual gallery
        gallery_path = os.path.join(base_dir, "error_gallery.png")
        slicer.plot_top_errors(errors, gallery_path)

    # ==========================================================================
    # STEP 9: CALIBRATION ANALYSIS
    # ==========================================================================
    # Checks if predicted probabilities are well-calibrated.
    # "When model says 70%, is it actually right 70% of the time?"
    
    print("[ANALYSIS] Computing Calibration Curves...")
    curves = metrics.compute_calibration_curve(n_bins=10)
    
    # Save calibration data to CSV for external plotting
    calib_path = os.path.join(base_dir, "calibration_data.csv")
    with open(calib_path, "w") as f:
        f.write("label,prob_pred,prob_true\n")
        for label, (y_true, y_pred) in curves.items():
            for yt, yp in zip(y_true, y_pred):
                f.write(f"{label},{yp:.4f},{yt:.4f}\n")
    print(f"[ANALYSIS] Calibration data saved to {calib_path}")


if __name__ == "__main__":
    evaluate_pipeline()
