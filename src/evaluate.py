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
    args = get_args()
    
    # 1. Path Resolution
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Allow command line or default relative paths
    csv_path = args.csv_file if os.path.isabs(args.csv_file) else os.path.join(base_dir, "src", args.csv_file)
    if not os.path.exists(csv_path): # Fallback
         csv_path = os.path.join(base_dir, args.csv_file)

    img_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(base_dir, args.data_dir)
    model_dir = args.model_dir if os.path.isabs(args.model_dir) else os.path.join(base_dir, args.model_dir)
    
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
    
    # 2. Schema Enforcement (Phase 1)
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"❌ Critical: schema.json not found in {model_dir}. Evaluation aborted.")
    
    manifest = DatasetManifest.load(schema_path)
    print(f"[SCHEMA] Loaded {manifest.num_classes} classes from {schema_path}")

    # 3. Data Loading (Patient-Aware)
    print("[DATA] Preparing Test Set...")
    project_ds = ProjectDataset(csv_path)
    project_ds.clean_column_names()
    project_ds.clean_labels("finding_labels") # Or "label", logic handles it
    
    # CRITICAL: Use Manifest Labels
    project_ds.one_hot_encode_labels("finding_labels", explicit_labels=manifest.label_list)
    
    # Check Image integrity
    project_ds.check_image_files(img_dir) # Warns but proceeds
    
    # Select only numeric targets (removes list-based label columns)
    project_ds.select_relevant_columns()
    
    # Split
    _, _, test_df = project_ds.patient_split()
    print(f"[DATA] Test Set Size: {len(test_df)} patients/images")
    
    # Torch Data
    _, val_tf, _ = get_transforms()
    test_ds = ChestXRayDataset(test_df, img_dir, val_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # 4. Model Loading
    print(f"[MODEL] Loading {model_path}...")
    model = MultiLabelResNet(num_classes=manifest.num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Validate
    manifest.validate_model(model)
    
    # 5. Core Evaluation
    print("[INFO] Starting Inference...")
    metrics = ClinicalMetricContainer(manifest.label_list)
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader):
            imgs = imgs.to(device)
            logits = model(imgs)
            metrics.update(labels, logits)
            
    # 6. Report Generation
    results = metrics.compute()
    
    print("\n" + "="*50)
    print(" [INFO] FINAL CLINICAL METRICS (Test Set)")
    print("="*50)
    print(f"{'Pathology':<25} | {'AUROC':<10}")
    print("-" * 40)
    
    # If Bootstrap requested (Phase 4)
    ci_low, ci_high = (None, None)
    if args.bootstrap:
        ci_low, ci_high = metrics.compute_bootstrap_ci(n_rounds=500) # 500 for speed demo
        
    for label, score in results["per_class_auroc"].items():
        print(f"{label:<25} | {score:.4f}")
        
    print("-" * 40)
    mean_auc = results["mean_auroc"]
    
    if args.bootstrap:
        print(f"MEAN AUROC:             | {mean_auc:.4f} (95% CI: {ci_low:.4f} - {ci_high:.4f})")
    else:
        print(f"MEAN AUROC:             | {mean_auc:.4f}")
    print("="*50 + "\n")
    
    # 7. Error Slicing (Phase 4)
    if args.slice_errors:
        print("[ANALYSIS] Running Error Slicing...")
        slicer = ErrorSlicer(model, test_loader, manifest, device, image_dir=img_dir)
        errors = slicer.find_top_errors(k=5)
        
        report_path = os.path.join(base_dir, "error_analysis_report.md")
        slicer.save_report(errors, report_path)
        
        # Visual Gallery (Task 9)
        gallery_path = os.path.join(base_dir, "error_gallery.png")
        slicer.plot_top_errors(errors, gallery_path)

    # 8. Calibration Curves (Phase 5)
    print("[ANALYSIS] Computing Calibration Curves...")
    curves = metrics.compute_calibration_curve(n_bins=10)
    
    # Save curves to simple CSV for plotting
    calib_path = os.path.join(base_dir, "calibration_data.csv")
    with open(calib_path, "w") as f:
        f.write("label,prob_pred,prob_true\n")
        for label, (y_true, y_pred) in curves.items():
            for yt, yp in zip(y_true, y_pred):
                f.write(f"{label},{yp:.4f},{yt:.4f}\n")
    print(f"[ANALYSIS] Calibration data saved to {calib_path}")

if __name__ == "__main__":
    evaluate_pipeline()
