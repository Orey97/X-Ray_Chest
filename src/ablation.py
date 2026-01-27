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
    print(f"\n" + "="*50)
    print(f"   RUNNING ABLATION VARIANT: {name}")
    print(f"   Augmentation: {'ENABLED' if use_aug else 'DISABLED'}")
    print("="*50)
    
    cmd = [
        sys.executable, "src/main.py",
        "--csv_file", args.csv_file,
        "--data_dir", args.data_dir,
        "--output_dir", output_dir,
        "--num_epochs", str(args.epochs)
    ]
    
    if not use_aug:
        cmd.append("--no_aug")
        
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False) # stream output to console
    
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for variant {name}. Exit Code: {result.returncode}")
        
    return os.path.join(output_dir, "best_model.pth")

def evaluate_model(model_path, test_loader, device, manifest):
    print(f"[EVAL] Loading {model_path}...")
    
    if not os.path.exists(model_path):
        # Fallback to final if best not found (e.g. no improvement)
        fallback = model_path.replace("best_model.pth", "final_model.pth")
        if os.path.exists(fallback):
            print(f"[WARN] fit using final_model.pth fallback")
            model_path = fallback
        else:
            return None

    model = MultiLabelResNet(num_classes=manifest.num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    metrics = ClinicalMetricContainer(manifest.label_list)
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            metrics.update(labels, logits)
            
    return metrics.compute()

def main():
    parser = argparse.ArgumentParser(description="Run Ablation Study (Augmentation vs Baseline)")
    parser.add_argument("--csv_file", type=str, default="test_env/data.csv")
    parser.add_argument("--data_dir", type=str, default="test_env/images")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    
    base_dir = "ablation_results"
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. Run Baseline (No Aug)
    path_baseline = train_variant("BASELINE", os.path.join(base_dir, "baseline"), False, args)
    
    # 2. Run Augmented (With Aug)
    path_aug = train_variant("AUGMENTED", os.path.join(base_dir, "augmented"), True, args)
    
    # 3. Comparative Evaluation
    print("\n" + "="*50)
    print("   COMPARING RESULTS")
    print("="*50)
    
    # Setup Data (Once)
    # We need to rely on the schema from one of the runs (they should be identical)
    schema_path = os.path.join(base_dir, "baseline", "schema.json")
    manifest = DatasetManifest.load(schema_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Test Data
    ds = ProjectDataset(args.csv_file) # Re-init logic
    ds.clean_column_names()
    label_col = "finding_labels" if "finding_labels" in ds.data.columns else "label"
    ds.clean_labels(label_col)
    ds.one_hot_encode_labels(label_col, explicit_labels=manifest.label_list)
    ds.select_relevant_columns()
    _, _, test_df = ds.patient_split()
    
    _, _, test_tf = get_transforms()
    test_ds = ChestXRayDataset(test_df, args.data_dir, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    res_base = evaluate_model(path_baseline, test_loader, device, manifest)
    res_aug = evaluate_model(path_aug, test_loader, device, manifest)
    
    if not res_base or not res_aug:
        print("[ERROR] Could not load models for comparison.")
        return

    print(f"\n{'Metric':<20} | {'Baseline':<10} | {'Augmented':<10} | {'Delta':<10}")
    print("-" * 60)
    
    base_auc = res_base["mean_auroc"]
    aug_auc = res_aug["mean_auroc"]
    delta = aug_auc - base_auc
    
    print(f"{'Mean AUROC':<20} | {base_auc:.4f}     | {aug_auc:.4f}     | {delta:+.4f}")
    print("-" * 60)
    
    # Save Report
    with open("ablation_report.md", "w") as f:
        f.write("# Ablation Study Report\n")
        f.write(f"Mean AUROC Delta: **{delta:+.4f}**\n\n")
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
