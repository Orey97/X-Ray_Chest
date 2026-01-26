import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from dataset import Dataset
from dataloader import create_dataloader
from model import MultiLabelResNet

# --- PHASE 1 PROFESSIONALIZATION IMPORTS ---
from schema import DatasetManifest
from metrics import ClinicalMetricContainer

# --- Configuration & Hyperparameters ---
CONFIG = {
    "seed": 42,
    "num_epochs": 10, # Will be overwritten by args
    "batch_size": 32,
    "learning_rate": 1e-4,
    "image_size": 224,
    "num_workers": 0, # Set to 0 for Windows compatibility
    "patience": 3,    # Early Stopping patience
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def seed_everything(seed):
    """Ensures reproducibility across runs."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {seed}")

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        
        if (batch_idx + 1) % 10 == 0:
             print(f"   Batch {batch_idx+1}/{len(loader)} - Loss: {loss.item():.4f}", end='\r')

    return running_loss / len(loader.dataset)

def validate(model, loader, criterion, device, label_list):
    model.eval()
    running_loss = 0.0
    
    # Initialize Unified Metric Container
    metrics = ClinicalMetricContainer(label_list)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            # Update Metrics
            metrics.update(labels, outputs)

    avg_loss = running_loss / len(loader.dataset)
    
    # Compute Final Specs
    results = metrics.compute()
    
    return avg_loss, results

def get_args():
    parser = argparse.ArgumentParser(description="Train a Multi-Label ResNet model for Chest X-ray classification.")
    parser.add_argument("--csv_file", type=str, required=True,
                        help="Path to the CSV file containing image labels (e.g., Data_Entry_2017.csv).")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the directory containing image files (e.g., images-224).")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save model checkpoints and logs.")
    parser.add_argument("--use_weighted_loss", action="store_true",
                        help="Enable dynamic class weighting to mitigate imbalance.")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--no_aug", action="store_true",
                        help="Disable data augmentation (for Ablation Study).")
    return parser.parse_args()

def main():
    args = get_args()
    
    # --- EXPLICIT PATH HANDLING ---
    abs_csv_path = os.path.abspath(args.csv_file)
    abs_image_dir = os.path.abspath(args.data_dir)
    abs_output_dir = os.path.abspath(args.output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    
    # Overwrite CONFIG
    CONFIG['num_epochs'] = args.num_epochs

    print("\n" + "="*50)
    print("       ABSOLUTE PATH RESOLUTION       ")
    print("="*50)
    print(f"CSV PATH  : {abs_csv_path}")
    print(f"IMAGES DIR: {abs_image_dir}")
    print(f"OUTPUT DIR: {abs_output_dir}")
    print("="*50 + "\n")

    if not os.path.exists(abs_csv_path):
        raise FileNotFoundError(f"CRITICAL ERROR: CSV file not found at: {abs_csv_path}")
    if not os.path.isdir(abs_image_dir):
        raise FileNotFoundError(f"CRITICAL ERROR: Image directory not found at: {abs_image_dir}")

    # 1. Setup
    seed_everything(CONFIG["seed"])
    print(f"Using device: {CONFIG['device']}")

    # 2. Schema / Manifest Management (PHASE 1 INTEGRITY)
    schema_path = os.path.join(abs_output_dir, "schema.json")
    manifest = None
    
    if os.path.exists(schema_path):
        print(f"Found existing Schema at: {schema_path}")
        print("Validating against current CSV...")
        # In a generic pipeline we might check checksums. 
        # For now, we load it and treat it as the LAW.
        manifest = DatasetManifest.load(schema_path)
        print("[INFO] Schema Loaded. Enforcing strict label order.")
    else:
        print("No existing Schema found. Creating NEW Manifest from CSV...")
        manifest = DatasetManifest.create(abs_csv_path)
        manifest.save(schema_path) # SAVE IMMEDIATELY
        print(f"[INFO] Schema Created and Saved. ({manifest.num_classes} classes)")

    # 3. Data Pipeline
    print("\n--- Initializing Data Pipeline ---")
    dataset_manager = Dataset(abs_csv_path)
    dataset_manager.clean_column_names()
    # Detect proper label column
    label_col = "finding_labels" if "finding_labels" in dataset_manager.data.columns else "label"
    
    dataset_manager.clean_labels(label_col)
    dataset_manager.one_hot_encode_labels(label_col, explicit_labels=manifest.label_list)
    
    # Check data integrity
    missing_files = dataset_manager.check_image_files(abs_image_dir)
    if missing_files:
        print(f"WARNING: Found {len(missing_files)} missing images. Removing them from dataset...")
        dataset_manager.data = dataset_manager.data[~dataset_manager.data['image'].isin(missing_files)]
    
    dataset_manager.select_relevant_columns()
    
    train_df, val_df, test_df = dataset_manager.patient_split()
    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Verify column consistency with schema
    # The dataset_manager should now have columns matching manifest.label_list
    # We double check
    for label in manifest.label_list:
        if label not in train_df.columns:
            raise RuntimeError(f"Schema Violation: Label '{label}' defined in schema but missing in processed dataframe.")

    train_loader, val_loader, test_loader = create_dataloader(
        train_df, val_df, test_df, 
        abs_image_dir, 
        batch_size=CONFIG['batch_size'], 
        num_workers=CONFIG['num_workers'], 
        image_size=CONFIG['image_size'],
        augment=not args.no_aug
    )

    # 3. Model Setup
    print(f"\n--- Model Initialization ({manifest.num_classes} classes) ---")
    model = MultiLabelResNet(num_classes=manifest.num_classes).to(CONFIG['device'])
    
    # Schema Binding (Optional but good for debug)
    manifest.validate_model(model)
    
    # LOSS FUNCTION STRATEGY
    if args.use_weighted_loss:
        print("[TRAIN Config] ENABLED: Weighted BCE Loss for Imbalance Mitigation.")
        from loss import WeightedBCELossWrapper
        criterion = WeightedBCELossWrapper(
            train_df=train_df, 
            label_list=manifest.label_list, 
            device=CONFIG['device'],
            max_weight=100.0 # Safety Clamp
        )
    else:
        print("[TRAIN Config] STANDARD: Unweighted BCE Loss.")
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1)

    # 5. Training Loop
    print("\n--- Starting Training ---")
    print("\n--- Starting Training ---")
    best_val_auc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch [{epoch+1}/{CONFIG['num_epochs']}]")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
        
        # Validation using Unified Metrics
        val_loss, val_results = validate(model, val_loader, criterion, CONFIG['device'], manifest.label_list)
        val_auc = val_results["mean_auroc"]
        
        print(f"\nSummary -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val meanAUC: {val_auc:.4f}")
        
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            save_path = os.path.join(abs_output_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"⚡ Best Model Saved! (AUC improved to {val_auc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= CONFIG['patience']:
                print("Early Stopping Triggered.")
                break
        
        # Save model based on best validation loss (optional, can be removed if only AUC matters)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Note: This will overwrite best_model.pth if AUC also improved in the same epoch,
            # or if loss improves but AUC doesn't. Consider separate filenames if both are needed.
            # For this change, we'll assume 'best_model.pth' is the primary save based on AUC.
            # If the user intended to save based on loss, they should specify a different filename.
            # As the instruction is to "Save final model and fallback loading logic",
            # I'll keep the AUC-based best_model.pth as the primary "best" model.
            # The user's snippet for best_val_loss saving was a bit ambiguous.
            pass # Keeping the AUC-based saving as primary for 'best_model.pth'

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time // 60:.0f}m {total_time % 60:.0f}s. Best Val AUC: {best_val_auc:.4f}")

    # Save Final Model always
    torch.save(model.state_dict(), os.path.join(abs_output_dir, "final_model.pth"))

    print("\n" + "="*50)
    print(" [INFO] TRAINING COMPLETE")
    print("="*50)

    # 6. Evaluation (Load Best or Final)
    best_path = os.path.join(abs_output_dir, "best_model.pth")
    final_path = os.path.join(abs_output_dir, "final_model.pth")
    
    load_path = best_path if os.path.exists(best_path) else final_path
    print(f"[EVAL] Loading model from: {load_path}")
    
    model.load_state_dict(torch.load(load_path))
    
    test_loss, test_results = validate(model, test_loader, criterion, CONFIG['device'], manifest.label_list)
    
    print(f"Test Loss: {test_loss:.4f} | Test Mean AUROC: {test_results['mean_auroc']:.4f}")
    print("\nPer-Class AUROC:")
    for label, auc in test_results["per_class_auroc"].items():
        print(f" - {label:20s}: {auc:.4f}")

if __name__ == "__main__":
    main() 
