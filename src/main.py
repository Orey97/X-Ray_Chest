"""
=============================================================================
                        MAIN.PY - Training Orchestrator
=============================================================================

PURPOSE:
    The main entry point for training the chest X-ray classification model.
    Coordinates all components: data loading, model creation, training loop,
    validation, checkpointing, and evaluation.

TRAINING WORKFLOW:

    1. SETUP
       ├── Parse CLI arguments
       ├── Resolve absolute paths
       ├── Set random seeds (reproducibility)
       └── Load or create schema

    2. DATA PIPELINE
       ├── Load CSV metadata
       ├── Clean and encode labels
       ├── Patient-aware split (train/val/test)
       └── Create DataLoaders

    3. MODEL SETUP
       ├── Initialize ResNet-50 with pretrained weights
       ├── Attach 14-class head
       ├── Configure loss function (BCE)
       └── Configure optimizer (Adam)

    4. TRAINING LOOP
       ├── For each epoch:
       │   ├── Train on all batches
       │   ├── Validate performance
       │   ├── Adjust learning rate (scheduler)
       │   ├── Check for improvement
       │   └── Early stopping if needed
       └── Save best model

    5. EVALUATION
       ├── Load best model
       └── Report test set performance

COMMAND LINE INTERFACE:

    python src/main.py \\
        --csv_file src/Data_Entry_2017.csv \\
        --data_dir images-224 \\
        --output_dir output \\
        --num_epochs 10 \\
        --use_weighted_loss  # Optional: for class imbalance

=============================================================================
"""

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

# Schema and Metrics imports for consistent tracking
from schema import DatasetManifest
from metrics import ClinicalMetricContainer


# =============================================================================
# CONFIGURATION
# =============================================================================
# Centralized hyperparameters for easy modification and documentation.
# These can be overridden by CLI arguments where applicable.

CONFIG = {
    "seed": 42,              # Random seed for reproducibility
    "num_epochs": 10,        # Training epochs (overridden by --num_epochs)
    "batch_size": 32,        # Images per batch (GPU memory dependent)
    "learning_rate": 1e-4,   # Initial learning rate (Adam optimizer)
    "image_size": 224,       # Image dimensions (must match pretrained model)
    "num_workers": 0,        # DataLoader workers (0 for Windows compatibility)
    "patience": 3,           # Early stopping patience (epochs without improvement)
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


def seed_everything(seed):
    """
    Set random seeds across all libraries for reproducibility.
    
    Without this, training results would vary between runs due to:
    - Random weight initialization
    - Random data shuffling
    - Random augmentation
    - CUDA non-determinism
    
    By fixing all seeds, we get reproducible experiments.
    
    Args:
        seed (int): The seed value to use everywhere
    """
    random.seed(seed)                           # Python's random module
    os.environ['PYTHONHASHSEED'] = str(seed)    # Python hash function
    np.random.seed(seed)                        # NumPy random
    torch.manual_seed(seed)                     # PyTorch CPU
    torch.cuda.manual_seed(seed)                # PyTorch GPU
    torch.backends.cudnn.deterministic = True   # CuDNN deterministic mode
    torch.backends.cudnn.benchmark = False      # Disable CuDNN auto-tuning
    print(f"Global seed set to {seed}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one complete pass through the training data.
    
    One "epoch" means seeing every training sample once.
    
    Process for each batch:
        1. Forward pass: Get model predictions
        2. Compute loss: Compare predictions to ground truth
        3. Backward pass: Compute gradients
        4. Update weights: Apply gradients via optimizer
    
    Args:
        model: The neural network
        loader: DataLoader providing batched data
        criterion: Loss function (BCE)
        optimizer: Weight update algorithm (Adam)
        device: 'cuda' or 'cpu'
        
    Returns:
        float: Average loss across all batches
    """
    model.train()  # Set model to training mode (enables dropout, batch norm updates)
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(loader):
        # Move data to GPU if available
        images, labels = images.to(device), labels.to(device)

        # ═══════════════════════════════════════════════════════════════
        # FORWARD PASS
        # ═══════════════════════════════════════════════════════════════
        optimizer.zero_grad()  # Clear gradients from previous batch
        outputs = model(images)  # Get predictions (raw logits)
        loss = criterion(outputs, labels)  # Compute BCE loss
        
        # ═══════════════════════════════════════════════════════════════
        # BACKWARD PASS
        # ═══════════════════════════════════════════════════════════════
        loss.backward()  # Compute gradients (backpropagation)
        optimizer.step()  # Update weights using gradients

        # Accumulate loss for reporting
        running_loss += loss.item() * images.size(0)
        
        # Progress indicator (every 10 batches)
        if (batch_idx + 1) % 10 == 0:
             print(f"   Batch {batch_idx+1}/{len(loader)} - Loss: {loss.item():.4f}", end='\r')

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device, label_list):
    """
    Evaluate model performance on validation/test data.
    
    Unlike training, validation:
    - Does NOT update weights
    - Does NOT use dropout (model.eval())
    - Computes metrics on entire dataset
    
    Args:
        model: The neural network
        loader: DataLoader for validation data
        criterion: Loss function
        device: 'cuda' or 'cpu'
        label_list: Class names for per-class metrics
        
    Returns:
        tuple: (average_loss, metrics_dict)
               metrics_dict contains AUROC, F1, per-class scores
    """
    model.eval()  # Set model to evaluation mode (disables dropout)
    running_loss = 0.0
    
    # Initialize unified metric container
    metrics = ClinicalMetricContainer(label_list)

    # ═══════════════════════════════════════════════════════════════════
    # NO GRADIENT COMPUTATION (faster, less memory)
    # ═══════════════════════════════════════════════════════════════════
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            # Accumulate predictions for metric calculation
            metrics.update(labels, outputs)

    avg_loss = running_loss / len(loader.dataset)
    
    # Compute all metrics at once
    results = metrics.compute()
    
    return avg_loss, results


def get_args():
    """
    Parse command line arguments.
    
    Example usage:
        python src/main.py --csv_file data.csv --data_dir images/ --num_epochs 10
    """
    parser = argparse.ArgumentParser(
        description="Train a Multi-Label ResNet model for Chest X-ray classification."
    )
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
    """
    Main training function. Orchestrates the entire training pipeline.
    """
    args = get_args()
    
    # ==========================================================================
    # STEP 1: PATH RESOLUTION
    # ==========================================================================
    # Always convert to absolute paths to avoid "file not found" issues
    # when running from different directories.
    
    abs_csv_path = os.path.abspath(args.csv_file)
    abs_image_dir = os.path.abspath(args.data_dir)
    abs_output_dir = os.path.abspath(args.output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)  # Create output dir if needed
    
    # Override CONFIG with CLI arguments
    CONFIG['num_epochs'] = args.num_epochs

    print("\n" + "="*50)
    print("       ABSOLUTE PATH RESOLUTION       ")
    print("="*50)
    print(f"CSV PATH  : {abs_csv_path}")
    print(f"IMAGES DIR: {abs_image_dir}")
    print(f"OUTPUT DIR: {abs_output_dir}")
    print("="*50 + "\n")

    # Validate paths exist
    if not os.path.exists(abs_csv_path):
        raise FileNotFoundError(f"CRITICAL ERROR: CSV file not found at: {abs_csv_path}")
    if not os.path.isdir(abs_image_dir):
        raise FileNotFoundError(f"CRITICAL ERROR: Image directory not found at: {abs_image_dir}")

    # ==========================================================================
    # STEP 2: REPRODUCIBILITY SETUP
    # ==========================================================================
    seed_everything(CONFIG["seed"])
    print(f"Using device: {CONFIG['device']}")

    # ==========================================================================
    # STEP 3: SCHEMA MANAGEMENT
    # ==========================================================================
    # The schema defines the canonical label order. This ensures consistency
    # between training and inference.
    
    schema_path = os.path.join(abs_output_dir, "schema.json")
    manifest = None
    
    if os.path.exists(schema_path):
        # Load existing schema (ensures consistency with previous training)
        print(f"Found existing Schema at: {schema_path}")
        print("Validating against current CSV...")
        manifest = DatasetManifest.load(schema_path)
        print("[INFO] Schema Loaded. Enforcing strict label order.")
    else:
        # Create new schema from CSV
        print("No existing Schema found. Creating NEW Manifest from CSV...")
        manifest = DatasetManifest.create(abs_csv_path)
        manifest.save(schema_path)  # Save immediately for future runs
        print(f"[INFO] Schema Created and Saved. ({manifest.num_classes} classes)")

    # ==========================================================================
    # STEP 4: DATA PIPELINE
    # ==========================================================================
    print("\n--- Initializing Data Pipeline ---")
    
    # Load and preprocess dataset
    dataset_manager = Dataset(abs_csv_path)
    dataset_manager.clean_column_names()
    
    # Detect which column contains labels
    label_col = "finding_labels" if "finding_labels" in dataset_manager.data.columns else "label"
    
    dataset_manager.clean_labels(label_col)
    # Use schema's label list for consistent ordering
    dataset_manager.one_hot_encode_labels(label_col, explicit_labels=manifest.label_list)
    
    # Check for missing images
    missing_files = dataset_manager.check_image_files(abs_image_dir)
    if missing_files:
        print(f"WARNING: Found {len(missing_files)} missing images. Removing them from dataset...")
        dataset_manager.data = dataset_manager.data[~dataset_manager.data['image'].isin(missing_files)]
    
    dataset_manager.select_relevant_columns()
    
    # Patient-aware split (prevents data leakage)
    train_df, val_df, test_df = dataset_manager.patient_split()
    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Verify schema compatibility
    for label in manifest.label_list:
        if label not in train_df.columns:
            raise RuntimeError(f"Schema Violation: Label '{label}' defined in schema but missing in processed dataframe.")

    # Create PyTorch DataLoaders
    train_loader, val_loader, test_loader = create_dataloader(
        train_df, val_df, test_df, 
        abs_image_dir, 
        batch_size=CONFIG['batch_size'], 
        num_workers=CONFIG['num_workers'], 
        image_size=CONFIG['image_size'],
        augment=not args.no_aug  # Disable augmentation if --no_aug flag
    )

    # ==========================================================================
    # STEP 5: MODEL SETUP
    # ==========================================================================
    print(f"\n--- Model Initialization ({manifest.num_classes} classes) ---")
    model = MultiLabelResNet(num_classes=manifest.num_classes).to(CONFIG['device'])
    
    # Validate model architecture against schema
    manifest.validate_model(model)
    
    # ==========================================================================
    # STEP 6: LOSS FUNCTION
    # ==========================================================================
    if args.use_weighted_loss:
        print("[TRAIN Config] ENABLED: Weighted BCE Loss for Imbalance Mitigation.")
        from loss import WeightedBCELossWrapper
        criterion = WeightedBCELossWrapper(
            train_df=train_df, 
            label_list=manifest.label_list, 
            device=CONFIG['device'],
            max_weight=100.0  # Safety clamp for extreme imbalance
        )
    else:
        print("[TRAIN Config] STANDARD: Unweighted BCE Loss.")
        criterion = nn.BCEWithLogitsLoss()

    # ==========================================================================
    # STEP 7: OPTIMIZER & SCHEDULER
    # ==========================================================================
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # ReduceLROnPlateau: Lower learning rate when validation stops improving
    # - mode='max': We want to MAXIMIZE AUROC
    # - factor=0.1: Reduce LR by 10x when triggered
    # - patience=1: Trigger after 1 epoch without improvement
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1)

    # ==========================================================================
    # STEP 8: TRAINING LOOP
    # ==========================================================================
    print("\n--- Starting Training ---")
    best_val_auc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch [{epoch+1}/{CONFIG['num_epochs']}]")
        
        # Train one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
        
        # Validate
        val_loss, val_results = validate(model, val_loader, criterion, CONFIG['device'], manifest.label_list)
        val_auc = val_results["mean_auroc"]
        
        print(f"\nSummary -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val meanAUC: {val_auc:.4f}")
        
        # Update learning rate scheduler
        scheduler.step(val_auc)

        # ══════════════════════════════════════════════════════════════════
        # CHECKPOINTING: Save best model
        # ══════════════════════════════════════════════════════════════════
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            save_path = os.path.join(abs_output_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"⚡ Best Model Saved! (AUC improved to {val_auc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            
            # Early stopping: Stop if no improvement for 'patience' epochs
            if epochs_no_improve >= CONFIG['patience']:
                print("Early Stopping Triggered.")
                break
        
        # Track best loss (secondary metric)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time // 60:.0f}m {total_time % 60:.0f}s. Best Val AUC: {best_val_auc:.4f}")

    # Save final model (regardless of best)
    torch.save(model.state_dict(), os.path.join(abs_output_dir, "final_model.pth"))

    print("\n" + "="*50)
    print(" [INFO] TRAINING COMPLETE")
    print("="*50)

    # ==========================================================================
    # STEP 9: FINAL EVALUATION
    # ==========================================================================
    # Load the best model for test evaluation
    best_path = os.path.join(abs_output_dir, "best_model.pth")
    final_path = os.path.join(abs_output_dir, "final_model.pth")
    
    load_path = best_path if os.path.exists(best_path) else final_path
    print(f"[EVAL] Loading model from: {load_path}")
    
    model.load_state_dict(torch.load(load_path))
    
    # Evaluate on test set
    test_loss, test_results = validate(model, test_loader, criterion, CONFIG['device'], manifest.label_list)
    
    print(f"Test Loss: {test_loss:.4f} | Test Mean AUROC: {test_results['mean_auroc']:.4f}")
    print("\nPer-Class AUROC:")
    for label, auc in test_results["per_class_auroc"].items():
        print(f" - {label:20s}: {auc:.4f}")


if __name__ == "__main__":
    main() 
