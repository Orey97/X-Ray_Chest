
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import argparse
from dataset import Dataset
from dataloader import create_dataloader
from model import MultiLabelResNet

# --- Configuration & Hyperparameters ---
CONFIG = {
    "seed": 42,
    "num_epochs": 10,
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

def calculate_metrics(y_true, y_pred, labels_list):
    """
    Computes ROC AUC score for each class and the mean.
    Handles cases where a class might not be present in the validation batch.
    """
    class_auc = {}
    valid_classes = 0
    sum_auc = 0.0
    
    # y_pred are logits, we assume they will be passed to a sigmoid implicitly by the metric or explicitly here
    # scikit-learn roc_auc_score needs probabilities, not logits.
    # We apply Sigmoid here since our model outputs raw logits.
    y_probs = torch.sigmoid(torch.from_numpy(y_pred)).numpy()

    for i, label in enumerate(labels_list):
        # Check if the class is present in y_true (needs at least one positive and one negative sample)
        if len(np.unique(y_true[:, i])) > 1:
            auc = roc_auc_score(y_true[:, i], y_probs[:, i])
            class_auc[label] = auc
            sum_auc += auc
            valid_classes += 1
        else:
            # Cannot calculate AUC if only one class is present
            class_auc[label] = -1.0 

    mean_auc = sum_auc / valid_classes if valid_classes > 0 else 0.0
    return mean_auc, class_auc

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

def validate(model, loader, criterion, device, labels_list):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            # Store predictions and labels for AUC calculation
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    
    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    mean_auc, class_auc = calculate_metrics(all_targets, all_preds, labels_list)
    
    return avg_loss, mean_auc, class_auc

def get_args():
    parser = argparse.ArgumentParser(description="Train a Multi-Label ResNet model for Chest X-ray classification.")
    parser.add_argument("--csv_file", type=str, required=True,
                        help="Path to the CSV file containing image labels (e.g., Data_Entry_2017.csv).")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the directory containing image files (e.g., images-224).")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save model checkpoints and logs.")
    return parser.parse_args()

def main():
    args = get_args()
    
    # --- EXPLICIT PATH HANDLING (No Magic) ---
    # Resolve to absolute paths immediately
    abs_csv_path = os.path.abspath(args.csv_file)
    abs_image_dir = os.path.abspath(args.data_dir)
    abs_output_dir = os.path.abspath(args.output_dir)

    print("\n" + "="*50)
    print("       ABSOLUTE PATH RESOLUTION       ")
    print("="*50)
    print(f"CSV PATH  : {abs_csv_path}")
    print(f"IMAGES DIR: {abs_image_dir}")
    print(f"OUTPUT DIR: {abs_output_dir}")
    print("="*50 + "\n")

    # Critical Existence Checks
    if not os.path.exists(abs_csv_path):
        raise FileNotFoundError(f"CRITICAL ERROR: CSV file not found at: {abs_csv_path}")
    if not os.path.isdir(abs_image_dir):
        raise FileNotFoundError(f"CRITICAL ERROR: Image directory not found at: {abs_image_dir}")

    # 1. Setup
    os.makedirs(abs_output_dir, exist_ok=True)
    seed_everything(CONFIG["seed"])
    print(f"Using device: {CONFIG['device']}")

    # 2. Data Pipeline
    print("\n--- Initializing Data Pipeline ---")
    # PASS ABSOLUTE PATHS ONLY
    dataset_manager = Dataset(abs_csv_path)
    dataset_manager.clean_column_names()
    dataset_manager.clean_labels("label")
    dataset_manager.one_hot_encode_labels("label")
    
    # Check data integrity
    missing_files = dataset_manager.check_image_files(abs_image_dir)
    if missing_files:
        print(f"WARNING: Found {len(missing_files)} missing images. Removing them from dataset...")
        # Filter out missing images to prevent DataLoader crashes
        dataset_manager.data = dataset_manager.data[~dataset_manager.data['image'].isin(missing_files)]
    
    dataset_manager.select_relevant_columns()
    
    train_df, val_df, test_df = dataset_manager.patient_split()
    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_loader, val_loader, test_loader = create_dataloader(
        train_df, val_df, test_df, 
        abs_image_dir, 
        batch_size=CONFIG['batch_size'], 
        num_workers=CONFIG['num_workers'], 
        image_size=CONFIG['image_size']
    )

    # 3. Model Setup
    labels_list = [col for col in train_df.columns if col not in ['image', 'patientid']]
    num_classes = len(labels_list)
    print(f"\n--- Model Initialization ({num_classes} classes) ---")
    
    model = MultiLabelResNet(num_classes=num_classes).to(CONFIG['device'])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1, verbose=True)

    # 4. Training Loop
    print("\n--- Starting Training ---")
    best_val_auc = 0.0
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch [{epoch+1}/{CONFIG['num_epochs']}]")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
        
        # Validate
        val_loss, val_auc, class_auc = validate(model, val_loader, criterion, CONFIG['device'], labels_list)
        
        print(f"\nSummary -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val meanAUC: {val_auc:.4f}")
        
        # Scheduler Step (Monitor AUC)
        scheduler.step(val_auc)

        # Checkpointing & Early Stopping
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
    
    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time // 60:.0f}m {total_time % 60:.0f}s. Best Val AUC: {best_val_auc:.4f}")

    # 5. Final Test Evaluation
    print("\n--- Running Final Evaluation on TEST SET ---")
    # Load best model
    model.load_state_dict(torch.load(os.path.join(abs_output_dir, "best_model.pth")))
    test_loss, test_auc, test_class_auc = validate(model, test_loader, criterion, CONFIG['device'], labels_list)
    
    print(f"Test Loss: {test_loss:.4f} | Test Mean AUROC: {test_auc:.4f}")
    print("\nPer-Class AUROC:")
    for label, auc in test_class_auc.items():
        print(f" - {label:20s}: {auc:.4f}")

if __name__ == "__main__":
    main()
 
