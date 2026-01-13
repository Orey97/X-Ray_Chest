
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Dataset
from dataloader import create_dataloader
from model import MultiLabelResNet
import time

def train_model(num_epochs=5, batch_size=32, learning_rate=1e-4, image_size=224):
    # --- 1. Setup Paths and Device ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_ENTRY_PATH = os.path.join(PROJECT_ROOT, "Data_Entry_2017.csv")
    IMAGE_DIR = os.path.join(PROJECT_ROOT, "images-224")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Data Preparation ---
    print("Initializing Dataset...")
    dataset_manager = Dataset(DATA_ENTRY_PATH)
    
    # Preprocessing steps
    dataset_manager.clean_column_names()
    dataset_manager.clean_labels("label")
    dataset_manager.one_hot_encode_labels("label")
    
    # Ensure we only keep valid images
    missing_files = dataset_manager.check_image_files(IMAGE_DIR)
    if len(missing_files) > 0:
        print(f"Warning: {len(missing_files)} images not found in {IMAGE_DIR}. They will be handled or might cause errors.")
        # In a real scenario, you'd filter these out from the dataframe
    
    dataset_manager.select_relevant_columns()
    
    # Train/Val/Test Split (Patient-aware)
    train_df, val_df, test_df = dataset_manager.patient_split()
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")

    # Create DataLoaders
    # Note: num_workers=0 for Windows compatibility to avoid multiprocessing issues in basic script
    train_loader, val_loader, test_loader = create_dataloader(
        train_df, val_df, test_df, 
        IMAGE_DIR, 
        batch_size=batch_size, 
        num_workers=0, 
        image_size=image_size
    )

    # --- 3. Model Initialization ---
    # Determine number of classes from the processed dataframe columns (excluding image and patientid)
    # The columns are [image, patientid, class1, class2, ...]
    labels_list = [col for col in train_df.columns if col not in ['image', 'patientid']]
    num_classes = len(labels_list)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {labels_list}")

    model = MultiLabelResNet(num_classes=num_classes).to(device)

    # --- 4. Loss and Optimizer ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 5. Training Loop ---
    print("Starting training...")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        
        # --- Validation Step ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Complete. Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    total_time = time.time() - start_time
    print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")

    # --- 6. Save Model ---
    model_save_path = os.path.join(PROJECT_ROOT, "chest_xray_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_model(num_epochs=1) # Run simple test with 1 epoch 
