"""
=============================================================================
                    DATALOADER.PY - PyTorch Data Pipeline
=============================================================================

PURPOSE:
    Bridges the gap between pandas DataFrames and PyTorch's training loop.
    Implements data augmentation for improved generalization.

ARCHITECTURE:
    
    DataFrame → ChestXRayDataset → DataLoader → Training Loop
                     ↓
              [Load Image]
                     ↓
              [Apply Transforms]
                     ↓
              [Return Tensor]

DATA AUGMENTATION STRATEGY:

    Neural networks can memorize exact pixel values. Augmentation forces
    them to learn CONCEPTS instead of SPECIFICS.
    
    For Chest X-rays, we use DOMAIN-AWARE augmentations:
    
    ✅ INCLUDED (medically valid):
    - Horizontal Flip: Anatomically possible (mirror image)
    - Small Rotation (±7°): Simulates patient positioning variance
    - Small Zoom (90-110%): Simulates different imaging distances
    - Brightness/Contrast: Simulates different X-ray equipment
    
    ❌ EXCLUDED (would create invalid images):
    - Vertical Flip: Humans don't stand upside down
    - Extreme Rotation: Would destroy diagnostic context
    - Color Shifts: X-rays are grayscale (no color to shift)
    - Elastic Deformation: Would distort anatomy unrealistically

IMAGENET NORMALIZATION:

    We normalize using ImageNet's mean/std because our model uses
    ImageNet pretrained weights. The model expects inputs in the
    same distribution it was trained on.
    
    mean = [0.485, 0.456, 0.406]  # RGB channel means
    std  = [0.229, 0.224, 0.225]  # RGB channel stds

=============================================================================
"""

import os
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image


class ChestXRayDataset(TorchDataset):
    """
    Custom PyTorch Dataset for chest X-ray images.
    
    PyTorch's DataLoader needs a Dataset that implements:
    - __len__(): How many samples?
    - __getitem__(idx): Return the idx-th sample
    
    The DataLoader then:
    1. Calls __getitem__ multiple times to collect a batch
    2. Stacks samples into tensors
    3. Optionally shuffles for training
    4. Handles multi-worker parallel loading
    """
    
    def __init__(self, dataframe, image_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            dataframe (DataFrame): Pandas DataFrame with columns:
                                  - 'image': filename of each image
                                  - [label columns]: one-hot encoded targets
                                  
            image_dir (str): Path to directory containing image files.
                            Will be joined with filename from DataFrame.
                            
            transform: Torchvision transform pipeline to apply to images.
                      Training: augmentation + normalization
                      Validation/Test: normalization only
        """
        # Reset index to ensure 0, 1, 2, ... indexing (in case of splits)
        self.dataframe = dataframe.reset_index(drop=True)
        
        # Always use absolute path for reliability
        self.image_dir = os.path.abspath(image_dir)
        
        if not os.path.exists(self.image_dir):
             raise FileNotFoundError(f"CRITICAL ERROR (DataLoader): Image directory not found at: {self.image_dir}")

        self.transform = transform

        # Extract image filenames
        self.image_path = self.dataframe["image"].values
        
        # Identify label columns (everything except image and patientid)
        self.labels_col = [col for col in self.dataframe.columns if col not in ["image", "patientid"]]
        
        # ═══════════════════════════════════════════════════════════════════
        # PRE-EXTRACT LABELS AS NUMPY ARRAY
        # ═══════════════════════════════════════════════════════════════════
        # Why float32?
        #   PyTorch expects float tensors for gradient computation.
        #   The labels are 0/1 integers, but we need them as floats.
        #
        self.labels = self.dataframe[self.labels_col].values.astype('float32')

    def __len__(self):
        """Return total number of samples in dataset."""
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample (image + labels) by index.
        
        This method is called by DataLoader thousands of times per epoch.
        Keep it fast! Expensive operations should be in __init__ if possible.
        
        Args:
            idx (int): Index of sample to retrieve (0 to len-1)
            
        Returns:
            tuple: (image_tensor, label_tensor)
                  - image_tensor: [C, H, W] = [3, 224, 224]
                  - label_tensor: [num_classes] = [14] with 0/1 values
        """
        # Get image filename and construct full path
        img_name = self.image_path[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # ═══════════════════════════════════════════════════════════════════
        # LOAD IMAGE AND CONVERT TO RGB
        # ═══════════════════════════════════════════════════════════════════
        # Why RGB?
        #   X-rays are naturally GRAYSCALE (single channel)
        #   But pretrained CNNs expect 3 channels (RGB)
        #   convert("RGB") duplicates grayscale across 3 channels
        #   This is a standard practice that works well in practice
        #
        image = Image.open(img_path).convert("RGB")

        # ═══════════════════════════════════════════════════════════════════
        # APPLY TRANSFORMS
        # ═══════════════════════════════════════════════════════════════════
        # The transform pipeline does:
        #   1. Resize to 224x224 (if not already)
        #   2. Random augmentations (training only)
        #   3. Convert PIL Image → PyTorch Tensor
        #   4. Normalize with ImageNet statistics
        #
        if self.transform:
            image = self.transform(image)
            
        # Convert labels from numpy to PyTorch tensor
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return image, labels
    

def get_transforms(image_size=224):
    """
    Create transform pipelines for train/val/test.
    
    Returns three pipelines:
    1. train_transform: Augmentation + Normalization
    2. val_transform: Normalization only (no augmentation)
    3. test_transform: Normalization only (same as val)
    
    Args:
        image_size (int): Target image size (default: 224 for ResNet)
        
    Returns:
        tuple: (train_transform, val_transform, test_transform)
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # TRAINING AUGMENTATION PIPELINE
    # ═══════════════════════════════════════════════════════════════════════
    # Each transform is applied with some probability or randomness.
    # This creates slightly different versions of each image every epoch.
    #
    train_transform = transforms.Compose([
        # RandomResizedCrop: Zoom in/out by 90-110%
        # - Simulates different imaging distances
        # - Forces model to handle scale variations
        transforms.RandomResizedCrop(image_size, scale=(0.9, 1.1)),
        
        # RandomHorizontalFlip: 50% chance to mirror
        # - Anatomically valid (left/right symmetry)
        # - Doubles effective dataset size
        transforms.RandomHorizontalFlip(p=0.5), 
        
        # RandomRotation: Rotate ±7 degrees
        # - Simulates patient positioning variance
        # - Small enough to preserve diagnostic features
        transforms.RandomRotation(degrees=7),
        
        # ColorJitter: Brightness/contrast variation
        # - Simulates different X-ray equipment/settings
        # - 10% variation in brightness and contrast
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        
        # ToTensor: PIL Image [H,W,C] → Tensor [C,H,W]
        # - Also scales pixel values from 0-255 to 0.0-1.0
        transforms.ToTensor(), 
        
        # Normalize: Apply ImageNet statistics
        # - Required because we use ImageNet pretrained weights
        # - Formula: normalized = (pixel - mean) / std
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # VALIDATION/TEST PIPELINE (NO AUGMENTATION)
    # ═══════════════════════════════════════════════════════════════════════
    # During evaluation, we want DETERMINISTIC results.
    # Same image → Same prediction (no randomness)
    #
    base_transform = transforms.Compose([
        # Simple resize (no random cropping)
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, base_transform, base_transform


def create_dataloader(train_df, val_df, test_df, image_dir, batch_size=32, num_workers=0, image_size=224, augment=True):
    """
    Create PyTorch DataLoaders for train/val/test splits.
    
    DataLoader handles:
    - Batching: Groups samples into batches
    - Shuffling: Randomizes order each epoch (training only)
    - Parallel loading: Uses multiple CPU workers
    - Memory pinning: Optimizes CPU→GPU transfer
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        image_dir: Path to images
        batch_size: Samples per batch (default: 32)
        num_workers: CPU workers for loading (0 for Windows compatibility)
        image_size: Image dimensions (default: 224)
        augment: Whether to apply augmentation to training (default: True)
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_tf, val_tf, test_tf = get_transforms(image_size)

    # ═══════════════════════════════════════════════════════════════════════
    # ABLATION MODE: Disable augmentation for controlled experiments
    # ═══════════════════════════════════════════════════════════════════════
    # If augment=False, use validation transform (no randomness) for training.
    # This allows fair A/B comparison: "How much does augmentation help?"
    #
    final_train_tf = train_tf if augment else val_tf
    
    if not augment:
        print("[DATA] Augmentations DISABLED (Ablation Mode).")

    # Create Dataset objects
    train_set = ChestXRayDataset(train_df, image_dir, transform=final_train_tf)
    val_set = ChestXRayDataset(val_df, image_dir, transform=val_tf)
    test_set = ChestXRayDataset(test_df, image_dir, transform=test_tf)

    # Create DataLoader objects
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True,       # Randomize order each epoch for training
        num_workers=num_workers, 
        pin_memory=True     # Speeds up CPU→GPU transfer
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False,      # Keep order consistent for validation
        num_workers=num_workers, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False,      # Keep order consistent for testing
        num_workers=num_workers, 
        pin_memory=True
    )

    return train_loader, val_loader, test_loader 
