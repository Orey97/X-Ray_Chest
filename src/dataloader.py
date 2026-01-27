import os
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image

class ChestXRayDataset(TorchDataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True) # Reset indices to 0...n-1 after splitting
        
        # Explicit Absolute Path Resolution
        self.image_dir = os.path.abspath(image_dir)
        # We don't print here to avoid flooding logs (Dataset init happens often), 
        # but the passed path MUST be valid/absolute by now.
        if not os.path.exists(self.image_dir):
             raise FileNotFoundError(f"CRITICAL ERROR (DataLoader): Image directory not found at: {self.image_dir}")

        self.transform = transform # Transformation pipeline (Resize, ToTensor, Normalize, etc.)

        self.image_path = self.dataframe["image"].values
        # 'image' is not a label, and 'patientid' was used for splitting but is not needed for training input
        self.labels_col = [col for col in self.dataframe.columns if col not in ["image", "patientid"]]
        
        # Create matrix after one-hot-encoding, each row will have 0/1 values for each label
        # Convert to float32 because PyTorch works primarily with this type for gradients
        self.labels = self.dataframe[self.labels_col].values.astype('float32')
        # Necessary conversion to float32 for PyTorch

    def __len__(self):
        return len(self.dataframe) # Required by PyTorch to know when to stop iteration
    
    def __getitem__(self, idx): # idx stands for index (position)
        """
        Retrieves a single sample (image + label) from the dataset.
        This method is called by the DataLoader thousands of times to assemble batches.
        """
        img_name = self.image_path[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Rigorous check inside getitem might be too slow, trusting the initialization check.

        # Load image and convert to RGB (X-rays are usually grayscale, but Pre-trained CNNs expect 3 channels)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        '''
        Transforms our PIL image into a PyTorch tensor:
        1) Resize to 224x224 if not done already.
        2) Augmentation (slight variations like rotations, horizontal flips, zoom).
        3) Converts from Height x Width x Channels to Channels x Height x Width (PyTorch format).
        4) Normalizes tensor = (tensor - mean) / std for each RGB channel.
        '''
        labels = torch.tensor(self.labels[idx], dtype=torch.float32) # Convert numpy array to PyTorch tensor
        return image, labels
    

def get_transforms(image_size=224):
    """
    Defines the Image Augmentation Pipeline.
    Augmentation creates variations of images (flips, rotations) to prevent the model from memorizing exact pixels.
    It forces the model to learn structural features.
    """
    train_transform = transforms.Compose([ # Transform for training set
        transforms.RandomResizedCrop(image_size, scale=(0.9, 1.1)), # Zoom (Domain Aware)
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomRotation(degrees=7),   # Slight tilt
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Contrast modulation
        transforms.ToTensor(), 
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
        
    base_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), # Transform for validation and test sets
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return train_transform, base_transform, base_transform


def create_dataloader(train_df, val_df, test_df, image_dir, batch_size=32, num_workers=0, image_size=224, augment=True):
    train_tf, val_tf, test_tf = get_transforms(image_size)

    # If augmentation disabled, use base transform (val_tf) for training
    final_train_tf = train_tf if augment else val_tf
    
    if not augment:
        print("[DATA] Augmentations DISABLED (Ablation Mode).")

    train_set = ChestXRayDataset(train_df, image_dir, transform=final_train_tf)
    val_set = ChestXRayDataset(val_df, image_dir, transform=val_tf)
    test_set = ChestXRayDataset(test_df, image_dir, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # pin_memory optimizes performance when transferring data from CPU through to GPU

    return train_loader, val_loader, test_loader 
