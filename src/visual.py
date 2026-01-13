
import matplotlib.pyplot as plt
import numpy as np
import torch

def show_batch(dataloader, class_names, num_images=4):
    """
    Visualizes a batch of images from the dataloader with their labels.
    """
    images, labels = next(iter(dataloader))
    
    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        ax = plt.subplot(1, num_images, i + 1)
        
        # Un-normalize image for display
        # Standard ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        inp = images[i].numpy().transpose((1, 2, 0)) # Convert CHW to HWC
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        
        plt.imshow(inp)
        
        # Get active labels
        active_labels = [class_names[j] for j, label in enumerate(labels[i]) if label == 1]
        title = "\n".join(active_labels) if active_labels else "No Finding"
        
        plt.title(title)
        plt.axis("off")
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage (requires dataloader setup in main)
    print("This module is intended to be imported and used with an active dataloader.")
