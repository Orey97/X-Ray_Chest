
import torch
import torch.nn as nn
from torchvision import models

class MultiLabelResNet(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(MultiLabelResNet, self).__init__()
        # Load pre-trained ResNet50
        # Transfer Learning: We use weights learned from ImageNet.
        # This gives the model a "head start" in understanding shapes and textures.
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.base = models.resnet50(weights=weights)
        
        # Replace the final fully connected layer
        # ResNet50's original fc layer input features is 2048 and output is 1000 (ImageNet classes).
        # We perform "Surgery" here: removing the 1000-class head and attaching our 14-class head.
        in_features = self.base.fc.in_features
        self.base.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # We process the input through the ResNet backbone
        # The output is raw logits (numerical scores from -inf to +inf).
        # IMPORTANT: We do NOT apply Sigmoid here. 
        # The loss function 'BCEWithLogitsLoss' includes a Sigmoid layer internally.
        # This is numerically more stable than doing Sigmoid + BCELoss separately.
        return self.base(x)

if __name__ == "__main__":
    # Simple test to verify the model implementation
    model = MultiLabelResNet(num_classes=14)
    print(model)
    
    # Test with a dummy input
    dummy_input = torch.randn(1, 3, 224, 224) # Batch size 1, RGB, 224x224
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be [1, 14]
