
import torch
import torch.nn as nn
from torchvision import models

class MultiLabelResNet(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(MultiLabelResNet, self).__init__()
        # Load pre-trained ResNet50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet50(weights=weights)
        
        # Replace the final fully connected layer
        # ResNet50's fc layer input features is 2048
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # We process the input through the ResNet backbone
        # The output is raw logits (not probabilities) because 
        # BCEWithLogitsLoss will handle the sigmoid activation internally for numerical stability.
        return self.resnet(x)

if __name__ == "__main__":
    # Simple test to verify the model implementation
    model = MultiLabelResNet(num_classes=14)
    print(model)
    
    # Test with a dummy input
    dummy_input = torch.randn(1, 3, 224, 224) # Batch size 1, RGB, 224x224
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be [1, 14]
