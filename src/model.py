"""
=============================================================================
                        MODEL.PY - Neural Network Architecture
=============================================================================

PURPOSE:
    Defines the core deep learning model for multi-label chest X-ray 
    classification. Uses Transfer Learning from ImageNet pretrained weights.

ARCHITECTURE:
    ResNet-50 Backbone → Custom 14-class Classification Head
    
    Why ResNet-50?
    - Deep enough to learn complex patterns (50 layers)
    - Skip connections prevent vanishing gradients
    - Pretrained on 1M+ ImageNet images = excellent feature extraction
    - Well-documented and battle-tested in medical imaging

TRANSFER LEARNING STRATEGY:
    1. Load ResNet-50 with ImageNet weights (general features: edges, textures)
    2. "Freeze" early layers (keep ImageNet knowledge)
    3. Replace final classification head (1000 classes → 14 pathologies)
    4. Fine-tune on chest X-rays

WHY NOT SOFTMAX?
    Standard image classification uses Softmax (mutually exclusive classes).
    Medical diagnosis is MULTI-LABEL: a patient can have BOTH Pneumonia AND Effusion.
    We use independent Sigmoid per class = each pathology is a binary decision.

=============================================================================
"""

import torch
import torch.nn as nn
from torchvision import models


class MultiLabelResNet(nn.Module):
    """
    A ResNet-50 based model adapted for multi-label classification.
    
    Multi-Label vs Multi-Class:
    - Multi-Class: Image belongs to ONE class (e.g., cat OR dog)
    - Multi-Label: Image can have MULTIPLE labels (e.g., has_cat AND has_dog)
    
    In chest X-rays, a single scan can show multiple conditions simultaneously.
    Example: Patient has Cardiomegaly (enlarged heart) + Effusion (fluid) + Edema
    
    Attributes:
        base (nn.Module): The ResNet-50 backbone with modified final layer
    """
    
    def __init__(self, num_classes=14, pretrained=True):
        """
        Initialize the multi-label classification model.
        
        Args:
            num_classes (int): Number of pathologies to detect (default: 14)
                               The 14 NIH classes: Atelectasis, Cardiomegaly, 
                               Consolidation, Edema, Effusion, Emphysema, Fibrosis,
                               Hernia, Infiltration, Mass, Nodule, Pleural_Thickening,
                               Pneumonia, Pneumothorax
            
            pretrained (bool): If True, load ImageNet pretrained weights.
                              Always use True unless you have millions of X-rays!
                              
        Transfer Learning Rationale:
            ResNet-50 trained on ImageNet learned to detect:
            - Low-level: Edges, gradients, textures
            - Mid-level: Shapes, patterns, contours
            - High-level: Object parts, compositions
            
            These features TRANSFER well to medical imaging because X-rays
            also contain edges (ribs), textures (lung tissue), and shapes (heart).
        """
        super(MultiLabelResNet, self).__init__()
        
        # Load pre-trained ResNet50 with ImageNet weights
        # Modern PyTorch uses `weights=` instead of deprecated `pretrained=`
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.base = models.resnet50(weights=weights)
        
        # ═══════════════════════════════════════════════════════════════════
        # HEAD SURGERY: Replace the final classification layer
        # ═══════════════════════════════════════════════════════════════════
        # 
        # Original ResNet-50 structure:
        #   [Backbone Layers] → [AdaptiveAvgPool] → [FC: 2048 → 1000]
        #                                            ↑
        #                                      ImageNet classes
        #
        # We replace the final FC layer to output our pathologies:
        #   [Backbone Layers] → [AdaptiveAvgPool] → [FC: 2048 → 14]
        #                                            ↑
        #                                      Our pathologies
        #
        in_features = self.base.fc.in_features  # 2048 for ResNet-50
        self.base.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Batch of images, shape [Batch, 3, 224, 224]
                       - 3 channels (RGB, even though X-rays are grayscale)
                       - 224x224 is the ImageNet standard size
        
        Returns:
            Tensor: Raw logits, shape [Batch, NumClasses]
                   - NOT probabilities! These are unbounded values (-∞ to +∞)
                   - Convert to probabilities using: torch.sigmoid(logits)
        
        IMPORTANT: Why return logits instead of probabilities?
        
        1. Numerical Stability:
           BCEWithLogitsLoss combines Sigmoid + BCE in one numerically stable op.
           Doing sigmoid() then BCE() separately can cause precision loss.
           
        2. Flexibility:
           Inference can apply different thresholds post-sigmoid.
           Training remains numerically optimal.
           
        3. Performance:
           GPU can fuse sigmoid into loss computation = faster training.
        """
        return self.base(x)


# =============================================================================
# SELF-TEST: Verify the model works correctly
# =============================================================================
# Run with: python src/model.py

if __name__ == "__main__":
    print("=" * 60)
    print(" MODEL ARCHITECTURE TEST")
    print("=" * 60)
    
    # Create model
    model = MultiLabelResNet(num_classes=14)
    print(f"\n✓ Model created successfully")
    print(f"  - Architecture: ResNet-50")
    print(f"  - Output classes: 14")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)  # [Batch=1, Channels=3, H=224, W=224]
    output = model(dummy_input)
    
    print(f"\n✓ Forward pass successful")
    print(f"  - Input shape: {dummy_input.shape}")
    print(f"  - Output shape: {output.shape}")  # Should be [1, 14]
    
    # Verify output dimensions
    assert output.shape == (1, 14), f"Expected [1, 14], got {output.shape}"
    print(f"\n✓ All tests passed!")
    print("=" * 60)
