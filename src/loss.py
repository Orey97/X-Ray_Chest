import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class WeightedBCELossWrapper(nn.Module):
    """
    Wraps BCEWithLogitsLoss with dynamically calculated positive weights.
    Mitigates class imbalance by penalizing false negatives in rare classes.
    """
    def __init__(self, train_df, label_list, device, max_weight=100.0):
        super().__init__()
        self.device = device
        self.max_weight = max_weight
        self.label_list = label_list
        
        # Calculate weights
        pos_weights = self._calculate_pos_weights(train_df, label_list)
        
        # Log weights for review
        self._log_weights(pos_weights, label_list)
        
        # Convert to tensor
        self.pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)
        
        # Initialize standard BCE with weights
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor)

    def _calculate_pos_weights(self, df, labels):
        weights = []
        total_samples = len(df)
        
        for label in labels:
            if label not in df.columns:
                # Fallback if label missing (should be caught by schema, but safety first)
                weights.append(1.0)
                continue
                
            num_pos = df[label].sum()
            num_neg = total_samples - num_pos
            
            # Formula: neg / max(pos, 1)
            # Intuition: If pos is 1, weight is neg/1 ~ total. 
            # If pos is 50/50, weight is 1.0.
            weight = num_neg / max(num_pos, 1)
            
            # Safety Clamp
            if weight > self.max_weight:
                weight = self.max_weight
                
            weights.append(weight)
            
        return weights

    def _log_weights(self, weights, labels):
        print("\n" + "="*40)
        print(" [INFO] CLASS IMBALANCE WEIGHTS (Max Clamp: {})".format(self.max_weight))
        print("="*40)
        print(f"{'Pathology':<20} | {'Weight':<10}")
        print("-" * 35)
        
        sorted_indices = np.argsort(weights)[::-1] # descending
        for idx in sorted_indices:
             print(f"{labels[idx]:<20} | {weights[idx]:.2f}")
        print("="*40 + "\n")

    def forward(self, input, target):
        return self.criterion(input, target)
