"""
=============================================================================
                         LOSS.PY - Class Imbalance Handling
=============================================================================

PURPOSE:
    Implements a Weighted Binary Cross-Entropy (BCE) loss function that
    automatically adjusts for class imbalance in medical datasets.

THE IMBALANCE PROBLEM:

    In medical imaging, some conditions are MUCH rarer than others:
    
    | Condition      | Prevalence  | Problem                    |
    |---------------|-------------|----------------------------|
    | Infiltration  | ~20%        | Common, many examples      |
    | Hernia        | ~0.2%       | Rare, very few examples    |
    
    Standard BCE treats all errors equally:
    - Miss a Hernia (0.2% of data) = small loss
    - Miss an Infiltration (20% of data) = large loss
    
    Result: Model learns to IGNORE rare diseases (always predict 0)
            This achieves high accuracy but TERRIBLE clinical value!

THE SOLUTION - Weighted BCE:

    Increase the COST of missing rare diseases:
    
    Weight = (Negative Samples) / (Positive Samples)
    
    Example for Hernia (0.2% positive):
    - 998 negatives, 2 positives
    - Weight = 998 / 2 = 499
    - Missing ONE Hernia costs 499× more than a false positive
    
    This forces the model to PAY ATTENTION to rare diseases.

MATHEMATICAL FORMULATION:

    Standard BCE: L = -[y·log(p) + (1-y)·log(1-p)]
    
    Weighted BCE: L = -[w·y·log(p) + (1-y)·log(1-p)]
                       ↑
                       Weight only applied to POSITIVE class
    
    This increases the gradient signal when a positive sample is 
    misclassified, pushing the model to learn rare classes better.

=============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class WeightedBCELossWrapper(nn.Module):
    """
    A wrapper around BCEWithLogitsLoss that automatically calculates
    positive weights from the training dataframe.
    
    Why "WithLogits"?
        BCEWithLogitsLoss expects RAW LOGITS (unbounded values).
        It applies Sigmoid internally before computing BCE.
        This is numerically more stable than: sigmoid() → BCELoss()
    
    Usage:
        criterion = WeightedBCELossWrapper(train_df, label_list, device)
        loss = criterion(model_output, targets)  # model_output = raw logits
    """
    
    def __init__(self, train_df, label_list, device, max_weight=100.0):
        """
        Initialize the weighted loss function.
        
        Args:
            train_df (DataFrame): Training data with one-hot encoded labels.
                                 Used to count positive samples per class.
                                 
            label_list (list): Ordered list of class names from schema.
                              Must match the columns in train_df.
                              
            device (torch.device): 'cuda' or 'cpu' for tensor placement.
            
            max_weight (float): Maximum allowed weight (default: 100.0).
                               Prevents extreme weights for ultra-rare classes.
                               
        Weight Calculation:
            For each class i:
                num_pos = count of samples where class i = 1
                num_neg = count of samples where class i = 0
                weight_i = num_neg / max(num_pos, 1)
                
            The max(num_pos, 1) prevents division by zero if a class
            has zero positive samples in the training set.
        """
        super().__init__()
        self.device = device
        self.max_weight = max_weight
        self.label_list = label_list
        
        # Calculate weights from training data
        pos_weights = self._calculate_pos_weights(train_df, label_list)
        
        # Log weights for review (important for debugging imbalance)
        self._log_weights(pos_weights, label_list)
        
        # Convert to tensor and move to device (GPU/CPU)
        self.pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)
        
        # Initialize the actual loss function with computed weights
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor)

    def _calculate_pos_weights(self, df, labels):
        """
        Calculate positive class weights based on class frequencies.
        
        Formula: weight = num_negatives / num_positives
        
        Intuition:
        - If 50% positive, weight = 1.0 (no adjustment needed)
        - If 1% positive, weight = 99.0 (misses are 99× more costly)
        
        Args:
            df (DataFrame): Training dataframe with class columns
            labels (list): List of class names
            
        Returns:
            list: Weight for each class
        """
        weights = []
        total_samples = len(df)
        
        for label in labels:
            if label not in df.columns:
                # Safety fallback if label missing (shouldn't happen with schema)
                weights.append(1.0)
                continue
                
            num_pos = df[label].sum()  # Count of 1s in this column
            num_neg = total_samples - num_pos  # Count of 0s
            
            # ═══════════════════════════════════════════════════════════════
            # WEIGHT FORMULA: neg / max(pos, 1)
            # ═══════════════════════════════════════════════════════════════
            # Why max(pos, 1)?
            #   - Prevents division by zero if class has 0 positive samples
            #   - In this case, weight = total_samples (maximum penalty)
            #
            weight = num_neg / max(num_pos, 1)
            
            # ═══════════════════════════════════════════════════════════════
            # SAFETY CLAMP: Prevent extreme weights
            # ═══════════════════════════════════════════════════════════════
            # Ultra-rare classes could get weights of 1000+
            # This destabilizes training (exploding gradients)
            # We cap at max_weight for numerical stability
            #
            if weight > self.max_weight:
                weight = self.max_weight
                
            weights.append(weight)
            
        return weights

    def _log_weights(self, weights, labels):
        """
        Print a formatted table of class weights for user review.
        
        This is important for:
        1. Verifying weights make sense (rare classes should have high weights)
        2. Debugging training issues (weights too high → unstable gradients)
        3. Documentation of the training configuration
        """
        print("\n" + "="*40)
        print(" [INFO] CLASS IMBALANCE WEIGHTS (Max Clamp: {})".format(self.max_weight))
        print("="*40)
        print(f"{'Pathology':<20} | {'Weight':<10}")
        print("-" * 35)
        
        # Sort by weight descending to highlight rare classes
        sorted_indices = np.argsort(weights)[::-1]
        for idx in sorted_indices:
             print(f"{labels[idx]:<20} | {weights[idx]:.2f}")
        print("="*40 + "\n")

    def forward(self, input, target):
        """
        Compute the weighted BCE loss.
        
        Args:
            input (Tensor): Model predictions (RAW LOGITS), shape [Batch, NumClasses]
                           Do NOT apply sigmoid before passing to this function!
                           
            target (Tensor): Ground truth labels (0 or 1), shape [Batch, NumClasses]
            
        Returns:
            Tensor: Scalar loss value
            
        Note:
            The pos_weight is applied automatically by BCEWithLogitsLoss.
            It multiplies the log(p) term for positive samples, effectively
            making false negatives more costly than false positives.
        """
        return self.criterion(input, target)
