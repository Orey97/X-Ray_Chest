"""
=============================================================================
                      SCHEMA.PY - Data Contract Enforcement
=============================================================================

PURPOSE:
    Implements a "Schema Contract" pattern that ensures consistency between:
    - Training data (CSV labels)
    - Model architecture (output neurons)  
    - Inference pipeline (prediction interpretation)

THE PROBLEM THIS SOLVES:
    
    Scenario: You train a model on Monday, deploy on Friday.
    
    ❌ WITHOUT SCHEMA:
        - Training CSV had columns: ["Pneumonia", "Edema", "Hernia", ...]
        - Someone reorders the CSV columns
        - Model predicts index 0 = "Pneumonia", but CSV column 0 is now "Atelectasis"
        - SILENT FAILURE: Model "works" but predictions are WRONG
    
    ✅ WITH SCHEMA:
        - Schema locks the CANONICAL label order at training time
        - Schema is saved alongside model (.pth + schema.json)
        - Inference MUST load schema and validate dimensions
        - Any mismatch → FAIL FAST with clear error message

DESIGN PATTERN: "Contract-Based Programming"
    
    The schema acts as a CONTRACT between components:
    - TRAINING says: "I produce models with THIS exact output structure"
    - INFERENCE says: "I expect inputs with THIS exact structure"
    - If structures don't match → Reject immediately, don't guess
    
=============================================================================
"""

import json
import hashlib
import os
import pandas as pd
import torch


class DatasetManifest:
    """
    The Single Source of Truth for the Dataset Schema.
    
    This class enforces strict compatibility between:
    - Data processing pipeline (dataset.py)
    - Model architecture (model.py)
    - Inference engine (inference.py)
    
    Think of it as a "contract" that all components must honor.
    
    Key Properties:
        label_list: Ordered list of class names (determines prediction indices)
        num_classes: Total number of pathologies (must match model output)
        metadata_checksum: MD5 hash of training CSV (detects data changes)
        timestamp: When the schema was created (version tracking)
    """
    
    def __init__(self, label_list, metadata_checksum=None, timestamp=None):
        """
        Initialize a DatasetManifest.
        
        Args:
            label_list (list): Ordered list of class labels. 
                              ORDER MATTERS! Index 0 = first label, etc.
                              Example: ["Atelectasis", "Cardiomegaly", ..., "Pneumothorax"]
                              
            metadata_checksum (str): MD5 hash of the source CSV.
                                    Used to detect if underlying data changed.
                                    
            timestamp (str): ISO format timestamp of creation.
                            Example: "2026-01-27T10:30:00"
        """
        self.label_list = label_list
        self.num_classes = len(label_list)
        self.metadata_checksum = metadata_checksum
        self.timestamp = timestamp

    @classmethod
    def create(cls, csv_path, label_col="label"):
        """
        Create a NEW manifest by scanning the training CSV.
        
        This establishes the CANONICAL label order that will be enforced
        throughout the entire pipeline.
        
        Args:
            csv_path (str): Path to the training CSV file
            label_col (str): Name of the column containing labels
        
        Returns:
            DatasetManifest: A new manifest instance
            
        Process:
            1. Compute MD5 checksum of CSV (for change detection)
            2. Parse multi-label column ("Pneumonia|Edema" → ["Pneumonia", "Edema"])
            3. Collect all unique labels across entire dataset
            4. Sort alphabetically for consistent ordering
            5. Store as the CANONICAL order
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: Calculate MD5 Checksum
        # ═══════════════════════════════════════════════════════════════════
        # Why? To detect if the underlying data changes between training and
        # inference. If someone modifies the CSV, the checksum will differ.
        
        md5 = hashlib.md5()
        with open(csv_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        checksum = md5.hexdigest()

        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: Parse CSV and Extract Labels
        # ═══════════════════════════════════════════════════════════════════
        df = pd.read_csv(csv_path)
        
        # Clean column names to match dataset.py logic (lowercase, no spaces)
        df.columns = (
            df.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_")
                .str.replace("(", "")
                .str.replace(")", "")
        )
        
        # Handle different column naming conventions
        # NIH dataset uses "finding_labels", others might use "label"
        target_col = label_col
        if label_col not in df.columns and "finding_labels" in df.columns:
            target_col = "finding_labels"
        
        if target_col not in df.columns:
             raise ValueError(f"Label column '{label_col}' (or 'finding_labels') not found in CSV.")

        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: Extract Unique Labels
        # ═══════════════════════════════════════════════════════════════════
        # Labels are stored as pipe-separated strings: "Pneumonia|Edema|Atelectasis"
        # We split and collect ALL unique labels across the entire dataset
        
        raw_labels = df[target_col].astype(str).str.replace(" ", "").str.split('|')
        unique_labels = sorted(set(l for sublist in raw_labels for l in sublist if l != ''))
        
        # sorted() ensures CONSISTENT ordering across different runs
        # Without sorting, set() iteration order could vary!
        
        return cls(unique_labels, metadata_checksum=checksum, timestamp=pd.Timestamp.now().isoformat())

    def save(self, path):
        """
        Persist the schema to a JSON file.
        
        This file MUST be saved alongside the model weights (.pth).
        During inference, both files are required.
        
        Args:
            path (str): Output path for schema.json
            
        Output Format:
            {
                "label_list": ["Atelectasis", "Cardiomegaly", ...],
                "num_classes": 14,
                "metadata_checksum": "a1b2c3d4e5f6...",
                "timestamp": "2026-01-27T10:30:00",
                "version": "1.0"
            }
        """
        data = {
            "label_list": self.label_list,
            "num_classes": self.num_classes,
            "metadata_checksum": self.metadata_checksum,
            "timestamp": self.timestamp,
            "version": "1.0"
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"[MANIFEST] Schema saved to {path} (Checksum: {self.metadata_checksum[:8]}...)")

    @classmethod
    def load(cls, path):
        """
        Load an existing schema from JSON.
        
        This is the CRITICAL step during inference:
        - Load the schema that was created during training
        - Use it to interpret model outputs correctly
        
        Args:
            path (str): Path to schema.json
            
        Returns:
            DatasetManifest: Loaded manifest instance
            
        Raises:
            FileNotFoundError: If schema.json doesn't exist
                              (This is a FATAL error - inference cannot proceed)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Schema artifact not found at {path}. Model is missing its contract.")
        
        with open(path, 'r') as f:
            data = json.load(f)
            
        return cls(
            label_list=data["label_list"],
            metadata_checksum=data.get("metadata_checksum"),
            timestamp=data.get("timestamp")
        )

    def validate_model(self, model):
        """
        FAIL FAST: Verify model architecture matches this schema.
        
        This catches a critical error: loading a model trained on different
        data than expected. For example:
        - Schema says: 14 classes
        - Model has: 5 output neurons
        - This MUST fail immediately, not silently produce garbage
        
        Args:
            model: PyTorch model to validate
            
        Raises:
            RuntimeError: If model output dimensions don't match schema
        """
        # Find the final linear layer (handles different architectures)
        fc = None
        if hasattr(model, 'base') and hasattr(model.base, 'fc'):
            fc = model.base.fc  # Our MultiLabelResNet structure
        elif hasattr(model, 'fc'):
            fc = model.fc  # Standard ResNet structure
            
        if fc is None:
            print("[MANIFEST] [WARNING] Could not locate final linear layer 'fc' to validate dimensions.")
            return

        # ═══════════════════════════════════════════════════════════════════
        # CRITICAL CHECK: Do output neurons match schema classes?
        # ═══════════════════════════════════════════════════════════════════
        if fc.out_features != self.num_classes:
            raise RuntimeError(
                f"❌ CRITICAL SCHEMA MISMATCH: Model has {fc.out_features} output neurons, "
                f"but Schema defines {self.num_classes} classes.\n"
                f"Schema Labels: {self.label_list}"
            )
        print("[MANIFEST] [OK] Model dimensions match schema.")

    def validate_prediction(self, logits):
        """
        FAIL FAST: Verify inference output matches expected dimensions.
        
        This catches runtime errors where a different model produces
        predictions incompatible with the loaded schema.
        
        Args:
            logits (Tensor): Model output, shape [Batch, NumClasses] or [NumClasses]
            
        Raises:
            RuntimeError: If prediction vector size doesn't match schema
        """
        if isinstance(logits, torch.Tensor):
            width = logits.shape[-1]
        else:
            width = logits.shape[-1]  # numpy array
            
        if width != self.num_classes:
            raise RuntimeError(
                f"❌ CRITICAL INFERENCE MISMATCH: Prediction vector has size {width}, "
                f"expected {self.num_classes}."
            )

    def __repr__(self):
        """String representation for debugging."""
        return f"DatasetManifest(classes={self.num_classes}, version={self.timestamp})"
