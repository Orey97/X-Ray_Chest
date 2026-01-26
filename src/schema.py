import json
import hashlib
import os
import pandas as pd
import torch

class DatasetManifest:
    """
    The Single Source of Truth for the Dataset Schema.
    Enforces strict compatibility between Data, Model, and Inference.
    """
    def __init__(self, label_list, metadata_checksum=None, timestamp=None):
        self.label_list = label_list
        self.num_classes = len(label_list)
        self.metadata_checksum = metadata_checksum
        self.timestamp = timestamp

    @classmethod
    def create(cls, csv_path, label_col="label"):
        """
        Creates a new Manifest by scanning the CSV.
        Defines the CANONICAL label order.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        # 1. Calculate Checksum (Metadata Only)
        md5 = hashlib.md5()
        with open(csv_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        checksum = md5.hexdigest()

        # 2. Extract Labels to define Canonical Order
        df = pd.read_csv(csv_path)
        
        # Auto-clean column names to match dataset.py logic
        df.columns = (
            df.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_")
                .str.replace("(", "")
                .str.replace(")", "")
        )
        
        # Handle "finding_labels" vs "label"
        target_col = label_col
        if label_col not in df.columns and "finding_labels" in df.columns:
            target_col = "finding_labels"
        
        if target_col not in df.columns:
             raise ValueError(f"Label column '{label_col}' (or 'finding_labels') not found in CSV.")

        # Parse labels (logic from dataset.py)
        # "Pneumonia|Edema" -> ["Pneumonia", "Edema"]
        raw_labels = df[target_col].astype(str).str.replace(" ", "").str.split('|')
        unique_labels = sorted(set(l for sublist in raw_labels for l in sublist if l != ''))
        
        # Remove 'NoFinding' if it's considered a negative class (Optional, but keeping consistent with Typical Multi-label)
        # However, typically 'No Finding' is a class. Let's keep it if dataset.py kept it.
        # dataset.py used "sorted(set(...))". We strictly follow that.
        
        return cls(unique_labels, metadata_checksum=checksum, timestamp=pd.Timestamp.now().isoformat())

    def save(self, path):
        """Persists the schema to JSON."""
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
        """Loads a strict schema from JSON."""
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
        FAIL FAST: Checks if model architecture matches this schema.
        """
        # Assume ResNet-like structure (model.fc or model.base.fc)
        # We need to handle the specific MultiLabelResNet wrapper which keeps .base
        # But wait, model.py: MultiLabelResNet.base.fc is the linear layer.
        
        # Introspect to find the final layer
        fc = None
        if hasattr(model, 'base') and hasattr(model.base, 'fc'):
            fc = model.base.fc
        elif hasattr(model, 'fc'):
            fc = model.fc
            
        if fc is None:
            # Could not confirm, warn but maybe don't crash if custom arch? 
            # No, strict mode: if we can't validate, we warn loudly.
            print("[MANIFEST] [WARNING] Could not locate final linear layer 'fc' to validate dimensions.")
            return

        if fc.out_features != self.num_classes:
            raise RuntimeError(
                f"❌ CRITICAL SCHEMA MISMATCH: Model has {fc.out_features} output neurons, "
                f"but Schema defines {self.num_classes} classes.\n"
                f"Schema Labels: {self.label_list}"
            )
        print("[MANIFEST] [OK] Model dimensions match schema.")

    def validate_prediction(self, logits):
        """
        FAIL FAST: Checks if inference output matches strict dimensions.
        Expects logits shape: [Batch, NumClasses] or [NumClasses]
        """
        if isinstance(logits, torch.Tensor):
            width = logits.shape[-1]
        else:
            width = logits.shape[-1] # numpy
            
        if width != self.num_classes:
            raise RuntimeError(
                f"❌ CRITICAL INFERENCE MISMATCH: Prediction vector has size {width}, "
                f"expected {self.num_classes}."
            )
