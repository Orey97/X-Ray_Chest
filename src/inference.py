"""
=============================================================================
                    INFERENCE.PY - Production Inference Engine
=============================================================================

PURPOSE:
    Performs predictions on new chest X-ray images using a trained model.
    Implements strict schema validation to prevent silent failures.

USAGE:
    # Predict on a specific image
    python src/inference.py --image path/to/xray.png
    
    # Let it pick a random image from dataset
    python src/inference.py --random

SAFETY FEATURES:

    1. SCHEMA VALIDATION
       - Loads schema.json alongside the model
       - Verifies model output dimensions match schema
       - Ensures correct label interpretation
    
    2. FAIL FAST DESIGN
       - Missing model file → Clear error message
       - Missing schema → Refuses to run
       - Dimension mismatch → Raised exception
    
    3. CONFIDENCE THRESHOLDS
       - High Confidence (≥50%): Strong signal for pathology
       - Low Confidence (5-50%): Worth clinical review
       - Clean (< 5%): No significant findings

MEDICAL DISCLAIMER:
    This is a demonstration tool, NOT a medical device.
    Do NOT use for actual medical diagnosis without proper validation.

=============================================================================
"""

import torch
import os
import argparse
import random
import numpy as np
from PIL import Image
from model import MultiLabelResNet
from dataloader import get_transforms
from schema import DatasetManifest


def load_model_and_schema(model_path, device):
    """
    Load a trained model along with its schema contract.
    
    The schema is CRITICAL for correct inference:
    - It defines the exact label order
    - It validates model architecture
    - Without it, predictions would be meaningless numbers
    
    Args:
        model_path (str): Path to the .pth model weights file
        device: torch.device for model placement
        
    Returns:
        tuple: (model, manifest) - The loaded model and its schema
        
    Raises:
        FileNotFoundError: If model or schema files are missing
        RuntimeError: If model doesn't match schema dimensions
    """
    print(f"[INFERENCE] Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found at: {model_path}\n   -> Did you download it from Drive?")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: LOAD SCHEMA (STRICT REQUIREMENT)
    # ═══════════════════════════════════════════════════════════════════════
    # The schema MUST exist alongside the model. It defines:
    # - Number of output classes
    # - Label names and their ORDER
    # - Without schema, we can't interpret the model's output!
    
    model_dir = os.path.dirname(model_path)
    schema_path = os.path.join(model_dir, "schema.json")
    
    if not os.path.exists(schema_path):
        raise FileNotFoundError(
            f"❌ CRITICAL ERROR: schema.json not found at {schema_path}.\n"
            f"   Phase 1 Protocol: Inference is strictly forbidden without a versioned schema."
        )
    
    manifest = DatasetManifest.load(schema_path)
    print(f"[INFERENCE] ✅ Schema Loaded. Expecting {manifest.num_classes} classes.")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: INITIALIZE MODEL ARCHITECTURE
    # ═══════════════════════════════════════════════════════════════════════
    # The model architecture MUST match what was used during training.
    # Using num_classes from schema ensures consistency.
    
    model = MultiLabelResNet(num_classes=manifest.num_classes)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: LOAD TRAINED WEIGHTS
    # ═══════════════════════════════════════════════════════════════════════
    # map_location handles CPU/GPU compatibility:
    # - Model saved on GPU can be loaded on CPU
    # - Model saved on CPU can be loaded on GPU
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()  # Set to evaluation mode (disables dropout)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: FAIL FAST VALIDATION
    # ═══════════════════════════════════════════════════════════════════════
    # Verify the model's output layer matches the schema's expected classes.
    # If there's a mismatch, something is VERY wrong (wrong model loaded).
    
    manifest.validate_model(model)
    
    return model, manifest


def predict_single(model, manifest, image_path, device):
    """
    Predict pathologies for a single image.
    
    Pipeline:
    1. Load image from disk
    2. Apply same transforms used during training
    3. Run through model to get logits
    4. Apply sigmoid to convert to probabilities
    
    Args:
        model: The loaded neural network
        manifest: Schema with label information
        image_path: Path to the X-ray image
        device: torch.device
        
    Returns:
        numpy.array: Probabilities for each pathology (0.0 to 1.0)
                    Shape: [num_classes]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: GET VALIDATION TRANSFORMS
    # ═══════════════════════════════════════════════════════════════════════
    # Use the SAME transforms that were used during validation.
    # No augmentation - we want deterministic predictions.
    
    _, val_tf, _ = get_transforms()
    
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error opening image: {e}")
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: PREPARE INPUT TENSOR
    # ═══════════════════════════════════════════════════════════════════════
    # Transform: Resize → ToTensor → Normalize
    # unsqueeze(0) adds batch dimension: [C,H,W] → [1,C,H,W]
    
    img_tensor = val_tf(img).unsqueeze(0).to(device)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: RUN INFERENCE
    # ═══════════════════════════════════════════════════════════════════════
    with torch.no_grad():  # No gradient computation needed
        logits = model(img_tensor)
        
        # Validate output dimensions against schema
        manifest.validate_prediction(logits)
        
        # Convert logits to probabilities using sigmoid
        # Logits: (-∞, +∞) → Probabilities: (0, 1)
        probs = torch.sigmoid(logits)
        
    return probs.cpu().numpy()[0]  # Remove batch dimension


def print_report(probs, filename, manifest):
    """
    Print a formatted diagnostic report.
    
    Color-coded output:
    🔴 High Confidence (≥50%): Strong indication of pathology
    🟡 Low Confidence (5-50%): Worth clinical review
    🟢 Clean: No significant findings
    
    Args:
        probs: Probability array for each class
        filename: Name of the image file
        manifest: Schema with label names
    """
    print(f"\n{'='*40}")
    print(f" 🩺 DIAGNOSTIC REPORT: {filename}")
    print(f"{'='*40}")
    
    # Use labels from schema (canonical order)
    class_names = manifest.label_list
    
    # Sort by probability (highest first)
    pairs = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)
    
    # Confidence thresholds
    high_conf = 0.50  # Strong signal
    low_conf  = 0.05  # Worth noting
    
    findings = 0
    
    for label, score in pairs:
        pct = score * 100
        
        if score >= high_conf:
            print(f"  🔴 {label.upper():<20} {pct:.1f}%  (High Confidence)")
            findings += 1
        elif score >= low_conf:
            print(f"  🟡 {label.upper():<20} {pct:.1f}%  (Low Confidence)")
            findings += 1
            
    if findings == 0:
        print("  🟢 CLEAN SCAN (No significant findings)")
    
    print("-" * 40)
    print(f"Schema Version: {manifest.timestamp}")


def main():
    """Main entry point for inference CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Specific image path to test")
    parser.add_argument("--random", action="store_true", help="Pick a random image from default dir")
    args = parser.parse_args()

    # ═══════════════════════════════════════════════════════════════════════
    # PATH CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    # Default paths are relative to the repository root.
    # Assumes standard project structure: X-Ray_Chest/src/inference.py
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "output", "best_model.pth")
    IMAGES_DIR = os.path.join(BASE_DIR, "images-224")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Device: {device}")

    try:
        # Load model and schema
        print(f"[SYSTEM] Looking for model at: {MODEL_PATH}")
        model, manifest = load_model_and_schema(MODEL_PATH, device)
        
        # Determine which image to process
        target_path = None
        
        if args.image:
            target_path = os.path.abspath(args.image)
        else:
            # Auto-pick a random image from the dataset
            img_dir = os.path.abspath(IMAGES_DIR)
            if os.path.exists(img_dir):
                candidates = [f for f in os.listdir(img_dir) if f.endswith(".png")]
                if candidates:
                    target_path = os.path.join(img_dir, random.choice(candidates))
                    print(f"[AUTO] Selected random image from dataset.")
        
        if not target_path:
            print("❌ No image selected. Use --image PATH or ensure ../images-224 exists.")
            return

        # Run prediction and display results
        probs = predict_single(model, manifest, target_path, device)
        if probs is not None:
            print_report(probs, os.path.basename(target_path), manifest)

    except Exception as e:
        print(f"\n❌ FATAL: {e}")


if __name__ == "__main__":
    main()
