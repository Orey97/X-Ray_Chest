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
    print(f"[INFERENCE] Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found at: {model_path}\n   -> Did you download it from Drive?")

    # 1. LOAD SCHEMA (STRICT REQUIREMENT)
    model_dir = os.path.dirname(model_path)
    schema_path = os.path.join(model_dir, "schema.json")
    
    if not os.path.exists(schema_path):
        raise FileNotFoundError(
            f"❌ CRITICAL ERROR: schema.json not found at {schema_path}.\n"
            f"   Phase 1 Protocol: Inference is strictly forbidden without a versioned schema."
        )
    
    manifest = DatasetManifest.load(schema_path)
    print(f"[INFERENCE] ✅ Schema Loaded. Expecting {manifest.num_classes} classes.")

    # 2. Initialize Architecture with Correct Heads
    model = MultiLabelResNet(num_classes=manifest.num_classes)
    
    # 3. Load Weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    # 4. FAIL FAST: Validate Architecture against Schema
    manifest.validate_model(model)
    
    return model, manifest

def predict_single(model, manifest, image_path, device):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # 1. Image Prep
    _, val_tf, _ = get_transforms()
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error opening image: {e}")
        return None

    # 2. Transform & Batch
    img_tensor = val_tf(img).unsqueeze(0).to(device)

    # 3. Forecast
    with torch.no_grad():
        logits = model(img_tensor)
        
        # 4. FAIL FAST: Validate Output Dimension
        manifest.validate_prediction(logits)
        
        probs = torch.sigmoid(logits)
        
    return probs.cpu().numpy()[0]

def print_report(probs, filename, manifest):
    print(f"\n{'='*40}")
    print(f" 🩺 DIAGNOSTIC REPORT: {filename}")
    print(f"{'='*40}")
    
    # Use Manifest Labels, NOT hardcoded list
    class_names = manifest.label_list
    
    # Zip and Sort by Probability
    pairs = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)
    
    # Thresholds
    high_conf = 0.50
    low_conf  = 0.05
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Specific image path to test")
    parser.add_argument("--random", action="store_true", help="Pick a random image from default dir")
    args = parser.parse_args()

    # PATHS CONFIGURATION
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "output", "best_model.pth")
    IMAGES_DIR = os.path.join(BASE_DIR, "images-224")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Device: {device}")

    try:
        # Load Brain & Schema
        print(f"[SYSTEM] Looking for model at: {MODEL_PATH}")
        model, manifest = load_model_and_schema(MODEL_PATH, device)
        
        # Select Target
        target_path = None
        
        if args.image:
            target_path = os.path.abspath(args.image)
        else:
            # Auto-Pick Random
            img_dir = os.path.abspath(IMAGES_DIR)
            if os.path.exists(img_dir):
                candidates = [f for f in os.listdir(img_dir) if f.endswith(".png")]
                if candidates:
                    target_path = os.path.join(img_dir, random.choice(candidates))
                    print(f"[AUTO] Selected random image from dataset.")
        
        if not target_path:
            print("❌ No image selected. Use --image PATH or ensure ../images-224 exists.")
            return

        # Execute
        probs = predict_single(model, manifest, target_path, device)
        if probs is not None:
            print_report(probs, os.path.basename(target_path), manifest)

    except Exception as e:
        print(f"\n❌ FATAL: {e}")

if __name__ == "__main__":
    main()
