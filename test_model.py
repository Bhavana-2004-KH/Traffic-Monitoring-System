"""
Independent test script for validating the custom trained YOLOv8 model (best.pt).
Verifies the model file existence, loads the model, and inspects its classes.
"""
import os
import sys
from ultralytics import YOLO
import config

def validate_model():
    print("=" * 60)
    print("TESTING CUSTOM TRAINED YOLOv8 MODEL")
    print("=" * 60)
    
    model_path = config.MODEL_PATH
    print(f"Model path: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at: {model_path}")
        sys.exit(1)
        
    print(f"Model file size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)
        
    classes = model.names
    num_classes = len(classes)
    
    print(f"\nNumber of classes detected in model: {num_classes}")
    print("Model classes mapping:")
    for class_id, class_name in classes.items():
        print(f"  [{class_id}] -> {class_name}")
        
    expected_classes = config.VEHICLE_CLASSES
    print("\nVerifying classes against expected project specification:")
    all_matched = True
    for cid, cname in expected_classes.items():
        actual_name = classes.get(cid, "MISSING")
        matches = actual_name.lower().replace("-", " ") == cname.lower().replace("-", " ")
        status = "OK" if matches else "MISMATCH"
        if not matches:
            all_matched = False
        print(f"  Class {cid}: Expected '{cname}' | Got '{actual_name}' [{status}]")
        
    if all_matched:
        print("\nSUCCESS: All 7 vehicle classes strictly match project requirements!")
    else:
        print("\nWARNING: Some class names differed. Verify training configuration.")
        
    print("=" * 60)
    return model

if __name__ == "__main__":
    validate_model()
