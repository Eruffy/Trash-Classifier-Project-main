import cv2
import numpy as np
import joblib
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features.feature_extractor import extract_features

# ---------------- Load Models ----------------
print("[INFO] Loading models...")
super_svm = joblib.load("saved_models/super_svm_aug.pkl")
super_scaler = joblib.load("saved_models/super_svm_scaler_aug.pkl")
fiber_svm = joblib.load("saved_models/fiber_svm_aug.pkl")
fiber_scaler = joblib.load("saved_models/fiber_svm_scaler_aug.pkl")
rigid_svm = joblib.load("saved_models/rigid_svm_aug.pkl")
rigid_scaler = joblib.load("saved_models/rigid_svm_scaler_aug.pkl")
print("[SUCCESS] Models loaded!\n")

# Thresholds
T_SUPER = 0.60
T_FINE = 0.65

CLASS_NAMES = {
    0: "Glass",
    1: "Paper",
    2: "Cardboard",
    3: "Plastic",
    4: "Metal",
    5: "Trash",
    6: "Unknown"
}

# ---------- Prediction Function ----------
def predict_material(feature):
    # Stage 1: Super-class
    x = super_scaler.transform([feature])
    super_probs = super_svm.predict_proba(x)[0]
    super_cls = np.argmax(super_probs)
    super_conf = super_probs[super_cls]
    
    if super_conf < T_SUPER:
        return 6  # Unknown
    
    # Stage 2: Fine-class
    if super_cls == 0:  # Fiber
        x2 = fiber_scaler.transform([feature])
        probs = fiber_svm.predict_proba(x2)[0]
        pred = np.argmax(probs)
        if probs[pred] >= T_FINE:
            return 1 if pred == 0 else 2  # Paper or Cardboard
        return 6
    
    if super_cls == 1:  # Rigid
        x2 = rigid_scaler.transform([feature])
        probs = rigid_svm.predict_proba(x2)[0]
        pred = np.argmax(probs)
        if probs[pred] >= T_FINE:
            return 3 if pred == 0 else 4  # Plastic or Metal
        return 6
    
    if super_cls == 2:  # Transparent
        return 0  # Glass
    
    if super_cls == 3:  # Garbage
        return 5  # Trash
    
    return 6  # Unknown

# ---------- Video Capture ----------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Starting real-time classification. Press 'q' to quit.\n")

frame_count = 0
last_prediction = "Unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Predict every 5 frames for better performance
    if frame_count % 5 == 0:
        # Resize frame for feature extraction (center crop)
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        size = 256
        half = size // 2
        cropped = frame[cy-half:cy+half, cx-half:cx+half]
        
        # Extract features
        try:
            feature = extract_features(cropped)
            class_id = predict_material(feature)
            last_prediction = CLASS_NAMES[class_id]
        except Exception as e:
            last_prediction = "Error"
            print(f"[ERROR] {e}")
    
    frame_count += 1
    
    # Display prediction on frame
    cv2.putText(frame, f"Material: {last_prediction}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Waste Classifier - Real-Time", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n[INFO] Camera closed.")
