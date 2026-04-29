"""
Train Fine-Class k-NN Classifiers
Trains separate k-NN models for each super-class
"""

import os
import sys
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from features.feature_extractor import extract_features
from config import (
    DATASET_PATH, MODEL_DIR, CLASS_MAPPING, SUPER_CLASS_MAPPING,
    TEST_SIZE, RANDOM_STATE, KNN_N_NEIGHBORS, KNN_WEIGHTS
)


def load_data_for_super_class(super_class_id):
    """Load images belonging to a specific super-class"""
    X = []
    y = []
    
    # Define which fine-classes belong to this super-class
    super_class_members = {
        0: ['paper', 'cardboard'],      # Fiber
        1: ['plastic', 'metal'],         # Rigid
        2: ['glass'],                    # Transparent
        3: ['trash']                     # Garbage
    }
    
    classes = super_class_members.get(super_class_id, [])
    if not classes:
        return np.array([]), np.array([])
    
    for class_name in classes:
        class_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(class_path):
            print(f"[WARNING] Class folder not found: {class_path}")
            continue
        
        class_id = CLASS_MAPPING[class_name]
        print(f"[INFO] Loading {class_name} (ID: {class_id})...")
        
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img = cv2.resize(img, (224, 224))
                features = extract_features(img)
                
                X.append(features)
                y.append(class_id)
                
            except Exception as e:
                print(f"[WARNING] Error processing {img_path}: {e}")
                continue
        
        print(f"  Loaded {len([label for label in y if label == class_id])} samples")
    
    return np.array(X), np.array(y)


def train_fine_knn(super_class_id, super_class_name):
    """Train fine-class k-NN for a specific super-class"""
    print(f"\n{'='*60}")
    print(f"TRAINING FINE-CLASS K-NN FOR: {super_class_name}")
    print(f"{'='*60}")
    
    # Load data
    print(f"\n[STEP 1] Loading {super_class_name} classes...")
    X, y = load_data_for_super_class(super_class_id)
    
    if len(X) == 0:
        print(f"[WARNING] No data for {super_class_name}, skipping...")
        return
    
    if len(np.unique(y)) < 2:
        print(f"[WARNING] Only one class in {super_class_name}, skipping k-NN...")
        return
    
    print(f"Total samples: {len(X)}")
    print(f"Feature dimensions: {X.shape[1]}")
    print(f"Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    class_id_to_name = {v: k for k, v in CLASS_MAPPING.items()}
    for cls, count in zip(unique, counts):
        print(f"  {class_id_to_name[cls]}: {count} samples")
    
    # Split data
    print("\n[STEP 2] Splitting train/validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Standardize
    print("\n[STEP 3] Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train k-NN
    print(f"\n[STEP 4] Training k-NN (k={KNN_N_NEIGHBORS}, weights={KNN_WEIGHTS})...")
    knn = KNeighborsClassifier(
        n_neighbors=KNN_N_NEIGHBORS,
        weights=KNN_WEIGHTS,
        algorithm='auto',
        n_jobs=-1
    )
    knn.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("\n[STEP 5] Evaluating on validation set...")
    y_pred = knn.predict(X_val_scaled)
    
    target_names = [class_id_to_name[cls] for cls in np.unique(y)]
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=target_names, zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    
    # Save models
    print("\n[STEP 6] Saving models...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_name = super_class_name.lower()
    knn_path = os.path.join(MODEL_DIR, f'{model_name}_knn.joblib')
    scaler_path = os.path.join(MODEL_DIR, f'{model_name}_scaler_knn.joblib')
    
    joblib.dump(knn, knn_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"  k-NN saved to: {knn_path}")
    print(f"  Scaler saved to: {scaler_path}")


def main():
    print("="*60)
    print("TRAINING ALL FINE-CLASS K-NN CLASSIFIERS")
    print("="*60)
    
    super_classes = [
        (0, 'fiber'),
        (1, 'rigid'),
        (2, 'transparent'),
        (3, 'garbage')
    ]
    
    for super_id, super_name in super_classes:
        train_fine_knn(super_id, super_name)
    
    print("\n" + "="*60)
    print("ALL FINE-CLASS K-NN TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
