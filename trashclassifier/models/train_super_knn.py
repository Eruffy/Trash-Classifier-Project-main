"""
Train Super-Class k-NN Classifier
Classifies into super-classes: Fiber, Rigid, Transparent, Garbage
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
from data.augmentations import augment_dataset
from config import (
    DATASET_PATH, MODEL_DIR, CLASS_MAPPING, SUPER_CLASS_MAPPING,
    CLASS_TARGETS, TEST_SIZE, RANDOM_STATE, KNN_N_NEIGHBORS, KNN_WEIGHTS
)


def load_data_with_super_labels():
    """Load images and assign super-class labels"""
    X = []
    y = []
    
    for class_name, class_id in CLASS_MAPPING.items():
        if class_name == 'unknown':
            continue
            
        class_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(class_path):
            print(f"[WARNING] Class folder not found: {class_path}")
            continue
        
        # Get super-class label
        super_class = SUPER_CLASS_MAPPING.get(class_name, 3)  # Default to Garbage
        
        print(f"[INFO] Loading {class_name} (super-class: {super_class})...")
        
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                # Resize to 224x224
                img = cv2.resize(img, (224, 224))
                
                # Extract features
                features = extract_features(img)
                
                X.append(features)
                y.append(super_class)
                
            except Exception as e:
                print(f"[WARNING] Error processing {img_path}: {e}")
                continue
        
        print(f"  Loaded {len([label for label in y if label == super_class])} samples")
    
    return np.array(X), np.array(y)


def main():
    print("="*60)
    print("TRAINING SUPER-CLASS K-NN CLASSIFIER")
    print("="*60)
    
    # Load data
    print("\n[STEP 1] Loading dataset...")
    X, y = load_data_with_super_labels()
    
    if len(X) == 0:
        print("[ERROR] No data loaded!")
        return
    
    print(f"Total samples: {len(X)}")
    print(f"Feature dimensions: {X.shape[1]}")
    print(f"Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    super_names = ['Fiber', 'Rigid', 'Transparent', 'Garbage']
    for cls, count in zip(unique, counts):
        print(f"  {super_names[cls]}: {count} samples")
    
    # Split data
    print("\n[STEP 2] Splitting train/validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Standardize features
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
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=super_names, zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    
    # Calculate per-class recall
    print("\nPer-Class Recall:")
    for i, name in enumerate(super_names):
        if i < len(cm):
            recall = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
            print(f"  {name}: {recall:.2%}")
    
    # Save models
    print("\n[STEP 6] Saving models...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    knn_path = os.path.join(MODEL_DIR, 'super_knn.joblib')
    scaler_path = os.path.join(MODEL_DIR, 'super_knn_scaler.joblib')
    
    joblib.dump(knn, knn_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"  k-NN saved to: {knn_path}")
    print(f"  Scaler saved to: {scaler_path}")
    
    print("\n" + "="*60)
    print("SUPER-CLASS K-NN TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
