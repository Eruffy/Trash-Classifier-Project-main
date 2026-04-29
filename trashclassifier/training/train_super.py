"""
Super-Class Training Module
Trains both SVM and k-NN models for the first stage (super-class) classification
Super-classes: 0=Fiber, 1=Rigid, 2=Transparent, 3=Garbage
"""
import os
import sys
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from config import Config
from features.feature_extractor import extract_features


# Super-class mapping
SUPER_CLASS_MAP = {
    'paper': 0,      # Fiber
    'cardboard': 0,  # Fiber
    'plastic': 1,    # Rigid
    'metal': 1,      # Rigid
    'glass': 2,      # Transparent
    'trash': 3       # Garbage
}

SUPER_CLASS_NAMES = ['Fiber', 'Rigid', 'Transparent', 'Garbage']


def load_super_class_data():
    """Load and prepare data for super-class training"""
    print("[INFO] Loading data for super-class training...")
    
    X = []
    y = []
    
    for class_name in Config.CLASS_NAMES.keys():
        class_path = os.path.join(Config.DATASET_PATH, class_name)
        
        if not os.path.exists(class_path):
            print(f"[WARNING] Skipping missing class: {class_name}")
            continue
        
        super_class = SUPER_CLASS_MAP.get(class_name)
        if super_class is None:
            continue
        
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"  Loading {class_name} → {SUPER_CLASS_NAMES[super_class]}: {len(images)} images")
        
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            # Resize and extract features
            img_resized = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            features = extract_features(img_resized)
            
            X.append(features)
            y.append(super_class)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n[INFO] Loaded {len(X)} samples")
    print(f"[INFO] Feature dimension: {X.shape[1]}")
    
    # Class distribution
    class_counts = Counter(y)
    for sc_id, count in sorted(class_counts.items()):
        print(f"  {SUPER_CLASS_NAMES[sc_id]}: {count} samples")
    
    return X, y


def calculate_class_weights(y):
    """Calculate class weights for imbalanced data"""
    class_counts = Counter(y)
    total = len(y)
    weights = {}
    
    for class_id, count in class_counts.items():
        # Inverse frequency weighting
        weights[class_id] = total / (len(class_counts) * count)
    
    # Boost transparent class (glass) slightly more
    if 2 in weights:
        weights[2] *= 1.3
    
    return weights


def train_super_models(svm_params=None, knn_params=None):
    """Train both SVM and k-NN super-class models"""
    
    # Load data
    X, y = load_super_class_data()
    
    # Calculate class weights
    class_weights = calculate_class_weights(y)
    print(f"\n[INFO] Class weights: {class_weights}")
    
    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
    )
    
    print(f"\n[INFO] Training set: {len(X_train)} samples")
    print(f"[INFO] Validation set: {len(X_val)} samples")
    
    # ==================== TRAIN SVM ====================
    print("\n" + "="*70)
    print("TRAINING SUPER-CLASS SVM".center(70))
    print("="*70)
    
    # Scale features
    svm_scaler = StandardScaler()
    X_train_scaled = svm_scaler.fit_transform(X_train)
    X_val_scaled = svm_scaler.transform(X_val)
    
    # SVM parameters
    if svm_params is None:
        svm_params = {
            'kernel': 'rbf',
            'C': 10,
            'gamma': 'scale',
            'probability': True,
            'class_weight': class_weights,
            'random_state': Config.RANDOM_STATE
        }
    else:
        svm_params['probability'] = True
        svm_params['class_weight'] = class_weights
        svm_params['random_state'] = Config.RANDOM_STATE
    
    print(f"[INFO] SVM parameters: {svm_params}")
    
    # Train SVM
    svm_model = SVC(**svm_params)
    print("[INFO] Training SVM...")
    svm_model.fit(X_train_scaled, y_train)
    
    # Evaluate SVM
    y_pred_svm = svm_model.predict(X_val_scaled)
    svm_accuracy = accuracy_score(y_val, y_pred_svm)
    
    print(f"\n[RESULT] SVM Validation Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_svm, target_names=SUPER_CLASS_NAMES, zero_division=0))
    
    # Save SVM models
    svm_path = os.path.join(Config.SAVED_MODELS_PATH, 'super_svm.pkl')
    scaler_path = os.path.join(Config.SAVED_MODELS_PATH, 'super_svm_scaler.pkl')
    
    joblib.dump(svm_model, svm_path)
    joblib.dump(svm_scaler, scaler_path)
    print(f"\n[INFO] SVM model saved to: {svm_path}")
    print(f"[INFO] SVM scaler saved to: {scaler_path}")
    
    # ==================== TRAIN K-NN ====================
    print("\n" + "="*70)
    print("TRAINING SUPER-CLASS K-NN".center(70))
    print("="*70)
    
    # Scale features for k-NN
    knn_scaler = StandardScaler()
    X_train_scaled_knn = knn_scaler.fit_transform(X_train)
    X_val_scaled_knn = knn_scaler.transform(X_val)
    
    # k-NN parameters
    if knn_params is None:
        knn_params = {
            'n_neighbors': 7,
            'weights': 'distance',
            'metric': 'euclidean',
            'n_jobs': -1
        }
    
    print(f"[INFO] k-NN parameters: {knn_params}")
    
    # Train k-NN
    knn_model = KNeighborsClassifier(**knn_params)
    print("[INFO] Training k-NN...")
    knn_model.fit(X_train_scaled_knn, y_train)
    
    # Evaluate k-NN
    y_pred_knn = knn_model.predict(X_val_scaled_knn)
    knn_accuracy = accuracy_score(y_val, y_pred_knn)
    
    print(f"\n[RESULT] k-NN Validation Accuracy: {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_knn, target_names=SUPER_CLASS_NAMES, zero_division=0))
    
    # Save k-NN models
    knn_path = os.path.join(Config.SAVED_MODELS_PATH, 'super_knn.pkl')
    knn_scaler_path = os.path.join(Config.SAVED_MODELS_PATH, 'super_knn_scaler.pkl')
    
    joblib.dump(knn_model, knn_path)
    joblib.dump(knn_scaler, knn_scaler_path)
    print(f"\n[INFO] k-NN model saved to: {knn_path}")
    print(f"[INFO] k-NN scaler saved to: {knn_scaler_path}")
    
    # Summary
    print("\n" + "="*70)
    print("SUPER-CLASS TRAINING COMPLETE".center(70))
    print("="*70)
    print(f"SVM Accuracy:  {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
    print(f"k-NN Accuracy: {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")
    print("="*70 + "\n")
    
    return {
        'svm_accuracy': svm_accuracy,
        'knn_accuracy': knn_accuracy
    }


if __name__ == "__main__":
    train_super_models()
