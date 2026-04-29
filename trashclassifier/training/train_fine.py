"""
Fine-Class Training Module
Trains SVM and k-NN models for second stage (fine-class) classification
Fiber: Paper vs Cardboard
Rigid: Plastic vs Metal
"""
import os
import sys
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from config import Config
from features.feature_extractor import extract_features


def load_fine_class_data(super_class_name, class_names):
    """
    Load data for fine-class training
    
    Args:
        super_class_name: 'fiber' or 'rigid'
        class_names: list of class names (e.g., ['paper', 'cardboard'])
    """
    print(f"[INFO] Loading {super_class_name} fine-class data...")
    
    X = []
    y = []
    
    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(Config.DATASET_PATH, class_name)
        
        if not os.path.exists(class_path):
            print(f"[WARNING] Skipping missing class: {class_name}")
            continue
        
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"  Loading {class_name}: {len(images)} images")
        
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            # Resize and extract features
            img_resized = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            features = extract_features(img_resized)
            
            X.append(features)
            y.append(idx)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n[INFO] Loaded {len(X)} samples")
    print(f"[INFO] Feature dimension: {X.shape[1]}")
    
    # Class distribution
    class_counts = Counter(y)
    for class_id, count in sorted(class_counts.items()):
        print(f"  {class_names[class_id]}: {count} samples")
    
    return X, y


def train_fiber_models(svm_params=None, knn_params=None):
    """Train Fiber fine-class models (Paper vs Cardboard)"""
    
    print("\n" + "="*70)
    print("TRAINING FIBER FINE-CLASS MODELS (Paper vs Cardboard)".center(70))
    print("="*70 + "\n")
    
    class_names = ['paper', 'cardboard']
    X, y = load_fine_class_data('fiber', class_names)
    
    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
    )
    
    print(f"\n[INFO] Training set: {len(X_train)} samples")
    print(f"[INFO] Validation set: {len(X_val)} samples")
    
    # ==================== TRAIN FIBER SVM ====================
    print("\n--- Training Fiber SVM ---")
    
    svm_scaler = StandardScaler()
    X_train_scaled = svm_scaler.fit_transform(X_train)
    X_val_scaled = svm_scaler.transform(X_val)
    
    if svm_params is None:
        svm_params = {
            'kernel': 'rbf',
            'C': 10,
            'gamma': 'scale',
            'probability': True,
            'random_state': Config.RANDOM_STATE
        }
    else:
        svm_params['probability'] = True
        svm_params['random_state'] = Config.RANDOM_STATE
    
    print(f"[INFO] Parameters: {svm_params}")
    
    svm_model = SVC(**svm_params)
    svm_model.fit(X_train_scaled, y_train)
    
    y_pred_svm = svm_model.predict(X_val_scaled)
    svm_accuracy = accuracy_score(y_val, y_pred_svm)
    
    print(f"\n[RESULT] Fiber SVM Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
    print(classification_report(y_val, y_pred_svm, target_names=class_names, zero_division=0))
    
    # Save
    svm_path = os.path.join(Config.SAVED_MODELS_PATH, 'fiber_svm.pkl')
    scaler_path = os.path.join(Config.SAVED_MODELS_PATH, 'fiber_svm_scaler.pkl')
    joblib.dump(svm_model, svm_path)
    joblib.dump(svm_scaler, scaler_path)
    print(f"[INFO] Saved to {svm_path}")
    
    # ==================== TRAIN FIBER K-NN ====================
    print("\n--- Training Fiber k-NN ---")
    
    knn_scaler = StandardScaler()
    X_train_scaled_knn = knn_scaler.fit_transform(X_train)
    X_val_scaled_knn = knn_scaler.transform(X_val)
    
    if knn_params is None:
        knn_params = {
            'n_neighbors': 7,
            'weights': 'distance',
            'metric': 'euclidean',
            'n_jobs': -1
        }
    
    print(f"[INFO] Parameters: {knn_params}")
    
    knn_model = KNeighborsClassifier(**knn_params)
    knn_model.fit(X_train_scaled_knn, y_train)
    
    y_pred_knn = knn_model.predict(X_val_scaled_knn)
    knn_accuracy = accuracy_score(y_val, y_pred_knn)
    
    print(f"\n[RESULT] Fiber k-NN Accuracy: {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")
    print(classification_report(y_val, y_pred_knn, target_names=class_names, zero_division=0))
    
    # Save
    knn_path = os.path.join(Config.SAVED_MODELS_PATH, 'fiber_knn.pkl')
    knn_scaler_path = os.path.join(Config.SAVED_MODELS_PATH, 'fiber_knn_scaler.pkl')
    joblib.dump(knn_model, knn_path)
    joblib.dump(knn_scaler, knn_scaler_path)
    print(f"[INFO] Saved to {knn_path}")
    
    return {'svm_accuracy': svm_accuracy, 'knn_accuracy': knn_accuracy}


def train_rigid_models(svm_params=None, knn_params=None):
    """Train Rigid fine-class models (Plastic vs Metal)"""
    
    print("\n" + "="*70)
    print("TRAINING RIGID FINE-CLASS MODELS (Plastic vs Metal)".center(70))
    print("="*70 + "\n")
    
    class_names = ['plastic', 'metal']
    X, y = load_fine_class_data('rigid', class_names)
    
    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
    )
    
    print(f"\n[INFO] Training set: {len(X_train)} samples")
    print(f"[INFO] Validation set: {len(X_val)} samples")
    
    # ==================== TRAIN RIGID SVM ====================
    print("\n--- Training Rigid SVM ---")
    
    svm_scaler = StandardScaler()
    X_train_scaled = svm_scaler.fit_transform(X_train)
    X_val_scaled = svm_scaler.transform(X_val)
    
    if svm_params is None:
        svm_params = {
            'kernel': 'rbf',
            'C': 10,
            'gamma': 'scale',
            'probability': True,
            'random_state': Config.RANDOM_STATE
        }
    else:
        svm_params['probability'] = True
        svm_params['random_state'] = Config.RANDOM_STATE
    
    print(f"[INFO] Parameters: {svm_params}")
    
    svm_model = SVC(**svm_params)
    svm_model.fit(X_train_scaled, y_train)
    
    y_pred_svm = svm_model.predict(X_val_scaled)
    svm_accuracy = accuracy_score(y_val, y_pred_svm)
    
    print(f"\n[RESULT] Rigid SVM Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
    print(classification_report(y_val, y_pred_svm, target_names=class_names, zero_division=0))
    
    # Save
    svm_path = os.path.join(Config.SAVED_MODELS_PATH, 'rigid_svm.pkl')
    scaler_path = os.path.join(Config.SAVED_MODELS_PATH, 'rigid_svm_scaler.pkl')
    joblib.dump(svm_model, svm_path)
    joblib.dump(svm_scaler, scaler_path)
    print(f"[INFO] Saved to {svm_path}")
    
    # ==================== TRAIN RIGID K-NN ====================
    print("\n--- Training Rigid k-NN ---")
    
    knn_scaler = StandardScaler()
    X_train_scaled_knn = knn_scaler.fit_transform(X_train)
    X_val_scaled_knn = knn_scaler.transform(X_val)
    
    if knn_params is None:
        knn_params = {
            'n_neighbors': 7,
            'weights': 'distance',
            'metric': 'euclidean',
            'n_jobs': -1
        }
    
    print(f"[INFO] Parameters: {knn_params}")
    
    knn_model = KNeighborsClassifier(**knn_params)
    knn_model.fit(X_train_scaled_knn, y_train)
    
    y_pred_knn = knn_model.predict(X_val_scaled_knn)
    knn_accuracy = accuracy_score(y_val, y_pred_knn)
    
    print(f"\n[RESULT] Rigid k-NN Accuracy: {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")
    print(classification_report(y_val, y_pred_knn, target_names=class_names, zero_division=0))
    
    # Save
    knn_path = os.path.join(Config.SAVED_MODELS_PATH, 'rigid_knn.pkl')
    knn_scaler_path = os.path.join(Config.SAVED_MODELS_PATH, 'rigid_knn_scaler.pkl')
    joblib.dump(knn_model, knn_path)
    joblib.dump(knn_scaler, knn_scaler_path)
    print(f"[INFO] Saved to {knn_path}")
    
    return {'svm_accuracy': svm_accuracy, 'knn_accuracy': knn_accuracy}


def train_fine_models(fiber_svm_params=None, fiber_knn_params=None, 
                      rigid_svm_params=None, rigid_knn_params=None):
    """Train all fine-class models"""
    
    # Train Fiber models
    fiber_results = train_fiber_models(fiber_svm_params, fiber_knn_params)
    
    # Train Rigid models
    rigid_results = train_rigid_models(rigid_svm_params, rigid_knn_params)
    
    # Summary
    print("\n" + "="*70)
    print("FINE-CLASS TRAINING COMPLETE".center(70))
    print("="*70)
    print(f"Fiber SVM:  {fiber_results['svm_accuracy']:.4f} ({fiber_results['svm_accuracy']*100:.2f}%)")
    print(f"Fiber k-NN: {fiber_results['knn_accuracy']:.4f} ({fiber_results['knn_accuracy']*100:.2f}%)")
    print(f"Rigid SVM:  {rigid_results['svm_accuracy']:.4f} ({rigid_results['svm_accuracy']*100:.2f}%)")
    print(f"Rigid k-NN: {rigid_results['knn_accuracy']:.4f} ({rigid_results['knn_accuracy']*100:.2f}%)")
    print("="*70 + "\n")
    
    return {
        'fiber': fiber_results,
        'rigid': rigid_results
    }


if __name__ == "__main__":
    train_fine_models()
