"""
Train new models on augmented dataset
This script trains SVM and KNN models (both super and fine) on the augmented dataset
and saves them with '_aug' suffix to distinguish from original models
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from config import Config
from training.train_super import train_super_models, load_super_class_data, SUPER_CLASS_NAMES, SUPER_CLASS_MAP
from training.train_fine import train_fine_models, train_fiber_models, train_rigid_models
import joblib
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter


# Path to augmented dataset
AUGMENTED_DATASET_PATH = r"C:\Users\ahmed\Desktop\FINAL_ML\augmented_data"


def load_super_class_data_augmented():
    """Load and prepare augmented data for super-class training"""
    print("[INFO] Loading augmented data for super-class training...")
    
    X = []
    y = []
    
    for class_name in Config.CLASS_NAMES.keys():
        class_path = os.path.join(AUGMENTED_DATASET_PATH, class_name)
        
        if not os.path.exists(class_path):
            print(f"[WARNING] Skipping missing class: {class_name}")
            continue
        
        super_class = SUPER_CLASS_MAP.get(class_name)
        if super_class is None:
            continue
        
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"  Loading {class_name} → {SUPER_CLASS_NAMES[super_class]}: {len(images)} images")
        
        from features.feature_extractor import extract_features
        
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


def train_super_models_augmented():
    """Train both SVM and k-NN super-class models on augmented data"""
    
    print("\n" + "="*70)
    print("TRAINING ON AUGMENTED DATASET - SUPER-CLASS MODELS".center(70))
    print("="*70 + "\n")
    
    # Load augmented data
    X, y = load_super_class_data_augmented()
    
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
    print("TRAINING SUPER-CLASS SVM (AUGMENTED)".center(70))
    print("="*70)
    
    # Scale features
    svm_scaler = StandardScaler()
    X_train_scaled = svm_scaler.fit_transform(X_train)
    X_val_scaled = svm_scaler.transform(X_val)
    
    # SVM parameters (same as original)
    svm_params = {
        'kernel': 'rbf',
        'C': 10,
        'gamma': 'scale',
        'probability': True,
        'class_weight': class_weights,
        'random_state': Config.RANDOM_STATE
    }
    
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
    
    # Save SVM models with '_aug' suffix
    svm_path = os.path.join(Config.SAVED_MODELS_PATH, 'super_svm_aug.pkl')
    scaler_path = os.path.join(Config.SAVED_MODELS_PATH, 'super_svm_scaler_aug.pkl')
    
    joblib.dump(svm_model, svm_path)
    joblib.dump(svm_scaler, scaler_path)
    print(f"\n[INFO] SVM model saved to: {svm_path}")
    print(f"[INFO] SVM scaler saved to: {scaler_path}")
    
    # ==================== TRAIN K-NN ====================
    print("\n" + "="*70)
    print("TRAINING SUPER-CLASS K-NN (AUGMENTED)".center(70))
    print("="*70)
    
    # Scale features for k-NN
    knn_scaler = StandardScaler()
    X_train_scaled_knn = knn_scaler.fit_transform(X_train)
    X_val_scaled_knn = knn_scaler.transform(X_val)
    
    # k-NN parameters (same as original)
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
    
    # Save k-NN models with '_aug' suffix
    knn_path = os.path.join(Config.SAVED_MODELS_PATH, 'super_knn_aug.pkl')
    knn_scaler_path = os.path.join(Config.SAVED_MODELS_PATH, 'super_knn_scaler_aug.pkl')
    
    joblib.dump(knn_model, knn_path)
    joblib.dump(knn_scaler, knn_scaler_path)
    print(f"\n[INFO] k-NN model saved to: {knn_path}")
    print(f"[INFO] k-NN scaler saved to: {knn_scaler_path}")
    
    # Summary
    print("\n" + "="*70)
    print("SUPER-CLASS TRAINING COMPLETE (AUGMENTED)".center(70))
    print("="*70)
    print(f"SVM Accuracy:  {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
    print(f"k-NN Accuracy: {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")
    print("="*70 + "\n")
    
    return {
        'svm_accuracy': svm_accuracy,
        'knn_accuracy': knn_accuracy
    }


def load_fine_class_data_augmented(super_class_name, class_names):
    """
    Load augmented data for fine-class training
    
    Args:
        super_class_name: 'fiber' or 'rigid'
        class_names: list of class names (e.g., ['paper', 'cardboard'])
    """
    print(f"[INFO] Loading augmented {super_class_name} fine-class data...")
    
    X = []
    y = []
    
    from features.feature_extractor import extract_features
    
    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(AUGMENTED_DATASET_PATH, class_name)
        
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


def train_fiber_models_augmented():
    """Train Fiber fine-class models on augmented data (Paper vs Cardboard)"""
    
    print("\n" + "="*70)
    print("TRAINING FIBER MODELS (AUGMENTED) - Paper vs Cardboard".center(70))
    print("="*70 + "\n")
    
    class_names = ['paper', 'cardboard']
    X, y = load_fine_class_data_augmented('fiber', class_names)
    
    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
    )
    
    print(f"\n[INFO] Training set: {len(X_train)} samples")
    print(f"[INFO] Validation set: {len(X_val)} samples")
    
    # ==================== TRAIN FIBER SVM ====================
    print("\n--- Training Fiber SVM (Augmented) ---")
    
    svm_scaler = StandardScaler()
    X_train_scaled = svm_scaler.fit_transform(X_train)
    X_val_scaled = svm_scaler.transform(X_val)
    
    svm_params = {
        'kernel': 'rbf',
        'C': 10,
        'gamma': 'scale',
        'probability': True,
        'random_state': Config.RANDOM_STATE
    }
    
    print(f"[INFO] Parameters: {svm_params}")
    
    svm_model = SVC(**svm_params)
    svm_model.fit(X_train_scaled, y_train)
    
    y_pred_svm = svm_model.predict(X_val_scaled)
    svm_accuracy = accuracy_score(y_val, y_pred_svm)
    
    print(f"\n[RESULT] Fiber SVM Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
    print(classification_report(y_val, y_pred_svm, target_names=class_names, zero_division=0))
    
    # Save with '_aug' suffix
    svm_path = os.path.join(Config.SAVED_MODELS_PATH, 'fiber_svm_aug.pkl')
    scaler_path = os.path.join(Config.SAVED_MODELS_PATH, 'fiber_svm_scaler_aug.pkl')
    joblib.dump(svm_model, svm_path)
    joblib.dump(svm_scaler, scaler_path)
    print(f"[INFO] Saved to {svm_path}")
    
    # ==================== TRAIN FIBER K-NN ====================
    print("\n--- Training Fiber k-NN (Augmented) ---")
    
    knn_scaler = StandardScaler()
    X_train_scaled_knn = knn_scaler.fit_transform(X_train)
    X_val_scaled_knn = knn_scaler.transform(X_val)
    
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
    
    # Save with '_aug' suffix
    knn_path = os.path.join(Config.SAVED_MODELS_PATH, 'fiber_knn_aug.pkl')
    knn_scaler_path = os.path.join(Config.SAVED_MODELS_PATH, 'fiber_knn_scaler_aug.pkl')
    joblib.dump(knn_model, knn_path)
    joblib.dump(knn_scaler, knn_scaler_path)
    print(f"[INFO] Saved to {knn_path}")
    
    return {'svm_accuracy': svm_accuracy, 'knn_accuracy': knn_accuracy}


def train_rigid_models_augmented():
    """Train Rigid fine-class models on augmented data (Plastic vs Metal)"""
    
    print("\n" + "="*70)
    print("TRAINING RIGID MODELS (AUGMENTED) - Plastic vs Metal".center(70))
    print("="*70 + "\n")
    
    class_names = ['plastic', 'metal']
    X, y = load_fine_class_data_augmented('rigid', class_names)
    
    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
    )
    
    print(f"\n[INFO] Training set: {len(X_train)} samples")
    print(f"[INFO] Validation set: {len(X_val)} samples")
    
    # ==================== TRAIN RIGID SVM ====================
    print("\n--- Training Rigid SVM (Augmented) ---")
    
    svm_scaler = StandardScaler()
    X_train_scaled = svm_scaler.fit_transform(X_train)
    X_val_scaled = svm_scaler.transform(X_val)
    
    svm_params = {
        'kernel': 'rbf',
        'C': 10,
        'gamma': 'scale',
        'probability': True,
        'random_state': Config.RANDOM_STATE
    }
    
    print(f"[INFO] Parameters: {svm_params}")
    
    svm_model = SVC(**svm_params)
    svm_model.fit(X_train_scaled, y_train)
    
    y_pred_svm = svm_model.predict(X_val_scaled)
    svm_accuracy = accuracy_score(y_val, y_pred_svm)
    
    print(f"\n[RESULT] Rigid SVM Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
    print(classification_report(y_val, y_pred_svm, target_names=class_names, zero_division=0))
    
    # Save with '_aug' suffix
    svm_path = os.path.join(Config.SAVED_MODELS_PATH, 'rigid_svm_aug.pkl')
    scaler_path = os.path.join(Config.SAVED_MODELS_PATH, 'rigid_svm_scaler_aug.pkl')
    joblib.dump(svm_model, svm_path)
    joblib.dump(svm_scaler, scaler_path)
    print(f"[INFO] Saved to {svm_path}")
    
    # ==================== TRAIN RIGID K-NN ====================
    print("\n--- Training Rigid k-NN (Augmented) ---")
    
    knn_scaler = StandardScaler()
    X_train_scaled_knn = knn_scaler.fit_transform(X_train)
    X_val_scaled_knn = knn_scaler.transform(X_val)
    
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
    
    # Save with '_aug' suffix
    knn_path = os.path.join(Config.SAVED_MODELS_PATH, 'rigid_knn_aug.pkl')
    knn_scaler_path = os.path.join(Config.SAVED_MODELS_PATH, 'rigid_knn_scaler_aug.pkl')
    joblib.dump(knn_model, knn_path)
    joblib.dump(knn_scaler, knn_scaler_path)
    print(f"[INFO] Saved to {knn_path}")
    
    return {'svm_accuracy': svm_accuracy, 'knn_accuracy': knn_accuracy}


def train_all_augmented():
    """Train all models on augmented dataset"""
    
    print("\n" + "="*70)
    print("TRAINING ALL MODELS ON AUGMENTED DATASET".center(70))
    print("="*70 + "\n")
    
    # Train Super-class models
    super_results = train_super_models_augmented()
    
    # Train Fine-class models
    fiber_results = train_fiber_models_augmented()
    rigid_results = train_rigid_models_augmented()
    
    # Final Summary
    print("\n" + "="*70)
    print("ALL MODELS TRAINED ON AUGMENTED DATASET".center(70))
    print("="*70)
    print("\nSuper-Class Models:")
    print(f"  SVM:  {super_results['svm_accuracy']:.4f} ({super_results['svm_accuracy']*100:.2f}%)")
    print(f"  k-NN: {super_results['knn_accuracy']:.4f} ({super_results['knn_accuracy']*100:.2f}%)")
    print("\nFine-Class Models (Fiber):")
    print(f"  SVM:  {fiber_results['svm_accuracy']:.4f} ({fiber_results['svm_accuracy']*100:.2f}%)")
    print(f"  k-NN: {fiber_results['knn_accuracy']:.4f} ({fiber_results['knn_accuracy']*100:.2f}%)")
    print("\nFine-Class Models (Rigid):")
    print(f"  SVM:  {rigid_results['svm_accuracy']:.4f} ({rigid_results['svm_accuracy']*100:.2f}%)")
    print(f"  k-NN: {rigid_results['knn_accuracy']:.4f} ({rigid_results['knn_accuracy']*100:.2f}%)")
    print("="*70 + "\n")
    
    print("[SUCCESS] All augmented models saved with '_aug' suffix in saved_models directory")
    print("[INFO] You can now run comparison tests between original and augmented models")
    
    return {
        'super': super_results,
        'fiber': fiber_results,
        'rigid': rigid_results
    }


if __name__ == "__main__":
    results = train_all_augmented()
