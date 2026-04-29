"""
Compare original vs augmented models
Tests both sets of models on the same test data and provides detailed comparison
"""
import os
import sys
import numpy as np
import cv2
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter

# Add parent directory to path
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


def load_test_data(dataset_path):
    """Load test data from either original or augmented dataset"""
    print(f"[INFO] Loading test data from: {dataset_path}")
    
    X = []
    y_super = []
    y_fine = []
    
    for class_name in Config.CLASS_NAMES.keys():
        class_path = os.path.join(dataset_path, class_name)
        
        if not os.path.exists(class_path):
            print(f"[WARNING] Skipping missing class: {class_name}")
            continue
        
        super_class = SUPER_CLASS_MAP.get(class_name)
        fine_class = Config.CLASS_NAMES[class_name]
        
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
            y_super.append(super_class)
            y_fine.append(fine_class)
    
    X = np.array(X)
    y_super = np.array(y_super)
    y_fine = np.array(y_fine)
    
    print(f"\n[INFO] Loaded {len(X)} samples")
    return X, y_super, y_fine


def evaluate_super_models(X_test, y_test, model_suffix=''):
    """Evaluate super-class models"""
    
    suffix_display = f" ({model_suffix.upper()})" if model_suffix else " (ORIGINAL)"
    
    print("\n" + "="*70)
    print(f"EVALUATING SUPER-CLASS MODELS{suffix_display}".center(70))
    print("="*70 + "\n")
    
    results = {}
    
    # Load and test SVM
    svm_path = os.path.join(Config.SAVED_MODELS_PATH, f'super_svm{model_suffix}.pkl')
    scaler_path = os.path.join(Config.SAVED_MODELS_PATH, f'super_svm_scaler{model_suffix}.pkl')
    
    if os.path.exists(svm_path) and os.path.exists(scaler_path):
        print("--- Testing SVM ---")
        svm_model = joblib.load(svm_path)
        svm_scaler = joblib.load(scaler_path)
        
        X_test_scaled = svm_scaler.transform(X_test)
        y_pred = svm_model.predict(X_test_scaled)
        
        svm_accuracy = accuracy_score(y_test, y_pred)
        results['svm_accuracy'] = svm_accuracy
        
        print(f"SVM Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=SUPER_CLASS_NAMES, zero_division=0))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print("Predicted →")
        print("Actual ↓")
        print("         ", "  ".join([f"{name[:6]:>6}" for name in SUPER_CLASS_NAMES]))
        for i, name in enumerate(SUPER_CLASS_NAMES):
            print(f"{name[:8]:<8}", "  ".join([f"{cm[i,j]:>6}" for j in range(len(SUPER_CLASS_NAMES))]))
    else:
        print(f"[WARNING] SVM model not found: {svm_path}")
        results['svm_accuracy'] = 0.0
    
    # Load and test KNN
    knn_path = os.path.join(Config.SAVED_MODELS_PATH, f'super_knn{model_suffix}.pkl')
    knn_scaler_path = os.path.join(Config.SAVED_MODELS_PATH, f'super_knn_scaler{model_suffix}.pkl')
    
    if os.path.exists(knn_path) and os.path.exists(knn_scaler_path):
        print("\n--- Testing k-NN ---")
        knn_model = joblib.load(knn_path)
        knn_scaler = joblib.load(knn_scaler_path)
        
        X_test_scaled = knn_scaler.transform(X_test)
        y_pred = knn_model.predict(X_test_scaled)
        
        knn_accuracy = accuracy_score(y_test, y_pred)
        results['knn_accuracy'] = knn_accuracy
        
        print(f"k-NN Accuracy: {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=SUPER_CLASS_NAMES, zero_division=0))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print("Predicted →")
        print("Actual ↓")
        print("         ", "  ".join([f"{name[:6]:>6}" for name in SUPER_CLASS_NAMES]))
        for i, name in enumerate(SUPER_CLASS_NAMES):
            print(f"{name[:8]:<8}", "  ".join([f"{cm[i,j]:>6}" for j in range(len(SUPER_CLASS_NAMES))]))
    else:
        print(f"[WARNING] k-NN model not found: {knn_path}")
        results['knn_accuracy'] = 0.0
    
    return results


def evaluate_fiber_models(X_test, y_test, model_suffix=''):
    """Evaluate fiber fine-class models"""
    
    suffix_display = f" ({model_suffix.upper()})" if model_suffix else " (ORIGINAL)"
    class_names = ['paper', 'cardboard']
    
    print("\n" + "="*70)
    print(f"EVALUATING FIBER MODELS{suffix_display}".center(70))
    print("="*70 + "\n")
    
    results = {}
    
    # Load and test SVM
    svm_path = os.path.join(Config.SAVED_MODELS_PATH, f'fiber_svm{model_suffix}.pkl')
    scaler_path = os.path.join(Config.SAVED_MODELS_PATH, f'fiber_svm_scaler{model_suffix}.pkl')
    
    if os.path.exists(svm_path) and os.path.exists(scaler_path):
        print("--- Testing Fiber SVM ---")
        svm_model = joblib.load(svm_path)
        svm_scaler = joblib.load(scaler_path)
        
        X_test_scaled = svm_scaler.transform(X_test)
        y_pred = svm_model.predict(X_test_scaled)
        
        svm_accuracy = accuracy_score(y_test, y_pred)
        results['svm_accuracy'] = svm_accuracy
        
        print(f"Fiber SVM Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
        print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    else:
        print(f"[WARNING] Fiber SVM model not found")
        results['svm_accuracy'] = 0.0
    
    # Load and test KNN
    knn_path = os.path.join(Config.SAVED_MODELS_PATH, f'fiber_knn{model_suffix}.pkl')
    knn_scaler_path = os.path.join(Config.SAVED_MODELS_PATH, f'fiber_knn_scaler{model_suffix}.pkl')
    
    if os.path.exists(knn_path) and os.path.exists(knn_scaler_path):
        print("\n--- Testing Fiber k-NN ---")
        knn_model = joblib.load(knn_path)
        knn_scaler = joblib.load(knn_scaler_path)
        
        X_test_scaled = knn_scaler.transform(X_test)
        y_pred = knn_model.predict(X_test_scaled)
        
        knn_accuracy = accuracy_score(y_test, y_pred)
        results['knn_accuracy'] = knn_accuracy
        
        print(f"Fiber k-NN Accuracy: {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")
        print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    else:
        print(f"[WARNING] Fiber k-NN model not found")
        results['knn_accuracy'] = 0.0
    
    return results


def evaluate_rigid_models(X_test, y_test, model_suffix=''):
    """Evaluate rigid fine-class models"""
    
    suffix_display = f" ({model_suffix.upper()})" if model_suffix else " (ORIGINAL)"
    class_names = ['plastic', 'metal']
    
    print("\n" + "="*70)
    print(f"EVALUATING RIGID MODELS{suffix_display}".center(70))
    print("="*70 + "\n")
    
    results = {}
    
    # Load and test SVM
    svm_path = os.path.join(Config.SAVED_MODELS_PATH, f'rigid_svm{model_suffix}.pkl')
    scaler_path = os.path.join(Config.SAVED_MODELS_PATH, f'rigid_svm_scaler{model_suffix}.pkl')
    
    if os.path.exists(svm_path) and os.path.exists(scaler_path):
        print("--- Testing Rigid SVM ---")
        svm_model = joblib.load(svm_path)
        svm_scaler = joblib.load(scaler_path)
        
        X_test_scaled = svm_scaler.transform(X_test)
        y_pred = svm_model.predict(X_test_scaled)
        
        svm_accuracy = accuracy_score(y_test, y_pred)
        results['svm_accuracy'] = svm_accuracy
        
        print(f"Rigid SVM Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
        print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    else:
        print(f"[WARNING] Rigid SVM model not found")
        results['svm_accuracy'] = 0.0
    
    # Load and test KNN
    knn_path = os.path.join(Config.SAVED_MODELS_PATH, f'rigid_knn{model_suffix}.pkl')
    knn_scaler_path = os.path.join(Config.SAVED_MODELS_PATH, f'rigid_knn_scaler{model_suffix}.pkl')
    
    if os.path.exists(knn_path) and os.path.exists(knn_scaler_path):
        print("\n--- Testing Rigid k-NN ---")
        knn_model = joblib.load(knn_path)
        knn_scaler = joblib.load(knn_scaler_path)
        
        X_test_scaled = knn_scaler.transform(X_test)
        y_pred = knn_model.predict(X_test_scaled)
        
        knn_accuracy = accuracy_score(y_test, y_pred)
        results['knn_accuracy'] = knn_accuracy
        
        print(f"Rigid k-NN Accuracy: {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")
        print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    else:
        print(f"[WARNING] Rigid k-NN model not found")
        results['knn_accuracy'] = 0.0
    
    return results


def compare_all_models(test_dataset_path=None):
    """
    Compare original vs augmented models
    
    Args:
        test_dataset_path: Path to test dataset. If None, uses augmented dataset
    """
    
    if test_dataset_path is None:
        test_dataset_path = r"C:\Users\ahmed\Desktop\FINAL_ML\augmented_data"
    
    print("\n" + "="*70)
    print("MODEL COMPARISON: ORIGINAL vs AUGMENTED".center(70))
    print("="*70 + "\n")
    
    # Load test data
    X_full, y_super_full, y_fine_full = load_test_data(test_dataset_path)
    
    # Split into test set (20% of data)
    _, X_test, _, y_super_test, _, y_fine_test = train_test_split(
        X_full, y_super_full, y_fine_full,
        test_size=0.2,
        random_state=Config.RANDOM_STATE,
        stratify=y_super_full
    )
    
    print(f"\n[INFO] Test set size: {len(X_test)} samples\n")
    
    # Evaluate original models
    print("\n" + "#"*70)
    print("# TESTING ORIGINAL MODELS".center(70))
    print("#"*70)
    
    orig_super = evaluate_super_models(X_test, y_super_test, model_suffix='')
    
    # Prepare fiber test data (paper + cardboard)
    fiber_mask = (y_super_test == 0)
    X_fiber = X_test[fiber_mask]
    y_fiber = []
    for i, is_fiber in enumerate(fiber_mask):
        if is_fiber:
            # Map fine class to 0 (paper) or 1 (cardboard)
            fine_class = y_fine_test[i]
            if fine_class == 1:  # paper
                y_fiber.append(0)
            elif fine_class == 2:  # cardboard
                y_fiber.append(1)
    y_fiber = np.array(y_fiber)
    
    orig_fiber = evaluate_fiber_models(X_fiber, y_fiber, model_suffix='') if len(X_fiber) > 0 else {}
    
    # Prepare rigid test data (plastic + metal)
    rigid_mask = (y_super_test == 1)
    X_rigid = X_test[rigid_mask]
    y_rigid = []
    for i, is_rigid in enumerate(rigid_mask):
        if is_rigid:
            # Map fine class to 0 (plastic) or 1 (metal)
            fine_class = y_fine_test[i]
            if fine_class == 3:  # plastic
                y_rigid.append(0)
            elif fine_class == 4:  # metal
                y_rigid.append(1)
    y_rigid = np.array(y_rigid)
    
    orig_rigid = evaluate_rigid_models(X_rigid, y_rigid, model_suffix='') if len(X_rigid) > 0 else {}
    
    # Evaluate augmented models
    print("\n" + "#"*70)
    print("# TESTING AUGMENTED MODELS".center(70))
    print("#"*70)
    
    aug_super = evaluate_super_models(X_test, y_super_test, model_suffix='_aug')
    aug_fiber = evaluate_fiber_models(X_fiber, y_fiber, model_suffix='_aug') if len(X_fiber) > 0 else {}
    aug_rigid = evaluate_rigid_models(X_rigid, y_rigid, model_suffix='_aug') if len(X_rigid) > 0 else {}
    
    # Final Comparison Summary
    print("\n" + "="*70)
    print("FINAL COMPARISON SUMMARY".center(70))
    print("="*70 + "\n")
    
    print(f"{'Model':<30} {'Original':<15} {'Augmented':<15} {'Improvement':<15}")
    print("-" * 70)
    
    # Super-class models
    print("\nSuper-Class Models:")
    orig_svm = orig_super.get('svm_accuracy', 0) * 100
    aug_svm = aug_super.get('svm_accuracy', 0) * 100
    print(f"  {'SVM':<28} {orig_svm:>6.2f}%         {aug_svm:>6.2f}%         {aug_svm - orig_svm:>+6.2f}%")
    
    orig_knn = orig_super.get('knn_accuracy', 0) * 100
    aug_knn = aug_super.get('knn_accuracy', 0) * 100
    print(f"  {'k-NN':<28} {orig_knn:>6.2f}%         {aug_knn:>6.2f}%         {aug_knn - orig_knn:>+6.2f}%")
    
    # Fiber models
    if orig_fiber and aug_fiber:
        print("\nFiber Models (Paper vs Cardboard):")
        orig_svm = orig_fiber.get('svm_accuracy', 0) * 100
        aug_svm = aug_fiber.get('svm_accuracy', 0) * 100
        print(f"  {'SVM':<28} {orig_svm:>6.2f}%         {aug_svm:>6.2f}%         {aug_svm - orig_svm:>+6.2f}%")
        
        orig_knn = orig_fiber.get('knn_accuracy', 0) * 100
        aug_knn = aug_fiber.get('knn_accuracy', 0) * 100
        print(f"  {'k-NN':<28} {orig_knn:>6.2f}%         {aug_knn:>6.2f}%         {aug_knn - orig_knn:>+6.2f}%")
    
    # Rigid models
    if orig_rigid and aug_rigid:
        print("\nRigid Models (Plastic vs Metal):")
        orig_svm = orig_rigid.get('svm_accuracy', 0) * 100
        aug_svm = aug_rigid.get('svm_accuracy', 0) * 100
        print(f"  {'SVM':<28} {orig_svm:>6.2f}%         {aug_svm:>6.2f}%         {aug_svm - orig_svm:>+6.2f}%")
        
        orig_knn = orig_rigid.get('knn_accuracy', 0) * 100
        aug_knn = aug_rigid.get('knn_accuracy', 0) * 100
        print(f"  {'k-NN':<28} {orig_knn:>6.2f}%         {aug_knn:>6.2f}%         {aug_knn - orig_knn:>+6.2f}%")
    
    print("\n" + "="*70)
    print("[INFO] Comparison complete!")
    print("="*70 + "\n")
    
    return {
        'original': {
            'super': orig_super,
            'fiber': orig_fiber,
            'rigid': orig_rigid
        },
        'augmented': {
            'super': aug_super,
            'fiber': aug_fiber,
            'rigid': aug_rigid
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare original vs augmented models')
    parser.add_argument('--test-data', type=str, default=None,
                       help='Path to test dataset (default: augmented dataset)')
    
    args = parser.parse_args()
    
    results = compare_all_models(args.test_data)
