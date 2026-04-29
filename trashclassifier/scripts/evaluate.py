"""
Model Evaluation Script
Evaluates both SVM and k-NN models on validation data.
"""
import sys
import os
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.ensemble_predictor import EnsemblePredictor
from config import Config


def load_validation_data():
    """Load validation images from dataset"""
    print("[INFO] Loading validation data...")
    
    X = []
    y = []
    
    for class_name in Config.CLASS_NAMES.keys():
        class_id = Config.CLASS_NAMES[class_name]
        class_path = os.path.join(Config.DATASET_PATH, class_name)
        
        if not os.path.exists(class_path):
            print(f"[WARNING] Class folder not found: {class_path}")
            continue
        
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Use 20% for validation
        val_count = max(1, len(images) // 5)
        val_images = images[:val_count]
        
        print(f"  {class_name}: {len(val_images)} validation images")
        
        for img_name in val_images:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            
            if img is not None:
                X.append(img)
                y.append(class_id)
    
    print(f"[INFO] Loaded {len(X)} validation images")
    return X, np.array(y)


def evaluate_models():
    """Evaluate both SVM and k-NN models"""
    
    # Load validation data
    X_val, y_true = load_validation_data()
    
    if len(X_val) == 0:
        print("[ERROR] No validation data loaded!")
        return
    
    # Load predictor
    print("\n[INFO] Loading models...")
    predictor = EnsemblePredictor()
    
    # Get predictions
    print("[INFO] Making predictions...")
    y_pred_svm = []
    y_pred_knn = []
    y_pred_ensemble = []
    
    for i, img in enumerate(X_val):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(X_val)} images...")
        
        results = predictor.predict_all(img)
        y_pred_svm.append(results['svm']['class_id'])
        y_pred_knn.append(results['knn']['class_id'])
        y_pred_ensemble.append(results['ensemble']['class_id'])
    
    y_pred_svm = np.array(y_pred_svm)
    y_pred_knn = np.array(y_pred_knn)
    y_pred_ensemble = np.array(y_pred_ensemble)
    
    # Calculate metrics
    print("\n" + "="*70)
    print("EVALUATION RESULTS".center(70))
    print("="*70)
    
    # SVM Results
    print("\n📊 SVM MODEL")
    print("-" * 70)
    svm_acc = accuracy_score(y_true, y_pred_svm)
    print(f"Accuracy: {svm_acc:.4f} ({svm_acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_svm, 
                               labels=list(range(6)),
                               target_names=list(Config.CLASS_NAMES.keys()),
                               zero_division=0))
    
    # k-NN Results
    print("\n📊 K-NN MODEL")
    print("-" * 70)
    knn_acc = accuracy_score(y_true, y_pred_knn)
    print(f"Accuracy: {knn_acc:.4f} ({knn_acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_knn, 
                               labels=list(range(6)),
                               target_names=list(Config.CLASS_NAMES.keys()),
                               zero_division=0))
    
    # Ensemble Results
    print("\n📊 ENSEMBLE MODEL (FINAL)")
    print("-" * 70)
    ensemble_acc = accuracy_score(y_true, y_pred_ensemble)
    print(f"Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_ensemble, 
                               labels=list(range(6)),
                               target_names=list(Config.CLASS_NAMES.keys()),
                               zero_division=0))
    
    # Model agreement
    agreement = np.mean(y_pred_svm == y_pred_knn)
    print(f"\n🤝 Model Agreement: {agreement:.4f} ({agreement*100:.2f}%)")
    
    print("\n" + "="*70)
    
    # Plot confusion matrices
    plot_confusion_matrices(y_true, y_pred_svm, y_pred_knn, y_pred_ensemble)


def plot_confusion_matrices(y_true, y_pred_svm, y_pred_knn, y_pred_ensemble):
    """Plot confusion matrices for all models"""
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    class_labels = list(Config.CLASS_NAMES.keys())
    
    # SVM
    cm_svm = confusion_matrix(y_true, y_pred_svm)
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels, ax=axes[0])
    axes[0].set_title('SVM Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # k-NN
    cm_knn = confusion_matrix(y_true, y_pred_knn)
    sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_labels, yticklabels=class_labels, ax=axes[1])
    axes[1].set_title('k-NN Confusion Matrix', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    # Ensemble
    cm_ensemble = confusion_matrix(y_true, y_pred_ensemble)
    sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=class_labels, yticklabels=class_labels, ax=axes[2])
    axes[2].set_title('Ensemble Confusion Matrix', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('True Label')
    axes[2].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(Config.SAVED_MODELS_PATH, 'confusion_matrices.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[INFO] Confusion matrices saved to: {output_path}")
    
    plt.show()


def main():
    evaluate_models()


if __name__ == "__main__":
    main()
