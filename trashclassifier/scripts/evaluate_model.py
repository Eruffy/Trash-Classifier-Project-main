"""
Model Evaluation Script
Evaluates trained models on validation/test set
"""

import os
import sys
import numpy as np
import cv2
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from features.feature_extractor import extract_features
from inference.predictor import HierarchicalPredictor
from config import DATASET_PATH, CLASS_MAPPING


def load_test_data():
    """Load all images from dataset for testing"""
    X = []
    y = []
    image_paths = []
    
    for class_name, class_id in CLASS_MAPPING.items():
        if class_name == 'unknown':
            continue
            
        class_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(class_path):
            print(f"[WARNING] Class folder not found: {class_path}")
            continue
        
        print(f"[INFO] Loading {class_name} (ID: {class_id})...")
        
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        count = 0
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
                image_paths.append(img_path)
                count += 1
                
            except Exception as e:
                print(f"[WARNING] Error processing {img_path}: {e}")
                continue
        
        print(f"  Loaded {count} samples")
    
    return np.array(X), np.array(y), image_paths


def evaluate_model(model_type='svm'):
    """Evaluate a specific model"""
    print(f"\n{'='*60}")
    print(f"EVALUATING {model_type.upper()} MODEL")
    print(f"{'='*60}")
    
    # Load test data
    print("\n[STEP 1] Loading test data...")
    X_test, y_test, image_paths = load_test_data()
    
    if len(X_test) == 0:
        print("[ERROR] No test data loaded!")
        return
    
    print(f"Total test samples: {len(X_test)}")
    
    # Initialize predictor
    print(f"\n[STEP 2] Loading {model_type.upper()} models...")
    predictor = HierarchicalPredictor(
        model_type=model_type,
        super_threshold=0.40,
        fine_threshold=0.60
    )
    
    # Make predictions
    print("\n[STEP 3] Making predictions...")
    y_pred = []
    confidences = []
    
    for i, features in enumerate(X_test):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(X_test)} samples...")
        
        class_id, confidence = predictor.predict(features)
        y_pred.append(class_id)
        confidences.append(confidence)
    
    y_pred = np.array(y_pred)
    confidences = np.array(confidences)
    
    # Calculate metrics
    print("\n[STEP 4] Calculating metrics...")
    
    class_names = ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal', 'Trash', 'Unknown']
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    
    print(f"Average Confidence: {np.mean(confidences):.2%}")
    print(f"Min Confidence: {np.min(confidences):.2%}")
    print(f"Max Confidence: {np.max(confidences):.2%}")
    
    # Confusion matrix
    print("\n[STEP 5] Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_type.upper()} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    output_path = os.path.join(parent_dir, f'confusion_matrix_{model_type}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {output_path}")
    plt.close()
    
    # Find misclassified samples
    print("\n[STEP 6] Finding misclassified samples...")
    misclassified_indices = np.where(y_test != y_pred)[0]
    print(f"Total misclassified: {len(misclassified_indices)}")
    
    if len(misclassified_indices) > 0:
        print("\nFirst 10 misclassifications:")
        for i in misclassified_indices[:10]:
            true_class = class_names[y_test[i]]
            pred_class = class_names[y_pred[i]]
            conf = confidences[i]
            img_path = os.path.basename(image_paths[i])
            print(f"  {img_path}: True={true_class}, Predicted={pred_class}, Conf={conf:.2%}")
    
    # Per-class accuracy
    print("\n[STEP 7] Per-class accuracy:")
    for i, class_name in enumerate(class_names[:-1]):  # Exclude Unknown
        class_mask = (y_test == i)
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
            print(f"  {class_name}: {class_acc:.2%} ({class_mask.sum()} samples)")


def main():
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Evaluate SVM
    evaluate_model('svm')
    
    # Evaluate k-NN
    evaluate_model('knn')
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
