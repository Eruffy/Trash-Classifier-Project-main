"""
Hybrid Model Predictor - Best of Both Worlds
Uses the optimal combination of original and augmented models:

SVM Strategy: Augmented for ALL stages (75-96% accuracy)
KNN Strategy: Original for Super-class (77.83%), Augmented for Fine-class (87-94%)
"""
import os
import sys
import cv2
import numpy as np
import joblib

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from config import Config
from features.feature_extractor import extract_features


class HybridPredictor:
    """
    Hybrid model predictor with two strategies:
    1. SVM: All augmented models
    2. KNN: Original super + Augmented fine models
    """
    
    def __init__(self):
        models_path = Config.SAVED_MODELS_PATH
        
        # Class mappings
        self.super_classes = ['Fiber', 'Rigid', 'Transparent', 'Garbage']
        self.fiber_classes = ['Paper', 'Cardboard']
        self.rigid_classes = ['Plastic', 'Metal']
        
        print("[INFO] Loading Hybrid Model Configuration...")
        print("\n=== SVM Strategy: All Augmented ===")
        
        # SVM Models - All Augmented
        print("  Loading Super-class SVM (Augmented)...")
        self.svm_super = joblib.load(os.path.join(models_path, 'super_svm_aug.pkl'))
        self.svm_super_scaler = joblib.load(os.path.join(models_path, 'super_svm_scaler_aug.pkl'))
        
        print("  Loading Fiber SVM (Augmented)...")
        self.svm_fiber = joblib.load(os.path.join(models_path, 'fiber_svm_aug.pkl'))
        self.svm_fiber_scaler = joblib.load(os.path.join(models_path, 'fiber_svm_scaler_aug.pkl'))
        
        print("  Loading Rigid SVM (Augmented)...")
        self.svm_rigid = joblib.load(os.path.join(models_path, 'rigid_svm_aug.pkl'))
        self.svm_rigid_scaler = joblib.load(os.path.join(models_path, 'rigid_svm_scaler_aug.pkl'))
        
        print("\n=== KNN Strategy: Original Super + Augmented Fine ===")
        
        # KNN Models - Original for Super, Augmented for Fine
        print("  Loading Super-class KNN (Original)...")
        self.knn_super = joblib.load(os.path.join(models_path, 'super_knn.pkl'))
        self.knn_super_scaler = joblib.load(os.path.join(models_path, 'super_knn_scaler.pkl'))
        
        print("  Loading Fiber KNN (Augmented)...")
        self.knn_fiber = joblib.load(os.path.join(models_path, 'fiber_knn_aug.pkl'))
        self.knn_fiber_scaler = joblib.load(os.path.join(models_path, 'fiber_knn_scaler_aug.pkl'))
        
        print("  Loading Rigid KNN (Augmented)...")
        self.knn_rigid = joblib.load(os.path.join(models_path, 'rigid_knn_aug.pkl'))
        self.knn_rigid_scaler = joblib.load(os.path.join(models_path, 'rigid_knn_scaler_aug.pkl'))
        
        print("\n[SUCCESS] Hybrid models loaded!\n")
    
    def predict_svm(self, image_path):
        """
        Predict using SVM strategy (All Augmented)
        Expected accuracy: Super 75.33%, Fiber 96.50%, Rigid 96.50%
        """
        img = cv2.imread(image_path)
        if img is None:
            return {'error': f'Could not load image: {image_path}'}
        
        img_resized = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        features = extract_features(img_resized)
        features = features.reshape(1, -1)
        
        # Stage 1: Super-class (Augmented SVM)
        features_scaled = self.svm_super_scaler.transform(features)
        super_pred = self.svm_super.predict(features_scaled)[0]
        super_prob = self.svm_super.predict_proba(features_scaled)[0]
        super_confidence = super_prob[super_pred]
        
        super_class = self.super_classes[super_pred]
        
        # Stage 2: Fine-class (Augmented SVM)
        if super_class == 'Fiber':
            features_scaled = self.svm_fiber_scaler.transform(features)
            fine_pred = self.svm_fiber.predict(features_scaled)[0]
            fine_prob = self.svm_fiber.predict_proba(features_scaled)[0]
            fine_confidence = fine_prob[fine_pred]
            final_class = self.fiber_classes[fine_pred]
            
        elif super_class == 'Rigid':
            features_scaled = self.svm_rigid_scaler.transform(features)
            fine_pred = self.svm_rigid.predict(features_scaled)[0]
            fine_prob = self.svm_rigid.predict_proba(features_scaled)[0]
            fine_confidence = fine_prob[fine_pred]
            final_class = self.rigid_classes[fine_pred]
            
        elif super_class == 'Transparent':
            final_class = 'Glass'
            fine_confidence = super_confidence
        else:  # Garbage
            final_class = 'Trash'
            fine_confidence = super_confidence
        
        return {
            'predicted_class': final_class,
            'super_class': super_class,
            'super_confidence': float(super_confidence),
            'final_confidence': float(fine_confidence),
            'strategy': 'SVM (All Augmented)'
        }
    
    def predict_knn(self, image_path):
        """
        Predict using KNN strategy (Original Super + Augmented Fine)
        Expected accuracy: Super 77.83%, Fiber 87.50%, Rigid 94.00%
        """
        img = cv2.imread(image_path)
        if img is None:
            return {'error': f'Could not load image: {image_path}'}
        
        img_resized = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        features = extract_features(img_resized)
        features = features.reshape(1, -1)
        
        # Stage 1: Super-class (Original KNN)
        features_scaled = self.knn_super_scaler.transform(features)
        super_pred = self.knn_super.predict(features_scaled)[0]
        super_prob = self.knn_super.predict_proba(features_scaled)[0]
        super_confidence = super_prob[super_pred]
        
        super_class = self.super_classes[super_pred]
        
        # Stage 2: Fine-class (Augmented KNN)
        if super_class == 'Fiber':
            features_scaled = self.knn_fiber_scaler.transform(features)
            fine_pred = self.knn_fiber.predict(features_scaled)[0]
            fine_prob = self.knn_fiber.predict_proba(features_scaled)[0]
            fine_confidence = fine_prob[fine_pred]
            final_class = self.fiber_classes[fine_pred]
            
        elif super_class == 'Rigid':
            features_scaled = self.knn_rigid_scaler.transform(features)
            fine_pred = self.knn_rigid.predict(features_scaled)[0]
            fine_prob = self.knn_rigid.predict_proba(features_scaled)[0]
            fine_confidence = fine_prob[fine_pred]
            final_class = self.rigid_classes[fine_pred]
            
        elif super_class == 'Transparent':
            final_class = 'Glass'
            fine_confidence = super_confidence
        else:  # Garbage
            final_class = 'Trash'
            fine_confidence = super_confidence
        
        return {
            'predicted_class': final_class,
            'super_class': super_class,
            'super_confidence': float(super_confidence),
            'final_confidence': float(fine_confidence),
            'strategy': 'KNN (Original Super + Augmented Fine)'
        }
    
    def predict_both(self, image_path):
        """Predict with both strategies and return comparison"""
        svm_result = self.predict_svm(image_path)
        knn_result = self.predict_knn(image_path)
        
        return {
            'svm': svm_result,
            'knn': knn_result,
            'agreement': svm_result['predicted_class'] == knn_result['predicted_class']
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid model predictor')
    parser.add_argument('image', type=str, help='Path to image file')
    parser.add_argument('--mode', type=str, choices=['svm', 'knn', 'both'],
                       default='both', help='Prediction mode (default: both)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)
    
    predictor = HybridPredictor()
    
    print("\n" + "="*70)
    print("HYBRID MODEL PREDICTION".center(70))
    print("="*70)
    print(f"\nImage: {os.path.basename(args.image)}\n")
    
    if args.mode == 'both':
        results = predictor.predict_both(args.image)
        
        # SVM Results
        print("--- SVM Strategy (All Augmented) ---")
        svm = results['svm']
        print(f"Predicted: {svm['predicted_class']}")
        print(f"Super Class: {svm['super_class']}")
        print(f"Confidence: {svm['final_confidence']:.2%}")
        
        print("\n--- KNN Strategy (Original Super + Augmented Fine) ---")
        knn = results['knn']
        print(f"Predicted: {knn['predicted_class']}")
        print(f"Super Class: {knn['super_class']}")
        print(f"Confidence: {knn['final_confidence']:.2%}")
        
        print("\n" + "-"*70)
        if results['agreement']:
            print(f"✅ Both strategies agree: {svm['predicted_class']}")
        else:
            print(f"⚠️  Strategies disagree:")
            print(f"    SVM: {svm['predicted_class']} ({svm['final_confidence']:.1%})")
            print(f"    KNN: {knn['predicted_class']} ({knn['final_confidence']:.1%})")
        
    elif args.mode == 'svm':
        result = predictor.predict_svm(args.image)
        print(f"Predicted: {result['predicted_class']}")
        print(f"Super Class: {result['super_class']}")
        print(f"Confidence: {result['final_confidence']:.2%}")
        print(f"Strategy: {result['strategy']}")
        
    else:  # knn
        result = predictor.predict_knn(args.image)
        print(f"Predicted: {result['predicted_class']}")
        print(f"Super Class: {result['super_class']}")
        print(f"Confidence: {result['final_confidence']:.2%}")
        print(f"Strategy: {result['strategy']}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
