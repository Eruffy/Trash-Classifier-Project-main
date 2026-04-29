"""
Quick prediction script using the BEST models from comparison
Based on the training results, this uses the augmented SVM models which showed the best performance
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


class BestModelPredictor:
    """
    Uses the best performing models from the comparison:
    - Super-class: Augmented SVM (75.33%)
    - Fiber: Augmented SVM (96.50%)
    - Rigid: Augmented SVM (96.50%)
    """
    
    def __init__(self):
        models_path = Config.SAVED_MODELS_PATH
        
        # Load super-class models (augmented SVM)
        print("[INFO] Loading best super-class model (Augmented SVM)...")
        self.super_svm = joblib.load(os.path.join(models_path, 'super_svm_aug.pkl'))
        self.super_scaler = joblib.load(os.path.join(models_path, 'super_svm_scaler_aug.pkl'))
        
        # Load fiber models (augmented SVM)
        print("[INFO] Loading best fiber model (Augmented SVM)...")
        self.fiber_svm = joblib.load(os.path.join(models_path, 'fiber_svm_aug.pkl'))
        self.fiber_scaler = joblib.load(os.path.join(models_path, 'fiber_svm_scaler_aug.pkl'))
        
        # Load rigid models (augmented SVM)
        print("[INFO] Loading best rigid model (Augmented SVM)...")
        self.rigid_svm = joblib.load(os.path.join(models_path, 'rigid_svm_aug.pkl'))
        self.rigid_scaler = joblib.load(os.path.join(models_path, 'rigid_svm_scaler_aug.pkl'))
        
        # Class mappings
        self.super_classes = ['Fiber', 'Rigid', 'Transparent', 'Garbage']
        self.fiber_classes = ['Paper', 'Cardboard']
        self.rigid_classes = ['Plastic', 'Metal']
        
        print("[SUCCESS] All best models loaded!\n")
    
    def predict(self, image_path):
        """
        Predict the class of an image using the best hierarchical model
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Prediction results with class name and confidence
        """
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            return {'error': f'Could not load image: {image_path}'}
        
        img_resized = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        features = extract_features(img_resized)
        features = features.reshape(1, -1)
        
        # Stage 1: Super-class classification
        features_scaled = self.super_scaler.transform(features)
        super_pred = self.super_svm.predict(features_scaled)[0]
        super_prob = self.super_svm.predict_proba(features_scaled)[0]
        super_confidence = super_prob[super_pred]
        
        super_class = self.super_classes[super_pred]
        
        # Stage 2: Fine-class classification (if applicable)
        if super_class == 'Fiber':
            # Paper vs Cardboard
            features_scaled = self.fiber_scaler.transform(features)
            fine_pred = self.fiber_svm.predict(features_scaled)[0]
            fine_prob = self.fiber_svm.predict_proba(features_scaled)[0]
            fine_confidence = fine_prob[fine_pred]
            
            final_class = self.fiber_classes[fine_pred]
            
        elif super_class == 'Rigid':
            # Plastic vs Metal
            features_scaled = self.rigid_scaler.transform(features)
            fine_pred = self.rigid_svm.predict(features_scaled)[0]
            fine_prob = self.rigid_svm.predict_proba(features_scaled)[0]
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
            'model_type': 'Augmented SVM (Best Performance)'
        }
    
    def predict_batch(self, image_paths):
        """Predict multiple images"""
        results = []
        for img_path in image_paths:
            result = self.predict(img_path)
            result['image_path'] = img_path
            results.append(result)
        return results


def predict_image(image_path):
    """Quick function to predict a single image"""
    predictor = BestModelPredictor()
    result = predictor.predict(image_path)
    
    if 'error' in result:
        print(f"[ERROR] {result['error']}")
        return
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS (BEST MODELS)".center(60))
    print("="*60)
    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Super Class: {result['super_class']}")
    print(f"Super Confidence: {result['super_confidence']:.2%}")
    print(f"Final Confidence: {result['final_confidence']:.2%}")
    print(f"Model: {result['model_type']}")
    print("="*60 + "\n")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict using best models')
    parser.add_argument('image', type=str, help='Path to image file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)
    
    predict_image(args.image)
