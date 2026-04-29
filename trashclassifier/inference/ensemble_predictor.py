"""
Ensemble Predictor Module
Combines SVM and k-NN predictions with confidence-weighted voting
"""
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from config import Config
from inference.predictor import HierarchicalPredictor


class EnsemblePredictor:
    """
    Ensemble predictor combining SVM and k-NN models
    Uses confidence-weighted voting for final prediction
    """
    
    CLASS_NAMES = ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal', 'Trash', 'Unknown']
    
    def __init__(self):
        """Initialize both SVM and k-NN predictors"""
        print("[INFO] Initializing Ensemble Predictor...")
        
        try:
            self.svm_predictor = HierarchicalPredictor(model_type='svm')
            self.knn_predictor = HierarchicalPredictor(model_type='knn')
            print("[INFO] Ensemble predictor ready!")
        except Exception as e:
            print(f"[ERROR] Failed to initialize ensemble: {e}")
            raise
    
    def predict_all(self, image):
        """
        Get predictions from both models and ensemble
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            dict with 'svm', 'knn', and 'ensemble' predictions
        """
        # Get SVM prediction
        svm_class_id, svm_conf = self.svm_predictor.predict(image)
        
        # Get k-NN prediction
        knn_class_id, knn_conf = self.knn_predictor.predict(image)
        
        # Ensemble prediction (confidence-weighted voting)
        if svm_class_id == knn_class_id:
            # Both agree - use average confidence
            ensemble_class_id = svm_class_id
            ensemble_conf = (svm_conf + knn_conf) / 2
        else:
            # Disagree - use higher confidence
            if svm_conf >= knn_conf:
                ensemble_class_id = svm_class_id
                ensemble_conf = svm_conf
            else:
                ensemble_class_id = knn_class_id
                ensemble_conf = knn_conf
        
        # Check if below unknown threshold
        if ensemble_conf < Config.UNKNOWN_THRESHOLD * 100:
            ensemble_class_id = 6  # Unknown
        
        return {
            'svm': {
                'class_id': svm_class_id,
                'class': self.CLASS_NAMES[svm_class_id],
                'confidence': svm_conf
            },
            'knn': {
                'class_id': knn_class_id,
                'class': self.CLASS_NAMES[knn_class_id],
                'confidence': knn_conf
            },
            'ensemble': {
                'class_id': ensemble_class_id,
                'class': self.CLASS_NAMES[ensemble_class_id],
                'confidence': ensemble_conf
            }
        }
    
    def predict(self, image):
        """
        Get ensemble prediction only
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            (class_id, confidence): Ensemble prediction
        """
        results = self.predict_all(image)
        return results['ensemble']['class_id'], results['ensemble']['confidence']
    
    def predict_batch(self, images):
        """
        Predict for multiple images
        
        Args:
            images: List of input images
        
        Returns:
            List of prediction dicts
        """
        return [self.predict_all(img) for img in images]


if __name__ == "__main__":
    # Test ensemble predictor
    print("Testing Ensemble Predictor...")
    
    try:
        predictor = EnsemblePredictor()
        print("✅ Ensemble predictor initialized successfully!")
        print(f"   Classes: {predictor.CLASS_NAMES}")
        print(f"   Unknown threshold: {Config.UNKNOWN_THRESHOLD}")
    except Exception as e:
        print(f"❌ Error: {e}")
