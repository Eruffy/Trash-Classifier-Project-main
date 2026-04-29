"""
Hierarchical Predictor Module
Implements 2-stage hierarchical classification with confidence-based rejection
"""
import os
import sys
import numpy as np
import cv2
import joblib

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from config import Config
from features.feature_extractor import extract_features


class HierarchicalPredictor:
    """
    Hierarchical waste classifier with 2-stage prediction
    Stage 1: Super-class (Fiber, Rigid, Transparent, Garbage)
    Stage 2: Fine-class (Paper/Cardboard, Plastic/Metal, Glass, Trash)
    """
    
    # Super-class to fine-class mapping
    SUPER_TO_FINE = {
        0: ['paper', 'cardboard'],      # Fiber
        1: ['plastic', 'metal'],        # Rigid
        2: ['glass'],                   # Transparent
        3: ['trash']                    # Garbage
    }
    
    # Fine-class name to ID mapping
    FINE_CLASS_IDS = {
        'glass': 0, 'paper': 1, 'cardboard': 2,
        'plastic': 3, 'metal': 4, 'trash': 5
    }
    
    SUPER_CLASS_NAMES = ['Fiber', 'Rigid', 'Transparent', 'Garbage']
    
    def __init__(self, model_type='svm', 
                 super_threshold=None, fine_threshold=None):
        """
        Initialize predictor
        
        Args:
            model_type: 'svm' or 'knn'
            super_threshold: Confidence threshold for super-class
            fine_threshold: Confidence threshold for fine-class
        """
        self.model_type = model_type
        self.super_threshold = super_threshold or Config.SUPER_THRESHOLD
        self.fine_threshold = fine_threshold or Config.FINE_THRESHOLD
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load all trained models"""
        models_path = Config.SAVED_MODELS_PATH
        
        try:
            # Super-class models
            if self.model_type == 'svm':
                self.super_model = joblib.load(os.path.join(models_path, 'super_svm.pkl'))
                self.super_scaler = joblib.load(os.path.join(models_path, 'super_svm_scaler.pkl'))
                
                # Fine-class models
                self.fiber_model = joblib.load(os.path.join(models_path, 'fiber_svm.pkl'))
                self.fiber_scaler = joblib.load(os.path.join(models_path, 'fiber_svm_scaler.pkl'))
                
                self.rigid_model = joblib.load(os.path.join(models_path, 'rigid_svm.pkl'))
                self.rigid_scaler = joblib.load(os.path.join(models_path, 'rigid_svm_scaler.pkl'))
                
            else:  # knn
                self.super_model = joblib.load(os.path.join(models_path, 'super_knn.pkl'))
                self.super_scaler = joblib.load(os.path.join(models_path, 'super_knn_scaler.pkl'))
                
                # Fine-class models
                self.fiber_model = joblib.load(os.path.join(models_path, 'fiber_knn.pkl'))
                self.fiber_scaler = joblib.load(os.path.join(models_path, 'fiber_knn_scaler.pkl'))
                
                self.rigid_model = joblib.load(os.path.join(models_path, 'rigid_knn.pkl'))
                self.rigid_scaler = joblib.load(os.path.join(models_path, 'rigid_knn_scaler.pkl'))
            
            print(f"[INFO] Loaded {self.model_type.upper()} models successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to load {self.model_type.upper()} models: {e}")
            raise
    
    def predict(self, image):
        """
        Predict class for a single image
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            (class_id, confidence): Predicted class ID and confidence
        """
        # Resize and extract features
        if image.shape[:2] != (Config.IMAGE_SIZE, Config.IMAGE_SIZE):
            image = cv2.resize(image, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        
        features = extract_features(image)
        features = features.reshape(1, -1)
        
        # Stage 1: Super-class prediction
        features_scaled = self.super_scaler.transform(features)
        super_probs = self.super_model.predict_proba(features_scaled)[0]
        super_class = np.argmax(super_probs)
        super_conf = super_probs[super_class]
        
        # Debug for k-NN only
        if self.model_type == 'knn' and np.random.rand() < 0.1:  # 10% of frames
            print(f"[DEBUG KNN] Super: {self.SUPER_CLASS_NAMES[super_class]} ({super_conf:.2f}) | Probs: {super_probs}")
        
        # Check super-class confidence
        if super_conf < self.super_threshold:
            return 6, super_conf * 100  # Unknown class
        
        # Stage 2: Fine-class prediction
        if super_class == 0:  # Fiber → Paper or Cardboard
            fine_features = self.fiber_scaler.transform(features)
            fine_probs = self.fiber_model.predict_proba(fine_features)[0]
            fine_class_idx = np.argmax(fine_probs)
            fine_conf = fine_probs[fine_class_idx]
            
            # Map to final class
            fine_name = ['paper', 'cardboard'][fine_class_idx]
            
        elif super_class == 1:  # Rigid → Plastic or Metal
            fine_features = self.rigid_scaler.transform(features)
            fine_probs = self.rigid_model.predict_proba(fine_features)[0]
            fine_class_idx = np.argmax(fine_probs)
            fine_conf = fine_probs[fine_class_idx]
            
            # Map to final class
            fine_name = ['plastic', 'metal'][fine_class_idx]
            
        elif super_class == 2:  # Transparent → Glass
            fine_name = 'glass'
            fine_conf = super_conf  # Use super confidence
            
        else:  # Garbage → Trash
            fine_name = 'trash'
            fine_conf = super_conf  # Use super confidence
        
        # Check fine-class confidence
        if fine_conf < self.fine_threshold:
            return 6, fine_conf * 100  # Unknown class
        
        # Get final class ID
        class_id = self.FINE_CLASS_IDS[fine_name]
        confidence = fine_conf * 100
        
        return class_id, confidence
    
    def predict_with_details(self, image):
        """
        Predict with detailed stage information
        
        Returns:
            dict with super_class, super_conf, fine_class, fine_conf, final_class, final_conf
        """
        # Resize and extract features
        if image.shape[:2] != (Config.IMAGE_SIZE, Config.IMAGE_SIZE):
            image = cv2.resize(image, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        
        features = extract_features(image)
        features = features.reshape(1, -1)
        
        # Stage 1: Super-class
        features_scaled = self.super_scaler.transform(features)
        super_probs = self.super_model.predict_proba(features_scaled)[0]
        super_class = np.argmax(super_probs)
        super_conf = super_probs[super_class]
        
        result = {
            'super_class': super_class,
            'super_class_name': self.SUPER_CLASS_NAMES[super_class],
            'super_confidence': super_conf * 100,
            'super_probs': super_probs * 100
        }
        
        if super_conf < self.super_threshold:
            result['final_class'] = 6
            result['final_class_name'] = 'Unknown'
            result['final_confidence'] = super_conf * 100
            return result
        
        # Stage 2: Fine-class
        if super_class == 0:  # Fiber
            fine_features = self.fiber_scaler.transform(features)
            fine_probs = self.fiber_model.predict_proba(fine_features)[0]
            fine_class_idx = np.argmax(fine_probs)
            fine_conf = fine_probs[fine_class_idx]
            fine_name = ['paper', 'cardboard'][fine_class_idx]
            
        elif super_class == 1:  # Rigid
            fine_features = self.rigid_scaler.transform(features)
            fine_probs = self.rigid_model.predict_proba(fine_features)[0]
            fine_class_idx = np.argmax(fine_probs)
            fine_conf = fine_probs[fine_class_idx]
            fine_name = ['plastic', 'metal'][fine_class_idx]
            
        elif super_class == 2:  # Transparent
            fine_name = 'glass'
            fine_conf = super_conf
            
        else:  # Garbage
            fine_name = 'trash'
            fine_conf = super_conf
        
        result['fine_class_name'] = fine_name
        result['fine_confidence'] = fine_conf * 100
        
        if fine_conf < self.fine_threshold:
            result['final_class'] = 6
            result['final_class_name'] = 'Unknown'
            result['final_confidence'] = fine_conf * 100
        else:
            result['final_class'] = self.FINE_CLASS_IDS[fine_name]
            result['final_class_name'] = fine_name.capitalize()
            result['final_confidence'] = fine_conf * 100
        
        return result


if __name__ == "__main__":
    # Test predictor
    print("Testing predictor...")
    
    predictor_svm = HierarchicalPredictor(model_type='svm')
    predictor_knn = HierarchicalPredictor(model_type='knn')
    
    print("✅ Predictor initialized successfully!")
    print(f"   SVM thresholds: super={predictor_svm.super_threshold}, fine={predictor_svm.fine_threshold}")
    print(f"   k-NN thresholds: super={predictor_knn.super_threshold}, fine={predictor_knn.fine_threshold}")
