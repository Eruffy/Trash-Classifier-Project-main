"""
Ensemble Predictor - Combines SVM and KNN Predictions via Voting
"""
import sys
import os
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, parent_dir)

import numpy as np
import cv2
from collections import Counter

from inference.predictor import HierarchicalPredictor


class EnsemblePredictor:
    """Ensemble classifier combining SVM and KNN via weighted voting"""
    
    def __init__(self, use_optimized=False, svm_weight=1.0, knn_weight=1.0):
        """
        Initialize ensemble predictor
        
        Args:
            use_optimized: Use models trained with optimized hyperparameters
            svm_weight: Weight for SVM predictions
            knn_weight: Weight for KNN predictions
        """
        self.svm_predictor = HierarchicalPredictor(model_type='svm', use_optimized=use_optimized)
        self.knn_predictor = HierarchicalPredictor(model_type='knn', use_optimized=use_optimized)
        self.svm_weight = svm_weight
        self.knn_weight = knn_weight
    
    def predict(self, image, return_confidence=True, voting_strategy='soft'):
        """
        Predict class using ensemble of SVM and KNN
        
        Args:
            image: OpenCV image (BGR format)
            return_confidence: Whether to return confidence score
            voting_strategy: 'hard' (majority vote) or 'soft' (weighted confidence)
            
        Returns:
            If return_confidence=True: (class_id, confidence, agreement_score)
            If return_confidence=False: class_id
        """
        # Get predictions from both models
        svm_class, svm_conf = self.svm_predictor.predict(image, return_confidence=True)
        knn_class, knn_conf = self.knn_predictor.predict(image, return_confidence=True)
        
        if voting_strategy == 'hard':
            # Hard voting: majority wins
            if svm_class == knn_class:
                # Both agree
                final_class = svm_class
                final_conf = (svm_conf + knn_conf) / 2
                agreement = 1.0
            else:
                # Disagree: use the one with higher confidence
                if svm_conf * self.svm_weight > knn_conf * self.knn_weight:
                    final_class = svm_class
                    final_conf = svm_conf
                else:
                    final_class = knn_class
                    final_conf = knn_conf
                agreement = 0.0
        
        else:  # soft voting
            # Soft voting: weighted confidence
            if svm_class == knn_class:
                # Both agree - high confidence
                final_class = svm_class
                final_conf = (svm_conf * self.svm_weight + knn_conf * self.knn_weight) / (self.svm_weight + self.knn_weight)
                agreement = 1.0
            else:
                # Disagree - choose based on weighted confidence
                svm_weighted = svm_conf * self.svm_weight
                knn_weighted = knn_conf * self.knn_weight
                
                if svm_weighted > knn_weighted:
                    final_class = svm_class
                    final_conf = svm_conf
                else:
                    final_class = knn_class
                    final_conf = knn_conf
                
                # Agreement score based on confidence difference
                conf_diff = abs(svm_weighted - knn_weighted)
                agreement = 1.0 - min(conf_diff, 1.0)
        
        if return_confidence:
            return final_class, final_conf, agreement
        return final_class
    
    def predict_with_details(self, image):
        """
        Predict with detailed information from both models
        
        Returns:
            dict with svm_prediction, knn_prediction, ensemble_prediction, agreement
        """
        svm_details = self.svm_predictor.predict_with_details(image)
        knn_details = self.knn_predictor.predict_with_details(image)
        
        # Ensemble prediction
        ensemble_class, ensemble_conf, agreement = self.predict(image, return_confidence=True, voting_strategy='soft')
        
        return {
            'svm': svm_details,
            'knn': knn_details,
            'ensemble': {
                'final_class': ensemble_class,
                'final_confidence': ensemble_conf,
                'agreement': agreement
            }
        }
    
    def predict_batch(self, images):
        """
        Predict classes for a batch of images
        
        Args:
            images: List of OpenCV images
            
        Returns:
            List of (class_id, confidence, agreement) tuples
        """
        results = []
        for img in images:
            results.append(self.predict(img, return_confidence=True))
        return results
