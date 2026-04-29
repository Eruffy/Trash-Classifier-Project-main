"""
Model Selector - Compare predictions between original and augmented models
Test any image with both model sets side-by-side
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


class ModelSelector:
    """Load and compare both original and augmented models"""
    
    def __init__(self):
        models_path = Config.SAVED_MODELS_PATH
        
        # Super-class mappings
        self.super_classes = ['Fiber', 'Rigid', 'Transparent', 'Garbage']
        self.fiber_classes = ['Paper', 'Cardboard']
        self.rigid_classes = ['Plastic', 'Metal']
        
        print("[INFO] Loading original models...")
        self.original = self._load_models('')
        
        print("[INFO] Loading augmented models...")
        self.augmented = self._load_models('_aug')
        
        print("[SUCCESS] All models loaded!\n")
    
    def _load_models(self, suffix):
        """Load a set of models (original or augmented)"""
        models_path = Config.SAVED_MODELS_PATH
        models = {}
        
        # Super-class models
        models['super_svm'] = joblib.load(os.path.join(models_path, f'super_svm{suffix}.pkl'))
        models['super_svm_scaler'] = joblib.load(os.path.join(models_path, f'super_svm_scaler{suffix}.pkl'))
        models['super_knn'] = joblib.load(os.path.join(models_path, f'super_knn{suffix}.pkl'))
        models['super_knn_scaler'] = joblib.load(os.path.join(models_path, f'super_knn_scaler{suffix}.pkl'))
        
        # Fiber models
        models['fiber_svm'] = joblib.load(os.path.join(models_path, f'fiber_svm{suffix}.pkl'))
        models['fiber_svm_scaler'] = joblib.load(os.path.join(models_path, f'fiber_svm_scaler{suffix}.pkl'))
        models['fiber_knn'] = joblib.load(os.path.join(models_path, f'fiber_knn{suffix}.pkl'))
        models['fiber_knn_scaler'] = joblib.load(os.path.join(models_path, f'fiber_knn_scaler{suffix}.pkl'))
        
        # Rigid models
        models['rigid_svm'] = joblib.load(os.path.join(models_path, f'rigid_svm{suffix}.pkl'))
        models['rigid_svm_scaler'] = joblib.load(os.path.join(models_path, f'rigid_svm_scaler{suffix}.pkl'))
        models['rigid_knn'] = joblib.load(os.path.join(models_path, f'rigid_knn{suffix}.pkl'))
        models['rigid_knn_scaler'] = joblib.load(os.path.join(models_path, f'rigid_knn_scaler{suffix}.pkl'))
        
        return models
    
    def predict_with_models(self, image_path, model_set, classifier_type='svm'):
        """
        Predict using specific model set and classifier
        
        Args:
            image_path: Path to image
            model_set: 'original' or 'augmented'
            classifier_type: 'svm' or 'knn'
        """
        # Select model set
        models = self.original if model_set == 'original' else self.augmented
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            return {'error': f'Could not load image: {image_path}'}
        
        img_resized = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        features = extract_features(img_resized)
        features = features.reshape(1, -1)
        
        # Stage 1: Super-class classification
        super_key = f'super_{classifier_type}'
        super_scaler_key = f'super_{classifier_type}_scaler'
        
        features_scaled = models[super_scaler_key].transform(features)
        super_pred = models[super_key].predict(features_scaled)[0]
        super_prob = models[super_key].predict_proba(features_scaled)[0]
        super_confidence = super_prob[super_pred]
        
        super_class = self.super_classes[super_pred]
        
        # Stage 2: Fine-class classification
        if super_class == 'Fiber':
            fine_key = f'fiber_{classifier_type}'
            fine_scaler_key = f'fiber_{classifier_type}_scaler'
            
            features_scaled = models[fine_scaler_key].transform(features)
            fine_pred = models[fine_key].predict(features_scaled)[0]
            fine_prob = models[fine_key].predict_proba(features_scaled)[0]
            fine_confidence = fine_prob[fine_pred]
            
            final_class = self.fiber_classes[fine_pred]
            
        elif super_class == 'Rigid':
            fine_key = f'rigid_{classifier_type}'
            fine_scaler_key = f'rigid_{classifier_type}_scaler'
            
            features_scaled = models[fine_scaler_key].transform(features)
            fine_pred = models[fine_key].predict(features_scaled)[0]
            fine_prob = models[fine_key].predict_proba(features_scaled)[0]
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
            'model_set': model_set,
            'classifier': classifier_type.upper()
        }
    
    def compare_all(self, image_path):
        """Compare predictions from all model combinations"""
        
        print("\n" + "="*70)
        print("COMPARING ALL MODEL COMBINATIONS".center(70))
        print("="*70)
        print(f"\nImage: {os.path.basename(image_path)}\n")
        
        results = {}
        
        # Test all combinations
        for model_set in ['original', 'augmented']:
            for classifier in ['svm', 'knn']:
                key = f"{model_set}_{classifier}"
                results[key] = self.predict_with_models(image_path, model_set, classifier)
        
        # Display results
        print("-" * 70)
        print(f"{'Model':<25} {'Prediction':<15} {'Super Class':<15} {'Confidence':<15}")
        print("-" * 70)
        
        for key, result in results.items():
            if 'error' in result:
                print(f"{key:<25} ERROR")
                continue
            
            model_label = f"{result['model_set'].title()} {result['classifier']}"
            pred = result['predicted_class']
            super_cls = result['super_class']
            conf = f"{result['final_confidence']:.1%}"
            
            print(f"{model_label:<25} {pred:<15} {super_cls:<15} {conf:<15}")
        
        print("-" * 70)
        
        # Show consensus
        predictions = [r['predicted_class'] for r in results.values() if 'predicted_class' in r]
        if predictions:
            from collections import Counter
            most_common = Counter(predictions).most_common(1)[0]
            consensus_class = most_common[0]
            consensus_count = most_common[1]
            
            print(f"\n{'Consensus:':<25} {consensus_class:<15} ({consensus_count}/4 models agree)")
        
        # Show best confidence
        best_result = max(results.values(), 
                         key=lambda x: x.get('final_confidence', 0))
        if 'predicted_class' in best_result:
            print(f"{'Highest Confidence:':<25} {best_result['predicted_class']:<15} "
                  f"({best_result['model_set'].title()} {best_result['classifier']} - "
                  f"{best_result['final_confidence']:.1%})")
        
        print("="*70 + "\n")
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare predictions across all models')
    parser.add_argument('image', type=str, help='Path to image file')
    parser.add_argument('--model', type=str, choices=['original', 'augmented', 'all'],
                       default='all', help='Which models to test (default: all)')
    parser.add_argument('--classifier', type=str, choices=['svm', 'knn', 'both'],
                       default='both', help='Which classifier to use (default: both)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)
    
    selector = ModelSelector()
    
    if args.model == 'all' and args.classifier == 'both':
        # Compare all models
        selector.compare_all(args.image)
    else:
        # Test specific model(s)
        models_to_test = ['original', 'augmented'] if args.model == 'all' else [args.model]
        classifiers_to_test = ['svm', 'knn'] if args.classifier == 'both' else [args.classifier]
        
        print("\n" + "="*70)
        print("MODEL PREDICTIONS".center(70))
        print("="*70)
        print(f"\nImage: {os.path.basename(args.image)}\n")
        
        for model_set in models_to_test:
            for classifier in classifiers_to_test:
                result = selector.predict_with_models(args.image, model_set, classifier)
                
                if 'error' in result:
                    print(f"\n[ERROR] {result['error']}")
                    continue
                
                print(f"\n--- {model_set.title()} {classifier.upper()} ---")
                print(f"Predicted: {result['predicted_class']}")
                print(f"Super Class: {result['super_class']}")
                print(f"Confidence: {result['final_confidence']:.2%}")
        
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
