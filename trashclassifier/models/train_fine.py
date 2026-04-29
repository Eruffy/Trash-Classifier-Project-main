"""
Train fine-class classifiers (Stage 2 of hierarchical system).
For each super-class, trains a classifier for its fine-classes:
- Fiber → Paper, Cardboard
- Rigid → Plastic, Metal
- Transparent → Glass (single class, but still trained)
- Garbage → Trash (single class, but still trained)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time

from data.dataset_loader import DatasetLoader
from features.feature_extractor import FeatureExtractor
from data.label_mapping import (
    fine_to_super_class, super_class_id_to_name,
    fine_class_id_to_name, SUPER_TO_FINE
)
from config.config import (
    SAVED_MODELS_PATH, CLASS_WEIGHT, RANDOM_STATE,
    VERBOSE_TRAINING
)


class FineClassTrainer:
    """Trains fine-class classifiers for each super-class."""
    
    def __init__(self, model_type="svm"):
        self.model_type = model_type  # "svm" or "knn"
        self.extractor = FeatureExtractor()
        self.loader = DatasetLoader()
        self.models = {}  # Dictionary of models per super-class
        self.scalers = {}  # Dictionary of scalers per super-class
    
    def load_and_prepare_data(self, super_class_id):
        """Load dataset and filter for specific super-class."""
        # Load full dataset
        images, fine_labels, _ = self.loader.load_dataset(verbose=False)
        
        # Filter for this super-class
        fine_classes_in_super = SUPER_TO_FINE[super_class_id]
        mask = np.isin(fine_labels, fine_classes_in_super)
        
        filtered_images = images[mask]
        filtered_labels = fine_labels[mask]
        
        if len(filtered_images) == 0:
            print(f"   [WARNING] No images found for super-class {super_class_id}")
            return None, None, None, None
        
        # Split dataset
        X_train_img, X_val_img, y_train, y_val = self.loader.split_dataset(
            filtered_images, filtered_labels, verbose=False
        )
        
        print(f"   Training samples: {len(X_train_img)}")
        print(f"   Validation samples: {len(X_val_img)}")
        
        # Extract features
        print(f"   Extracting features...")
        X_train = self.extractor.extract_features_batch(X_train_img, verbose=False)
        X_val = self.extractor.extract_features_batch(X_val_img, verbose=False)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Store scaler
        self.scalers[super_class_id] = scaler
        
        return X_train_scaled, X_val_scaled, y_train, y_val
    
    def train_classifier(self, X_train, y_train, params=None):
        """Train classifier (SVM or KNN)."""
        if self.model_type == "svm":
            if params is None:
                params = {
                    "C": 10,
                    "gamma": "scale",
                    "kernel": "rbf",
                    "probability": True,
                    "class_weight": CLASS_WEIGHT,
                    "random_state": RANDOM_STATE
                }
            model = SVC(**params)
        else:  # knn
            if params is None:
                params = {
                    "n_neighbors": 5,
                    "weights": "distance",
                    "metric": "euclidean"
                }
            model = KNeighborsClassifier(**params)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        print(f"   ✓ Trained in {train_time:.1f}s")
        
        return model
    
    def evaluate(self, model, X_val, y_val, super_class_name):
        """Evaluate model on validation set."""
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"\n   🎯 Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Get class names
        unique_classes = sorted(np.unique(y_val))
        class_names = [fine_class_id_to_name(c) for c in unique_classes]
        
        if len(unique_classes) > 1:  # Only show report if multiple classes
            print("\n   Classification Report:")
            print("   " + "-" * 60)
            report = classification_report(y_val, y_pred, target_names=class_names, digits=4)
            for line in report.split('\n'):
                print(f"   {line}")
            
            print("\n   Confusion Matrix:")
            print("   " + "-" * 60)
            cm = confusion_matrix(y_val, y_pred)
            
            # Print header
            print(f"   {'':12}", end="")
            for name in class_names:
                print(f"{name[:10]:>12}", end="")
            print()
            
            # Print matrix
            for i, name in enumerate(class_names):
                print(f"   {name[:12]:12}", end="")
                for j in range(len(class_names)):
                    print(f"{cm[i, j]:12}", end="")
                print()
        else:
            print(f"   Only one class present: {class_names[0]}")
        
        return accuracy
    
    def train_for_super_class(self, super_class_id, params=None):
        """Train fine-class classifier for a specific super-class."""
        super_name = super_class_id_to_name(super_class_id)
        fine_classes = SUPER_TO_FINE[super_class_id]
        fine_names = [fine_class_id_to_name(fc) for fc in fine_classes]
        
        print(f"\n{'='*70}")
        print(f"TRAINING: {super_name} → {', '.join(fine_names)}")
        print(f"{'='*70}")
        
        # Load and prepare data
        X_train, X_val, y_train, y_val = self.load_and_prepare_data(super_class_id)
        
        if X_train is None:
            print(f"   [SKIPPED] No data for {super_name}")
            return None
        
        # Train model
        print(f"   Training {self.model_type.upper()}...")
        model = self.train_classifier(X_train, y_train, params)
        
        # Evaluate
        accuracy = self.evaluate(model, X_val, y_val, super_name)
        
        # Store model
        self.models[super_class_id] = model
        
        return accuracy
    
    def save_models(self):
        """Save all models and scalers."""
        os.makedirs(SAVED_MODELS_PATH, exist_ok=True)
        
        print(f"\n💾 Saving models...")
        
        for super_class_id, model in self.models.items():
            super_name = super_class_id_to_name(super_class_id).lower()
            
            model_path = os.path.join(
                SAVED_MODELS_PATH,
                f"fine_{super_name}_{self.model_type}.joblib"
            )
            scaler_path = os.path.join(
                SAVED_MODELS_PATH,
                f"fine_{super_name}_{self.model_type}_scaler.joblib"
            )
            
            joblib.dump(model, model_path)
            joblib.dump(self.scalers[super_class_id], scaler_path)
            
            print(f"   ✓ {super_class_id_to_name(super_class_id)}: {model_path}")
    
    def train_all(self, params_dict=None):
        """Train fine-class classifiers for all super-classes."""
        print("=" * 70)
        print(f"FINE-CLASS {self.model_type.upper()} TRAINING")
        print("=" * 70)
        
        accuracies = {}
        
        # Train for each super-class
        for super_class_id in sorted(SUPER_TO_FINE.keys()):
            params = params_dict.get(super_class_id) if params_dict else None
            accuracy = self.train_for_super_class(super_class_id, params)
            
            if accuracy is not None:
                accuracies[super_class_id] = accuracy
        
        # Save all models
        self.save_models()
        
        # Print summary
        print("\n" + "=" * 70)
        print(f"FINE-CLASS {self.model_type.upper()} TRAINING COMPLETE")
        print("=" * 70)
        for super_id, acc in accuracies.items():
            super_name = super_class_id_to_name(super_id)
            print(f"{super_name:12}: {acc:.4f} ({acc*100:.2f}%)")
        
        if accuracies:
            avg_accuracy = np.mean(list(accuracies.values()))
            print(f"\nAverage Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
        print("=" * 70)
        
        return accuracies


def main():
    """Train both SVM and KNN fine-class classifiers."""
    print("\n" + "=" * 70)
    print("TRAINING FINE-CLASS CLASSIFIERS")
    print("=" * 70)
    
    # Train SVM
    print("\n[1/2] Training SVM fine-class classifiers...")
    svm_trainer = FineClassTrainer(model_type="svm")
    svm_accuracies = svm_trainer.train_all()
    
    # Train KNN
    print("\n\n[2/2] Training KNN fine-class classifiers...")
    knn_trainer = FineClassTrainer(model_type="knn")
    knn_accuracies = knn_trainer.train_all()
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    
    print("\nSVM Accuracies:")
    for super_id, acc in svm_accuracies.items():
        print(f"  {super_class_id_to_name(super_id):12}: {acc:.4f}")
    
    print("\nKNN Accuracies:")
    for super_id, acc in knn_accuracies.items():
        print(f"  {super_class_id_to_name(super_id):12}: {acc:.4f}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
