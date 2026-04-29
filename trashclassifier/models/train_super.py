"""
Train super-class classifier (Stage 1 of hierarchical system).
Classifies into: Fiber, Rigid, Transparent, Garbage
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
from data.label_mapping import fine_to_super_class, super_class_id_to_name
from config.config import (
    SAVED_MODELS_PATH, CLASS_WEIGHT, RANDOM_STATE,
    VERBOSE_TRAINING
)


class SuperClassTrainer:
    """Trains super-class classifier (Fiber/Rigid/Transparent/Garbage)."""
    
    def __init__(self, model_type="svm"):
        self.model_type = model_type  # "svm" or "knn"
        self.extractor = FeatureExtractor()
        self.loader = DatasetLoader()
        self.scaler = StandardScaler()
        self.model = None
    
    def load_and_prepare_data(self):
        """Load dataset and extract features."""
        print("=" * 70)
        print(f"SUPER-CLASS {self.model_type.upper()} TRAINING")
        print("=" * 70)
        
        # Load dataset
        images, fine_labels, _ = self.loader.load_dataset(verbose=VERBOSE_TRAINING)
        
        # Convert fine labels to super labels
        super_labels = np.array([fine_to_super_class(fl) for fl in fine_labels])
        
        # Split dataset
        X_train_img, X_val_img, y_train, y_val = self.loader.split_dataset(
            images, super_labels, verbose=VERBOSE_TRAINING
        )
        
        print("\n🔧 Extracting features from training set...")
        start_time = time.time()
        X_train = self.extractor.extract_features_batch(X_train_img, verbose=True)
        train_time = time.time() - start_time
        print(f"   Training features extracted in {train_time:.1f}s")
        
        print("\n🔧 Extracting features from validation set...")
        start_time = time.time()
        X_val = self.extractor.extract_features_batch(X_val_img, verbose=True)
        val_time = time.time() - start_time
        print(f"   Validation features extracted in {val_time:.1f}s")
        
        # Scale features
        print("\n📊 Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        return X_train_scaled, X_val_scaled, y_train, y_val
    
    def train_svm(self, X_train, y_train, params=None):
        """Train SVM classifier with specified or default parameters."""
        if params is None:
            # Default optimized parameters
            params = {
                "C": 10,
                "gamma": "scale",
                "kernel": "rbf",
                "probability": True,
                "class_weight": CLASS_WEIGHT,
                "random_state": RANDOM_STATE
            }
        
        print("\n🚀 Training SVM...")
        print(f"   Parameters: {params}")
        
        self.model = SVC(**params)
        
        start_time = time.time()
        self.model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        print(f"   ✓ SVM trained in {train_time:.1f}s")
        
        return self.model
    
    def train_knn(self, X_train, y_train, params=None):
        """Train KNN classifier with specified or default parameters."""
        if params is None:
            # Default optimized parameters
            params = {
                "n_neighbors": 5,
                "weights": "distance",
                "metric": "euclidean"
            }
        
        print("\n🚀 Training KNN...")
        print(f"   Parameters: {params}")
        
        self.model = KNeighborsClassifier(**params)
        
        start_time = time.time()
        self.model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        print(f"   ✓ KNN trained in {train_time:.1f}s")
        
        return self.model
    
    def evaluate(self, X_val, y_val):
        """Evaluate model on validation set."""
        print("\n📈 Evaluating on validation set...")
        
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"\n   🎯 Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\n   Classification Report:")
        print("   " + "-" * 60)
        class_names = [super_class_id_to_name(i) for i in sorted(np.unique(y_val))]
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
        
        return accuracy
    
    def save_model(self):
        """Save model and scaler."""
        os.makedirs(SAVED_MODELS_PATH, exist_ok=True)
        
        model_path = os.path.join(SAVED_MODELS_PATH, f"super_{self.model_type}.joblib")
        scaler_path = os.path.join(SAVED_MODELS_PATH, f"super_{self.model_type}_scaler.joblib")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"\n💾 Model saved:")
        print(f"   {model_path}")
        print(f"   {scaler_path}")
    
    def train(self, params=None):
        """Complete training pipeline."""
        # Load and prepare data
        X_train, X_val, y_train, y_val = self.load_and_prepare_data()
        
        # Train model
        if self.model_type == "svm":
            self.train_svm(X_train, y_train, params)
        else:
            self.train_knn(X_train, y_train, params)
        
        # Evaluate
        accuracy = self.evaluate(X_val, y_val)
        
        # Save
        self.save_model()
        
        print("\n" + "=" * 70)
        print(f"SUPER-CLASS {self.model_type.upper()} TRAINING COMPLETE")
        print(f"Final Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("=" * 70)
        
        return accuracy


def main():
    """Train both SVM and KNN super-class classifiers."""
    print("\n" + "=" * 70)
    print("TRAINING SUPER-CLASS CLASSIFIERS")
    print("=" * 70)
    
    # Train SVM
    print("\n[1/2] Training SVM...")
    svm_trainer = SuperClassTrainer(model_type="svm")
    svm_accuracy = svm_trainer.train()
    
    # Train KNN
    print("\n\n[2/2] Training KNN...")
    knn_trainer = SuperClassTrainer(model_type="knn")
    knn_accuracy = knn_trainer.train()
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"SVM Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
    print(f"KNN Accuracy: {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
