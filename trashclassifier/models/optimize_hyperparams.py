"""
Hyperparameter optimization using GridSearchCV.
Finds optimal parameters for SVM and KNN models.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import time

from data.dataset_loader import DatasetLoader
from features.feature_extractor import FeatureExtractor
from data.label_mapping import fine_to_super_class
from config.config import (
    SVM_PARAM_GRID, KNN_PARAM_GRID,
    GRID_SEARCH_CV, GRID_SEARCH_JOBS,
    CLASS_WEIGHT, RANDOM_STATE
)


class HyperparameterOptimizer:
    """Optimizes hyperparameters using GridSearchCV."""
    
    def __init__(self, model_type="svm", target="super"):
        self.model_type = model_type
        self.target = target  # "super" or super_class_id for fine
        self.extractor = FeatureExtractor()
        self.loader = DatasetLoader()
    
    def optimize_super_class(self):
        """Optimize hyperparameters for super-class classifier."""
        print("=" * 70)
        print(f"HYPERPARAMETER OPTIMIZATION: SUPER-CLASS {self.model_type.upper()}")
        print("=" * 70)
        
        # Load data
        print("\n📦 Loading dataset...")
        images, fine_labels, _ = self.loader.load_dataset(verbose=False)
        super_labels = np.array([fine_to_super_class(fl) for fl in fine_labels])
        
        # Split
        X_train_img, _, y_train, _ = self.loader.split_dataset(
            images, super_labels, verbose=False
        )
        
        # Extract features
        print("🔧 Extracting features...")
        X_train = self.extractor.extract_features_batch(X_train_img, verbose=True)
        
        # Scale
        print("📊 Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Setup GridSearch
        if self.model_type == "svm":
            base_model = SVC(
                probability=True,
                class_weight=CLASS_WEIGHT,
                random_state=RANDOM_STATE
            )
            param_grid = SVM_PARAM_GRID
        else:
            base_model = KNeighborsClassifier()
            param_grid = KNN_PARAM_GRID
        
        print(f"\n🔍 Running GridSearchCV with {GRID_SEARCH_CV}-fold CV...")
        print(f"   Parameter grid: {param_grid}")
        print(f"   Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=GRID_SEARCH_CV,
            scoring='accuracy',
            n_jobs=GRID_SEARCH_JOBS,
            verbose=2
        )
        
        start_time = time.time()
        grid_search.fit(X_train_scaled, y_train)
        search_time = time.time() - start_time
        
        print(f"\n✓ GridSearch completed in {search_time:.1f}s")
        print(f"\n🏆 Best Parameters: {grid_search.best_params_}")
        print(f"🎯 Best CV Score: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")
        
        # Show top 5 configurations
        print("\n📊 Top 5 Configurations:")
        print("-" * 70)
        results = grid_search.cv_results_
        indices = np.argsort(results['mean_test_score'])[::-1][:5]
        
        for i, idx in enumerate(indices, 1):
            print(f"\n{i}. Score: {results['mean_test_score'][idx]:.4f}")
            print(f"   Params: {results['params'][idx]}")
        
        return grid_search.best_params_, grid_search.best_score_


def main():
    """Run hyperparameter optimization for super-class models."""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    
    # Optimize SVM
    print("\n[1/2] Optimizing SVM...")
    svm_optimizer = HyperparameterOptimizer(model_type="svm", target="super")
    svm_best_params, svm_best_score = svm_optimizer.optimize_super_class()
    
    # Optimize KNN
    print("\n\n[2/2] Optimizing KNN...")
    knn_optimizer = HyperparameterOptimizer(model_type="knn", target="super")
    knn_best_params, knn_best_score = knn_optimizer.optimize_super_class()
    
    # Summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(f"\nSVM Best Parameters: {svm_best_params}")
    print(f"SVM Best Score: {svm_best_score:.4f} ({svm_best_score*100:.2f}%)")
    print(f"\nKNN Best Parameters: {knn_best_params}")
    print(f"KNN Best Score: {knn_best_score:.4f} ({knn_best_score*100:.2f}%)")
    print("=" * 70)
    
    print("\n💡 RECOMMENDATION:")
    print("Update the default parameters in config.py with these optimized values")
    print("before running the full training pipeline.")


if __name__ == "__main__":
    main()
