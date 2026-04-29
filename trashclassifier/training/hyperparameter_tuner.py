"""
Hyperparameter Tuning Module
Uses GridSearchCV to find optimal parameters for SVM and k-NN models
"""
import os
import sys
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, f1_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from config import Config
from features.feature_extractor import extract_features


def load_tuning_data(class_names, max_samples_per_class=300):
    """
    Load subset of data for hyperparameter tuning (faster)
    
    Args:
        class_names: List of class names to load
        max_samples_per_class: Maximum samples per class (for speed)
    """
    X = []
    y = []
    
    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(Config.DATASET_PATH, class_name)
        
        if not os.path.exists(class_path):
            continue
        
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit samples for speed
        images = images[:max_samples_per_class]
        
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            img_resized = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            features = extract_features(img_resized)
            
            X.append(features)
            y.append(idx)
    
    return np.array(X), np.array(y)


def tune_super_svm():
    """Tune super-class SVM hyperparameters"""
    
    print("\n" + "="*70)
    print("TUNING SUPER-CLASS SVM HYPERPARAMETERS".center(70))
    print("="*70 + "\n")
    
    # Load data
    print("[INFO] Loading tuning data...")
    class_names = ['paper', 'cardboard', 'plastic', 'metal', 'glass', 'trash']
    X, y = load_tuning_data(class_names, max_samples_per_class=200)
    
    # Map to super-classes
    super_map = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 3}  # paper/card→0, plastic/metal→1, glass→2, trash→3
    y_super = np.array([super_map[label] for label in y])
    
    print(f"[INFO] Loaded {len(X)} samples for tuning")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_super, test_size=0.25, random_state=Config.RANDOM_STATE, stratify=y_super
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Parameter grid
    param_grid = {
        'C': [1, 10, 50],
        'gamma': ['scale', 0.01, 0.001],
        'kernel': ['rbf']
    }
    
    print(f"[INFO] Testing {len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel'])} combinations...")
    
    # GridSearch
    svm = SVC(probability=True, random_state=Config.RANDOM_STATE)
    grid_search = GridSearchCV(
        svm, param_grid, cv=3, scoring='f1_weighted', 
        n_jobs=-1, verbose=1
    )
    
    print("[INFO] Running GridSearchCV...")
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\n[RESULT] Best parameters: {grid_search.best_params_}")
    print(f"[RESULT] Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_


def tune_super_knn():
    """Tune super-class k-NN hyperparameters"""
    
    print("\n" + "="*70)
    print("TUNING SUPER-CLASS K-NN HYPERPARAMETERS".center(70))
    print("="*70 + "\n")
    
    # Load data
    print("[INFO] Loading tuning data...")
    class_names = ['paper', 'cardboard', 'plastic', 'metal', 'glass', 'trash']
    X, y = load_tuning_data(class_names, max_samples_per_class=200)
    
    # Map to super-classes
    super_map = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 3}
    y_super = np.array([super_map[label] for label in y])
    
    print(f"[INFO] Loaded {len(X)} samples for tuning")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_super, test_size=0.25, random_state=Config.RANDOM_STATE, stratify=y_super
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Parameter grid
    param_grid = {
        'n_neighbors': [5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    print(f"[INFO] Testing {len(param_grid['n_neighbors']) * len(param_grid['weights']) * len(param_grid['metric'])} combinations...")
    
    # GridSearch
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(
        knn, param_grid, cv=3, scoring='f1_weighted',
        n_jobs=-1, verbose=1
    )
    
    print("[INFO] Running GridSearchCV...")
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\n[RESULT] Best parameters: {grid_search.best_params_}")
    print(f"[RESULT] Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_


def tune_fine_svm(super_class):
    """
    Tune fine-class SVM hyperparameters
    
    Args:
        super_class: 'fiber' or 'rigid'
    """
    
    print("\n" + "="*70)
    print(f"TUNING {super_class.upper()} FINE-CLASS SVM HYPERPARAMETERS".center(70))
    print("="*70 + "\n")
    
    # Load appropriate classes
    if super_class == 'fiber':
        class_names = ['paper', 'cardboard']
    else:  # rigid
        class_names = ['plastic', 'metal']
    
    print("[INFO] Loading tuning data...")
    X, y = load_tuning_data(class_names, max_samples_per_class=200)
    
    print(f"[INFO] Loaded {len(X)} samples for tuning")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=Config.RANDOM_STATE, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Parameter grid
    param_grid = {
        'C': [1, 10, 50],
        'gamma': ['scale', 0.01, 0.001],
        'kernel': ['rbf']
    }
    
    print(f"[INFO] Testing {len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel'])} combinations...")
    
    # GridSearch
    svm = SVC(probability=True, random_state=Config.RANDOM_STATE)
    grid_search = GridSearchCV(
        svm, param_grid, cv=3, scoring='f1_weighted',
        n_jobs=-1, verbose=1
    )
    
    print("[INFO] Running GridSearchCV...")
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\n[RESULT] Best parameters: {grid_search.best_params_}")
    print(f"[RESULT] Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_


def tune_fine_knn(super_class):
    """
    Tune fine-class k-NN hyperparameters
    
    Args:
        super_class: 'fiber' or 'rigid'
    """
    
    print("\n" + "="*70)
    print(f"TUNING {super_class.upper()} FINE-CLASS K-NN HYPERPARAMETERS".center(70))
    print("="*70 + "\n")
    
    # Load appropriate classes
    if super_class == 'fiber':
        class_names = ['paper', 'cardboard']
    else:  # rigid
        class_names = ['plastic', 'metal']
    
    print("[INFO] Loading tuning data...")
    X, y = load_tuning_data(class_names, max_samples_per_class=200)
    
    print(f"[INFO] Loaded {len(X)} samples for tuning")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=Config.RANDOM_STATE, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Parameter grid
    param_grid = {
        'n_neighbors': [5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    print(f"[INFO] Testing {len(param_grid['n_neighbors']) * len(param_grid['weights']) * len(param_grid['metric'])} combinations...")
    
    # GridSearch
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(
        knn, param_grid, cv=3, scoring='f1_weighted',
        n_jobs=-1, verbose=1
    )
    
    print("[INFO] Running GridSearchCV...")
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\n[RESULT] Best parameters: {grid_search.best_params_}")
    print(f"[RESULT] Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_


if __name__ == "__main__":
    print("="*70)
    print("HYPERPARAMETER TUNING".center(70))
    print("="*70)
    
    # Tune all models
    super_svm_params = tune_super_svm()
    super_knn_params = tune_super_knn()
    fiber_svm_params = tune_fine_svm('fiber')
    fiber_knn_params = tune_fine_knn('fiber')
    rigid_svm_params = tune_fine_svm('rigid')
    rigid_knn_params = tune_fine_knn('rigid')
    
    # Print summary
    print("\n" + "="*70)
    print("TUNING COMPLETE - BEST PARAMETERS".center(70))
    print("="*70)
    print(f"\nSuper SVM:  {super_svm_params}")
    print(f"Super k-NN: {super_knn_params}")
    print(f"Fiber SVM:  {fiber_svm_params}")
    print(f"Fiber k-NN: {fiber_knn_params}")
    print(f"Rigid SVM:  {rigid_svm_params}")
    print(f"Rigid k-NN: {rigid_knn_params}")
    print("="*70 + "\n")
