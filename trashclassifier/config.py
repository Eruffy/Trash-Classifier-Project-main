"""
Central configuration file for the trash classifier system.
"""
import os


class Config:
    """Central configuration class"""
    
    # ==================== PATHS ====================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = r"C:\Users\ahmed\Desktop\MACHINE_LEARNING\dataset"
    AUGMENTED_PATH = os.path.join(BASE_DIR, "data", "augmented_dataset")
    SAVED_MODELS_PATH = os.path.join(BASE_DIR, "saved_models")
    
    # ==================== CLASS DEFINITIONS ====================
    # Fine-class definitions (7 classes total)
    CLASS_NAMES = {
        'glass': 0, 'paper': 1, 'cardboard': 2,
        'plastic': 3, 'metal': 4, 'trash': 5
    }
    
    # ==================== DATA AUGMENTATION ====================
    TARGET_SAMPLES_PER_CLASS = 700  # Target after augmentation (30%+ increase from ~500)
    
    # ==================== FEATURE EXTRACTION ====================
    IMAGE_SIZE = 224  # Standard input size
    
    # ==================== MODEL TRAINING ====================
    TEST_SIZE = 0.2  # 20% for validation
    RANDOM_STATE = 42  # For reproducibility
    
    # ==================== INFERENCE ====================
    # Confidence thresholds for Unknown detection
    SUPER_THRESHOLD = 0.30  # Lower threshold for super-class (more lenient for real-world images)
    FINE_THRESHOLD = 0.50   # Higher threshold for fine-class (reduced for better coverage)
    UNKNOWN_THRESHOLD = 0.40  # Overall unknown threshold (more permissive)


# Legacy constants (kept for compatibility)
AUGMENTATION_CONFIG = {
    "rotation_range": 30,           # Rotate ±30 degrees
    "horizontal_flip": True,        # Random horizontal flip
    "vertical_flip": False,         # No vertical flip (unnatural for trash)
    "brightness_range": (0.7, 1.3), # Brightness variation ±30%
    "zoom_range": 0.2,              # Zoom in/out ±20%
    "shear_range": 0.1,             # Slight shearing
    "width_shift_range": 0.1,       # Horizontal shift ±10%
    "height_shift_range": 0.1,      # Vertical shift ±10%
    "channel_shift_range": 20,      # Color jitter ±20
    "fill_mode": "reflect"          # Fill mode for transformations
}

# Additional augmentation for problematic classes
BOOST_AUGMENTATION_FOR = ["glass"]  # Apply more augmentation to glass

# ==================== FEATURE EXTRACTION ====================
IMAGE_SIZE = (224, 224)  # Standard input size

# HOG parameters
HOG_CONFIG = {
    "orientations": 9,
    "pixels_per_cell": (16, 16),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "visualize": False,
    "feature_vector": True,
    "channel_axis": -1
}

# LBP parameters
LBP_CONFIG = {
    "P": 8,      # Number of circularly symmetric neighbor points
    "R": 1,      # Radius of circle
    "method": "uniform"
}

# HSV histogram parameters
HSV_BINS = 50  # Bins per HSV channel (50 * 3 = 150 features)

# ==================== MODEL TRAINING ====================
VALIDATION_SPLIT = 0.2  # 20% for validation
RANDOM_STATE = 42       # For reproducibility

# SVM hyperparameter search space
SVM_PARAM_GRID = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "kernel": ["rbf", "poly", "sigmoid"]
}

# KNN hyperparameter search space
KNN_PARAM_GRID = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "minkowski"]
}

# GridSearchCV settings
GRID_SEARCH_CV = 5  # 5-fold cross-validation
GRID_SEARCH_JOBS = -1  # Use all CPU cores

# Class weight strategy for imbalanced classes
CLASS_WEIGHT = "balanced"  # Auto-balance class weights

# ==================== INFERENCE ====================
# Confidence thresholds for Unknown detection
SUPER_CONFIDENCE_THRESHOLD = 0.40  # Lower threshold for super-class (more lenient)
FINE_CONFIDENCE_THRESHOLD = 0.60   # Higher threshold for fine-class (more strict)

# Ensemble voting strategy
ENSEMBLE_METHOD = "soft"  # "hard" or "soft" voting
ENSEMBLE_WEIGHTS = {
    "svm": 0.6,  # SVM gets 60% weight (typically more accurate)
    "knn": 0.4   # KNN gets 40% weight
}

# ==================== EVALUATION ====================
ACCURACY_TARGET = 0.90  # Target validation accuracy (90%+)

# ==================== CAMERA APP ====================
CAMERA_FPS_TARGET = 10  # Target frames per second
CAMERA_RESOLUTION = (640, 480)  # Camera capture resolution
CAMERA_DISPLAY_SIZE = (800, 600)  # Display window size

# Display colors (BGR format for OpenCV)
COLOR_SCHEME = {
    "svm": (0, 255, 0),      # Green for SVM predictions
    "knn": (255, 0, 0),      # Blue for KNN predictions
    "ensemble": (0, 255, 255),  # Yellow for ensemble
    "unknown": (0, 0, 255),  # Red for unknown
    "text": (255, 255, 255), # White for text
    "background": (50, 50, 50)  # Dark gray background
}

# ==================== LOGGING ====================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
VERBOSE_TRAINING = True  # Show detailed training progress
