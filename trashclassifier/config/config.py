"""
Central configuration file for the trash classifier system.
"""
import os

# ==================== PATHS ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = r"C:\Users\ahmed\Desktop\MACHINE_LEARNING\dataset"
AUGMENTED_PATH = os.path.join(BASE_DIR, "data", "augmented_dataset")
SAVED_MODELS_PATH = os.path.join(BASE_DIR, "saved_models")

# ==================== CLASS DEFINITIONS ====================
# Fine-class definitions (7 classes total)
FINE_CLASS_NAMES = {
    0: "Glass",
    1: "Paper",
    2: "Cardboard",
    3: "Plastic",
    4: "Metal",
    5: "Trash",
    6: "Unknown"
}

FINE_CLASS_IDS = {v: k for k, v in FINE_CLASS_NAMES.items()}

# Super-class definitions (4 classes)
SUPER_CLASS_NAMES = {
    0: "Fiber",        # Paper, Cardboard
    1: "Rigid",        # Plastic, Metal
    2: "Transparent",  # Glass
    3: "Garbage"       # Trash
}

SUPER_CLASS_IDS = {v: k for k, v in SUPER_CLASS_NAMES.items()}

# Mapping: fine-class → super-class
FINE_TO_SUPER = {
    0: 2,  # Glass → Transparent
    1: 0,  # Paper → Fiber
    2: 0,  # Cardboard → Fiber
    3: 1,  # Plastic → Rigid
    4: 1,  # Metal → Rigid
    5: 3,  # Trash → Garbage
    6: -1  # Unknown → None (handled by confidence)
}

# Mapping: super-class → fine-classes
SUPER_TO_FINE = {
    0: [1, 2],   # Fiber → Paper, Cardboard
    1: [3, 4],   # Rigid → Plastic, Metal
    2: [0],      # Transparent → Glass
    3: [5]       # Garbage → Trash
}

# Dataset folder names (from original dataset structure)
DATASET_FOLDERS = {
    0: "glass",
    1: "paper",
    2: "cardboard",
    3: "plastic",
    4: "metal",
    5: "trash"
}

# ==================== DATA AUGMENTATION ====================
TARGET_SAMPLES_PER_CLASS = 700  # Target after augmentation (30%+ increase from ~500)

# Augmentation techniques to apply
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
