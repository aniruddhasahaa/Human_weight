"""
Model Training Configuration
Settings and parameters for weight estimation model training.
"""

import os
import tensorflow as tf
import numpy as np
import random

# =====================================================
# BASE PATHS
# =====================================================
BASE_DIR = "/content/drive/MyDrive/Human Weight Extimation"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_dataset")
FOLDS_DIR = os.path.join(BASE_DIR, "folds_individual")
OUTPUT_DIR = os.path.join(BASE_DIR, "fusion_results_v1")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# IMAGE SETTINGS
# =====================================================
IMG_SHAPE = (224, 224, 3)

# =====================================================
# METADATA FEATURE COLUMNS
# =====================================================
# Option 1: Original features only
META_COLS_ORIGINAL = [
    "shoulder_width_px",
    "waist_width_px",
    "hip_width_px",
    "thigh_width_px",
    "W_by_Hip",
    "W_by_Shoulder",
    "W_by_Thigh",
    "Hip_by_Shoulder",
    "Area (px)"
]

# Option 2: Original + derived features
META_COLS_WITH_DERIVED = [
    # Original features
    "shoulder_width_px",
    "waist_width_px",
    "hip_width_px",
    "thigh_width_px",
    "Area (px)",
    
    # Derived features
    "height_est_px",
    "log_area",
    "waist_to_height",
    "hip_to_height",
    "thigh_to_height",
    "shoulder_to_height",
    "area_over_height2",
    "area_over_mean_width",
    "waist_to_hip",
    "thigh_to_hip"
]

# Option 3: Original + selected derived features
META_COLS_SELECTED = [
    # Original features
    "shoulder_width_px",
    "waist_width_px",
    "hip_width_px",
    "thigh_width_px",
    "W_by_Hip",
    "W_by_Shoulder",
    "W_by_Thigh",
    "Hip_by_Shoulder",
    "Area (px)",
    
    # Selected derived features
    "height_est_px",
    "log_area",
    "waist_to_height",
    "hip_to_height",
    "area_over_mean_width"
]

# Active configuration (change this to switch feature sets)
META_COLS = META_COLS_ORIGINAL

# =====================================================
# TRAINING HYPERPARAMETERS
# =====================================================

# Deep Learning Model (for future use)
BATCH_SIZE = 8
EPOCHS = 75
PATIENCE = 15
LEARNING_RATE = 1e-5

# XGBoost Hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 800,
    'max_depth': 4,
    'learning_rate': 0.03,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'reg_alpha': 0.5,      # L1 regularization
    'reg_lambda': 1.5,     # L2 regularization
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': -1
}

# =====================================================
# CROSS-VALIDATION SETTINGS
# =====================================================
NUM_FOLDS = 5

# 5-Fold CV Experiment Configurations
CV_EXPERIMENTS = [
    {"train": [1, 2, 3, 4], "test": 5},
    {"train": [1, 2, 3, 5], "test": 4},
    {"train": [1, 2, 4, 5], "test": 3},
    {"train": [1, 3, 4, 5], "test": 2},
    {"train": [2, 3, 4, 5], "test": 1},
]

# =====================================================
# REPRODUCIBILITY
# =====================================================
SEED = 42

def set_random_seeds():
    """Set all random seeds for reproducibility"""
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

# =====================================================
# GPU CONFIGURATION
# =====================================================
def configure_gpu():
    """Configure GPU memory growth to prevent OOM errors"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"⚠️ GPU configuration error: {e}")
    else:
        print("ℹ️ No GPU detected, using CPU")

# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def print_config():
    """Print current configuration settings"""
    print("="*60)
    print("MODEL TRAINING CONFIGURATION")
    print("="*60)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Folds Directory: {FOLDS_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"\nImage Shape: {IMG_SHAPE}")
    print(f"Metadata Features: {len(META_COLS)}")
    print(f"Features: {META_COLS}")
    print(f"\nXGBoost Parameters:")
    for key, value in XGBOOST_PARAMS.items():
        print(f"  {key}: {value}")
    print(f"\nCross-Validation: {NUM_FOLDS}-Fold")
    print(f"Random Seed: {SEED}")
    print("="*60)

def verify_paths():
    """Verify that all required directories exist"""
    paths = {
        "Folds Directory": FOLDS_DIR,
        "Output Directory": OUTPUT_DIR
    }
    
    all_exist = True
    for name, path in paths.items():
        if os.path.exists(path):
            print(f"✓ {name}: {path}")
        else:
            print(f"❌ {name} NOT FOUND: {path}")
            all_exist = False
    
    return all_exist

# =====================================================
# INITIALIZATION
# =====================================================

if __name__ == "__main__":
    print_config()
    print("\nVerifying paths...")
    if verify_paths():
        print("\n✅ Configuration validated successfully")
    else:
        print("\n⚠️ Some paths are missing. Please check configuration.")
    
    # Set random seeds
    set_random_seeds()
    print("\n✓ Random seeds initialized")
    
    # Configure GPU
    configure_gpu()
