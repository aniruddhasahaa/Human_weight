"""
Data Loading Utilities
Functions to load and prepare data for model training.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import configuration
from config_training import FOLDS_DIR, META_COLS

# =====================================================
# METADATA LOADING FUNCTIONS
# =====================================================

def load_meta_train_val_from_folds(train_folds):
    """
    Load and combine metadata from multiple training folds.
    Split into train and validation sets.
    
    Args:
        train_folds: List of fold numbers to use for training (e.g., [1, 2, 3, 4])
    
    Returns:
        Tuple of ((X_train, y_train), (X_val, y_val), scaler)
    """
    
    print(f"\n[LOAD] Loading metadata from folds: {train_folds}")
    
    all_train_dfs = []
    all_val_dfs = []
    
    # Load data from each fold
    for fold_num in train_folds:
        fold_dir = os.path.join(FOLDS_DIR, f"fold-{fold_num}")
        
        train_csv = os.path.join(fold_dir, "train_metadata.csv")
        val_csv = os.path.join(fold_dir, "val_metadata.csv")
        
        if not os.path.exists(train_csv):
            raise FileNotFoundError(f"Train metadata not found: {train_csv}")
        if not os.path.exists(val_csv):
            raise FileNotFoundError(f"Val metadata not found: {val_csv}")
        
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        
        all_train_dfs.append(train_df)
        all_val_dfs.append(val_df)
        
        print(f"  Fold-{fold_num}: Train={len(train_df)}, Val={len(val_df)}")
    
    # Combine all folds
    combined_train = pd.concat(all_train_dfs, ignore_index=True)
    combined_val = pd.concat(all_val_dfs, ignore_index=True)
    
    print(f"\n[COMBINED] Total Train={len(combined_train)}, Total Val={len(combined_val)}")
    
    # Extract features and labels
    X_train = combined_train[META_COLS].values
    y_train = combined_train["weight"].values
    
    X_val = combined_val[META_COLS].values
    y_val = combined_val["weight"].values
    
    # Check for missing values
    train_missing = np.isnan(X_train).sum()
    val_missing = np.isnan(X_val).sum()
    
    if train_missing > 0:
        print(f"⚠️ Warning: {train_missing} missing values in training features")
    if val_missing > 0:
        print(f"⚠️ Warning: {val_missing} missing values in validation features")
    
    # Standardize features
    print("\n[SCALE] Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    print(f"✓ Features scaled")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Val shape: {X_val.shape}")
    
    return (X_train, y_train), (X_val, y_val), scaler


def load_meta_test_fold(test_fold, scaler):
    """
    Load metadata from a single test fold.
    
    Args:
        test_fold: Fold number to use for testing
        scaler: Fitted StandardScaler from training data
    
    Returns:
        Tuple of (X_test, y_test, filenames)
    """
    
    print(f"\n[LOAD] Loading test data from fold-{test_fold}")
    
    fold_dir = os.path.join(FOLDS_DIR, f"fold-{test_fold}")
    
    # Use validation split as test data
    test_csv = os.path.join(fold_dir, "val_metadata.csv")
    
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test metadata not found: {test_csv}")
    
    test_df = pd.read_csv(test_csv)
    print(f"  Test samples: {len(test_df)}")
    
    # Extract features, labels, and filenames
    X_test = test_df[META_COLS].values
    y_test = test_df["weight"].values
    filenames = test_df["filename"].values
    
    # Check for missing values
    test_missing = np.isnan(X_test).sum()
    if test_missing > 0:
        print(f"⚠️ Warning: {test_missing} missing values in test features")
    
    # Standardize using training scaler
    X_test = scaler.transform(X_test)
    
    print(f"✓ Test data loaded and scaled")
    print(f"  Test shape: {X_test.shape}")
    
    return X_test, y_test, filenames


# =====================================================
# IMAGE LOADING FUNCTIONS (for future use)
# =====================================================

def load_images_from_fold(fold_num, split='train'):
    """
    Load images from a specific fold.
    
    Args:
        fold_num: Fold number (1-5)
        split: 'train' or 'val'
    
    Returns:
        numpy array of images
    """
    fold_dir = os.path.join(FOLDS_DIR, f"fold-{fold_num}")
    img_path = os.path.join(fold_dir, f"{split}_images.npy")
    
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    
    images = np.load(img_path)
    print(f"[LOAD] Loaded {split} images from fold-{fold_num}: {images.shape}")
    
    return images


def load_images_train_val_from_folds(train_folds):
    """
    Load and combine images from multiple training folds.
    
    Args:
        train_folds: List of fold numbers to use for training
    
    Returns:
        Tuple of (train_images, val_images)
    """
    print(f"\n[LOAD] Loading images from folds: {train_folds}")
    
    all_train_images = []
    all_val_images = []
    
    for fold_num in train_folds:
        train_imgs = load_images_from_fold(fold_num, 'train')
        val_imgs = load_images_from_fold(fold_num, 'val')
        
        all_train_images.append(train_imgs)
        all_val_images.append(val_imgs)
    
    # Combine all folds
    combined_train = np.concatenate(all_train_images, axis=0)
    combined_val = np.concatenate(all_val_images, axis=0)
    
    print(f"\n[COMBINED] Train images: {combined_train.shape}")
    print(f"[COMBINED] Val images: {combined_val.shape}")
    
    return combined_train, combined_val


def load_images_test_fold(test_fold):
    """
    Load images from a test fold.
    
    Args:
        test_fold: Fold number to use for testing
    
    Returns:
        numpy array of test images
    """
    return load_images_from_fold(test_fold, 'val')


# =====================================================
# DATA VERIFICATION
# =====================================================

def verify_data_integrity(X, y, dataset_name="Dataset"):
    """
    Verify data integrity - check for NaN, inf, and shape consistency.
    
    Args:
        X: Feature matrix
        y: Target labels
        dataset_name: Name for logging
    
    Returns:
        bool: True if data is valid
    """
    print(f"\n[VERIFY] Checking {dataset_name} integrity...")
    
    issues = []
    
    # Check shapes match
    if len(X) != len(y):
        issues.append(f"Shape mismatch: X={len(X)}, y={len(y)}")
    
    # Check for NaN
    nan_count_x = np.isnan(X).sum()
    nan_count_y = np.isnan(y).sum()
    
    if nan_count_x > 0:
        issues.append(f"Features contain {nan_count_x} NaN values")
    if nan_count_y > 0:
        issues.append(f"Labels contain {nan_count_y} NaN values")
    
    # Check for inf
    inf_count_x = np.isinf(X).sum()
    inf_count_y = np.isinf(y).sum()
    
    if inf_count_x > 0:
        issues.append(f"Features contain {inf_count_x} inf values")
    if inf_count_y > 0:
        issues.append(f"Labels contain {inf_count_y} inf values")
    
    # Report results
    if issues:
        print("❌ Data integrity issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(f"✓ {dataset_name} integrity verified")
        print(f"  Shape: X={X.shape}, y={y.shape}")
        print(f"  Label range: [{y.min():.2f}, {y.max():.2f}]")
        return True


# =====================================================
# EXAMPLE USAGE
# =====================================================

if __name__ == "__main__":
    print("="*60)
    print("DATA LOADING UTILITIES - TEST")
    print("="*60)
    
    # Test metadata loading
    try:
        train_folds = [1, 2, 3, 4]
        test_fold = 5
        
        print("\n[TEST] Loading metadata for training...")
        (X_train, y_train), (X_val, y_val), scaler = \
            load_meta_train_val_from_folds(train_folds)
        
        print("\n[TEST] Loading metadata for testing...")
        X_test, y_test, filenames = load_meta_test_fold(test_fold, scaler)
        
        # Verify data
        verify_data_integrity(X_train, y_train, "Training")
        verify_data_integrity(X_val, y_val, "Validation")
        verify_data_integrity(X_test, y_test, "Test")
        
        print("\n✅ Data loading test completed successfully")
        
    except Exception as e:
        print(f"\n❌ Data loading test failed: {e}")
