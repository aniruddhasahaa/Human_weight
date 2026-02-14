"""
Image Data Loader for CNN Training
Functions to load and prepare image data for weight estimation.
"""

import numpy as np
import pandas as pd
import os

from config_training import FOLDS_DIR

# =====================================================
# IMAGE DATA LOADING FUNCTIONS
# =====================================================

def load_train_val_from_folds_image_only(fold_indices):
    """
    Load and prepare training + validation data for IMAGE-ONLY regression.
    
    Args:
        fold_indices: List of fold numbers to load (e.g., [1, 2, 3, 4])
    
    Returns:
        Tuple of ((X_train, y_train), (X_val, y_val))
    """
    
    train_imgs, train_y = [], []
    val_imgs, val_y = [], []
    
    print(f"\n[LOAD] Loading image data from folds: {fold_indices}")
    
    for i in fold_indices:
        fold_dir = os.path.join(FOLDS_DIR, f"fold-{i}")
        print(f"  → Processing Fold-{i}")
        
        # ------------------------
        # Load Images
        # ------------------------
        tr_imgs = np.load(os.path.join(fold_dir, "train_images.npy"))
        v_imgs = np.load(os.path.join(fold_dir, "val_images.npy"))
        
        # Convert to uint8 to save memory
        tr_imgs = tr_imgs.astype(np.uint8, copy=False)
        v_imgs = v_imgs.astype(np.uint8, copy=False)
        
        # ------------------------
        # Load Labels
        # ------------------------
        tr_df = pd.read_csv(os.path.join(fold_dir, "train_metadata.csv"))
        v_df = pd.read_csv(os.path.join(fold_dir, "val_metadata.csv"))
        
        # Verify image-label consistency
        assert len(tr_imgs) == len(tr_df), f"❌ Train image–label mismatch in fold-{i}!"
        assert len(v_imgs) == len(v_df), f"❌ Val image–label mismatch in fold-{i}!"
        
        # Extract weight labels
        tr_y = tr_df["weight"].values.astype(np.float32)
        v_y = v_df["weight"].values.astype(np.float32)
        
        # Handle NaN and inf values
        tr_y = np.nan_to_num(tr_y, nan=0.0, posinf=0.0, neginf=0.0)
        v_y = np.nan_to_num(v_y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Append to lists
        train_imgs.append(tr_imgs)
        train_y.append(tr_y)
        
        val_imgs.append(v_imgs)
        val_y.append(v_y)
        
        print(f"    ✓ Train: {len(tr_imgs)} images | Val: {len(v_imgs)} images")
    
    # ------------------------
    # Merge All Folds
    # ------------------------
    print("\n[MERGE] Combining all folds...")
    
    X_train = np.concatenate(train_imgs, axis=0)
    y_train = np.concatenate(train_y, axis=0)
    
    X_val = np.concatenate(val_imgs, axis=0)
    y_val = np.concatenate(val_y, axis=0)
    
    print(f"✓ Merged Train: {len(X_train)} samples")
    print(f"✓ Merged Val: {len(X_val)} samples")
    
    # ------------------------
    # Final Validation Checks
    # ------------------------
    assert X_train.shape[0] == y_train.shape[0], "Train image-label count mismatch!"
    assert X_val.shape[0] == y_val.shape[0], "Val image-label count mismatch!"
    
    print("\n[VALID] Data validation passed")
    print(f"[SHAPE] Train images: {X_train.shape}")
    print(f"[SHAPE] Val images: {X_val.shape}")
    print(f"[RANGE] Train weight: [{y_train.min():.1f}, {y_train.max():.1f}] lb")
    print(f"[RANGE] Val weight: [{y_val.min():.1f}, {y_val.max():.1f}] lb")
    
    # Check for invalid weights
    train_invalid = (y_train <= 0).sum()
    val_invalid = (y_val <= 0).sum()
    
    if train_invalid > 0:
        print(f"⚠️ Warning: {train_invalid} invalid weights in training set")
    if val_invalid > 0:
        print(f"⚠️ Warning: {val_invalid} invalid weights in validation set")
    
    return (X_train, y_train), (X_val, y_val)


def load_test_fold_image_only(fold_index):
    """
    Load full test fold (train + val combined) for IMAGE-ONLY regression.
    
    Args:
        fold_index: Fold number to load as test set
    
    Returns:
        Tuple of (X_test, y_test, filenames)
    """
    
    print(f"\n[TEST] Loading test data from fold-{fold_index}")
    
    fold_dir = os.path.join(FOLDS_DIR, f"fold-{fold_index}")
    
    # ------------------------
    # Load Images
    # ------------------------
    tr_imgs = np.load(os.path.join(fold_dir, "train_images.npy")).astype(np.uint8)
    v_imgs = np.load(os.path.join(fold_dir, "val_images.npy")).astype(np.uint8)
    
    # ------------------------
    # Load Labels and Filenames
    # ------------------------
    tr_df = pd.read_csv(os.path.join(fold_dir, "train_metadata.csv"))
    v_df = pd.read_csv(os.path.join(fold_dir, "val_metadata.csv"))
    
    # Combine train and val for full test set
    test_df = pd.concat([tr_df, v_df], ignore_index=True)
    
    y_test = test_df["weight"].values.astype(np.float32)
    filenames = test_df["filename"].tolist()
    
    # Handle NaN and inf values
    y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Combine images
    X_test = np.concatenate([tr_imgs, v_imgs], axis=0)
    
    # Verify consistency
    assert X_test.shape[0] == y_test.shape[0], "Test image-label count mismatch!"
    assert len(filenames) == len(y_test), "Filename-label count mismatch!"
    
    print(f"✓ Test samples loaded: {len(X_test)}")
    print(f"[SHAPE] Test images: {X_test.shape}")
    print(f"[RANGE] Test weight: [{y_test.min():.1f}, {y_test.max():.1f}] lb")
    
    # Check for invalid weights
    test_invalid = (y_test <= 0).sum()
    if test_invalid > 0:
        print(f"⚠️ Warning: {test_invalid} invalid weights in test set")
    
    return X_test, y_test, filenames


# =====================================================
# DATA AUGMENTATION (Optional)
# =====================================================

def create_data_generator(X, y, batch_size=32, augment=False):
    """
    Create data generator with optional augmentation.
    
    Args:
        X: Image data
        y: Labels
        batch_size: Batch size
        augment: Whether to apply data augmentation
    
    Returns:
        TensorFlow data generator
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    if augment:
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
    else:
        datagen = ImageDataGenerator(rescale=1./255)
    
    return datagen.flow(X, y, batch_size=batch_size)


# =====================================================
# DATA PREPROCESSING
# =====================================================

def preprocess_images(X, normalize=True):
    """
    Preprocess images for model input.
    
    Args:
        X: Image array (uint8, 0-255)
        normalize: Whether to normalize to [0, 1]
    
    Returns:
        Preprocessed images
    """
    if normalize:
        # Convert to float32 and normalize to [0, 1]
        return X.astype(np.float32) / 255.0
    return X.astype(np.float32)


def get_sample_weights(y, method='inverse'):
    """
    Calculate sample weights based on target distribution.
    Useful for handling imbalanced weight distributions.
    
    Args:
        y: Target labels
        method: 'inverse' or 'sqrt_inverse'
    
    Returns:
        Sample weights
    """
    # Create weight bins
    bins = np.linspace(y.min(), y.max(), 20)
    bin_indices = np.digitize(y, bins)
    
    # Calculate weights
    unique, counts = np.unique(bin_indices, return_counts=True)
    
    if method == 'inverse':
        weights = 1.0 / counts
    elif method == 'sqrt_inverse':
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Map weights to samples
    sample_weights = np.array([weights[np.where(unique == bin_idx)[0][0]] 
                               for bin_idx in bin_indices])
    
    # Normalize
    sample_weights = sample_weights / sample_weights.sum() * len(y)
    
    return sample_weights


# =====================================================
# EXAMPLE USAGE
# =====================================================

if __name__ == "__main__":
    print("="*60)
    print("IMAGE DATA LOADER - TEST")
    print("="*60)
    
    try:
        # Test loading training data
        train_folds = [1, 2, 3, 4]
        (X_train, y_train), (X_val, y_val) = \
            load_train_val_from_folds_image_only(train_folds)
        
        print("\n[SUCCESS] Training data loaded")
        print(f"  Train: {X_train.shape}, {y_train.shape}")
        print(f"  Val: {X_val.shape}, {y_val.shape}")
        
        # Test loading test data
        test_fold = 5
        X_test, y_test, filenames = load_test_fold_image_only(test_fold)
        
        print("\n[SUCCESS] Test data loaded")
        print(f"  Test: {X_test.shape}, {y_test.shape}")
        print(f"  Filenames: {len(filenames)}")
        
        # Test preprocessing
        print("\n[TEST] Testing preprocessing...")
        X_train_norm = preprocess_images(X_train[:10])
        print(f"  Normalized range: [{X_train_norm.min():.3f}, {X_train_norm.max():.3f}]")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
