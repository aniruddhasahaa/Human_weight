"""
Script 5: Create 5-Fold Cross-Validation Splits
This script splits the preprocessed data into 5 folds for cross-validation.
Each fold is split into train/validation sets with fixed random seed for reproducibility.
"""

import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split

# =====================================================
# CONFIGURATION
# =====================================================
BASE_DIR = "/content/drive/MyDrive/Human Weight Extimation"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_dataset")
PROCESSED_IMG_PATH = os.path.join(PROCESSED_DIR, "processed_images.npy")
PROCESSED_CSV_PATH = os.path.join(PROCESSED_DIR, "processed_metadata.csv")

# Output directory for folds
SAVE_FOLD_DIR = os.path.join(BASE_DIR, "folds_individual")

# Cross-validation settings
NUM_FOLDS = 5
VAL_SPLIT = 0.10  # 10% of each fold for validation

# Reproducibility seed
SEED = 42

# =====================================================
# REPRODUCIBILITY SETUP
# =====================================================

def set_random_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"✓ Random seeds set to {seed} for reproducibility")


# =====================================================
# FUNCTIONS
# =====================================================

def load_processed_data():
    """Load preprocessed images and metadata"""
    print("\n" + "="*60)
    print("LOADING PREPROCESSED DATA")
    print("="*60)
    
    print(f"\n[LOAD] Loading images from: {PROCESSED_IMG_PATH}")
    images = np.load(PROCESSED_IMG_PATH)
    print(f"✓ Images loaded: {images.shape}")
    
    print(f"\n[LOAD] Loading metadata from: {PROCESSED_CSV_PATH}")
    df = pd.read_csv(PROCESSED_CSV_PATH)
    print(f"✓ Metadata loaded: {df.shape}")
    
    filenames = df["filename"].tolist()
    N = len(filenames)
    
    print(f"\n✓ Total samples: {N}")
    
    return images, df, filenames


def create_fold_splits(N, num_folds=5, seed=42):
    """
    Create fold indices for cross-validation.
    
    Args:
        N: Total number of samples
        num_folds: Number of folds to create
        seed: Random seed for reproducibility
    
    Returns:
        List of arrays containing indices for each fold
    """
    print("\n" + "="*60)
    print(f"CREATING {num_folds}-FOLD SPLITS")
    print("="*60)
    
    # Create and shuffle indices
    indices = np.arange(N)
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    print(f"\n✓ Shuffled {N} indices with seed={seed}")
    
    # Calculate fold sizes
    fold_sizes = [N // num_folds] * num_folds
    remainder = N % num_folds
    
    # Distribute remainder across first folds
    for i in range(remainder):
        fold_sizes[i] += 1
    
    print(f"\nFold sizes: {fold_sizes}")
    
    # Split indices into folds
    folds = []
    start = 0
    for i, size in enumerate(fold_sizes, 1):
        fold_indices = indices[start:start+size]
        folds.append(fold_indices)
        print(f"  Fold {i}: {len(fold_indices)} samples")
        start += size
    
    return folds


def create_train_val_split(fold_idx, images, df, filenames, val_split=0.10, seed=42):
    """
    Split a fold into training and validation sets.
    
    Args:
        fold_idx: Indices for this fold
        images: Full image array
        df: Full metadata dataframe
        filenames: List of all filenames
        val_split: Fraction of data to use for validation
        seed: Random seed
    
    Returns:
        Dictionary containing train/val images and metadata
    """
    # Extract fold data
    fold_images = images[fold_idx]
    fold_files = [filenames[i] for i in fold_idx]
    fold_df = df[df["filename"].isin(fold_files)].reset_index(drop=True)
    
    # Split into train/val
    train_idx, val_idx = train_test_split(
        np.arange(len(fold_df)),
        test_size=val_split,
        random_state=seed,
        shuffle=True
    )
    
    # Extract train/val data
    train_images = fold_images[train_idx]
    val_images = fold_images[val_idx]
    
    train_df = fold_df.iloc[train_idx].reset_index(drop=True)
    val_df = fold_df.iloc[val_idx].reset_index(drop=True)
    
    return {
        'fold_size': len(fold_df),
        'train_images': train_images,
        'val_images': val_images,
        'train_df': train_df,
        'val_df': val_df
    }


def save_fold_data(fold_num, fold_data, save_dir):
    """
    Save fold data to disk.
    
    Args:
        fold_num: Fold number (1-based)
        fold_data: Dictionary with train/val data
        save_dir: Base directory to save folds
    """
    # Create fold directory
    fold_dir = os.path.join(save_dir, f"fold-{fold_num}")
    os.makedirs(fold_dir, exist_ok=True)
    
    # Save images
    np.save(os.path.join(fold_dir, "train_images.npy"), fold_data['train_images'])
    np.save(os.path.join(fold_dir, "val_images.npy"), fold_data['val_images'])
    
    # Save metadata
    fold_data['train_df'].to_csv(os.path.join(fold_dir, "train_metadata.csv"), index=False)
    fold_data['val_df'].to_csv(os.path.join(fold_dir, "val_metadata.csv"), index=False)
    
    return fold_dir


def verify_fold_split(fold_num, fold_data):
    """Print verification statistics for a fold"""
    fold_size = fold_data['fold_size']
    train_size = len(fold_data['train_df'])
    val_size = len(fold_data['val_df'])
    
    print(f"\nFold-{fold_num} Statistics:")
    print(f"  Total: {fold_size}")
    print(f"  Train: {train_size} ({100*train_size/fold_size:.1f}%)")
    print(f"  Val:   {val_size} ({100*val_size/fold_size:.1f}%)")
    print(f"  Train images: {fold_data['train_images'].shape}")
    print(f"  Val images:   {fold_data['val_images'].shape}")


# =====================================================
# MAIN EXECUTION
# =====================================================

def main():
    """Main function to create all folds"""
    
    print("="*60)
    print("5-FOLD CROSS-VALIDATION SPLIT CREATION")
    print("="*60)
    
    # Set random seeds
    set_random_seeds(SEED)
    
    # Create output directory
    os.makedirs(SAVE_FOLD_DIR, exist_ok=True)
    print(f"\n✓ Output directory: {SAVE_FOLD_DIR}")
    
    # Load data
    images, df, filenames = load_processed_data()
    N = len(filenames)
    
    # Create fold splits
    folds = create_fold_splits(N, NUM_FOLDS, SEED)
    
    # Process each fold
    print("\n" + "="*60)
    print("CREATING INDIVIDUAL FOLDS")
    print("="*60)
    
    fold_stats = []
    
    for fold_num, fold_idx in enumerate(folds, start=1):
        print(f"\n{'='*60}")
        print(f"  PROCESSING FOLD-{fold_num}")
        print(f"{'='*60}")
        
        # Create train/val split
        fold_data = create_train_val_split(
            fold_idx, images, df, filenames, 
            val_split=VAL_SPLIT, 
            seed=SEED
        )
        
        # Save fold data
        fold_dir = save_fold_data(fold_num, fold_data, SAVE_FOLD_DIR)
        print(f"\n✓ Saved to: {fold_dir}")
        
        # Verify and display statistics
        verify_fold_split(fold_num, fold_data)
        
        # Store stats for summary
        fold_stats.append({
            'fold': fold_num,
            'total': fold_data['fold_size'],
            'train': len(fold_data['train_df']),
            'val': len(fold_data['val_df'])
        })
    
    # Print final summary
    print("\n" + "="*60)
    print("FOLD CREATION SUMMARY")
    print("="*60)
    
    stats_df = pd.DataFrame(fold_stats)
    print(f"\n{stats_df.to_string(index=False)}")
    
    print(f"\nTotal samples: {N}")
    print(f"Total train samples: {stats_df['train'].sum()}")
    print(f"Total val samples: {stats_df['val'].sum()}")
    print(f"\nAverage fold size: {stats_df['total'].mean():.1f}")
    print(f"Average train size: {stats_df['train'].mean():.1f}")
    print(f"Average val size: {stats_df['val'].mean():.1f}")
    
    print("\n✅ ALL 5 INDIVIDUAL FOLDS CREATED SUCCESSFULLY!")
    print(f"\nOutput location: {SAVE_FOLD_DIR}")
    print("\nFold structure:")
    print("  fold-1/")
    print("    ├── train_images.npy")
    print("    ├── val_images.npy")
    print("    ├── train_metadata.csv")
    print("    └── val_metadata.csv")
    print("  fold-2/")
    print("  ...")
    print("  fold-5/")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
