"""
Script 6: Add Weight Labels to Fold Metadata
This script loads the weight labels CSV and merges them into all fold metadata files.
"""

import os
import pandas as pd

# =====================================================
# CONFIGURATION
# =====================================================
BASE_DIR = "/content/drive/MyDrive/Human Weight Extimation"
INPUT_DIR = os.path.join(BASE_DIR, "Input Data")
FOLDS_DIR = os.path.join(BASE_DIR, "folds_individual")

# Weight labels CSV path
WEIGHT_CSV = os.path.join(INPUT_DIR, "All_labels_224.csv")

# Number of folds
NUM_FOLDS = 5

# =====================================================
# FUNCTIONS
# =====================================================

def load_weight_labels():
    """Load and validate the weight labels CSV"""
    print("="*60)
    print("LOADING WEIGHT LABELS")
    print("="*60)
    
    print(f"\n[LOAD] Reading weight file from: {WEIGHT_CSV}")
    
    if not os.path.exists(WEIGHT_CSV):
        raise FileNotFoundError(f"Weight CSV not found: {WEIGHT_CSV}")
    
    weight_df = pd.read_csv(WEIGHT_CSV)
    
    print(f"✓ Weight file loaded: {weight_df.shape}")
    
    # Display first few rows
    print("\n[INFO] First 5 rows:")
    print(weight_df.head())
    
    # Verify required columns
    print("\n[VERIFY] Checking required columns...")
    
    if "filename" not in weight_df.columns:
        raise ValueError("ERROR: 'filename' column missing in weight CSV")
    
    if "weight" not in weight_df.columns:
        raise ValueError("ERROR: 'weight' column missing in weight CSV")
    
    print("✓ Required columns present: 'filename', 'weight'")
    
    # Check for missing weights
    missing_weights = weight_df["weight"].isna().sum()
    total_rows = len(weight_df)
    
    print(f"\n[STATS] Weight statistics:")
    print(f"  Total rows: {total_rows}")
    print(f"  Missing weights: {missing_weights}")
    print(f"  Valid weights: {total_rows - missing_weights}")
    
    if missing_weights > 0:
        print(f"  ⚠️ Warning: {missing_weights} rows have missing weights")
    
    # Display weight distribution
    print(f"\n[STATS] Weight distribution:")
    print(weight_df["weight"].describe())
    
    return weight_df


def merge_weights_to_fold(fold_num, weight_df, folds_dir):
    """
    Merge weight column into train and validation metadata for a specific fold.
    
    Args:
        fold_num: Fold number (1-based)
        weight_df: DataFrame containing weights
        folds_dir: Directory containing fold data
    
    Returns:
        Dictionary with merge statistics
    """
    fold_dir = os.path.join(folds_dir, f"fold-{fold_num}")
    
    if not os.path.exists(fold_dir):
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")
    
    # File paths
    train_path = os.path.join(fold_dir, "train_metadata.csv")
    val_path = os.path.join(fold_dir, "val_metadata.csv")
    
    # Load metadata
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Store original counts
    train_orig = len(train_df)
    val_orig = len(val_df)
    
    # Merge weight column
    train_merged = train_df.merge(weight_df[["filename", "weight"]], on="filename", how="left")
    val_merged = val_df.merge(weight_df[["filename", "weight"]], on="filename", how="left")
    
    # Check for missing weights
    train_missing = train_merged["weight"].isna().sum()
    val_missing = val_merged["weight"].isna().sum()
    
    # Save updated files
    train_merged.to_csv(train_path, index=False)
    val_merged.to_csv(val_path, index=False)
    
    return {
        'fold': fold_num,
        'train_total': train_orig,
        'train_missing': train_missing,
        'train_valid': train_orig - train_missing,
        'val_total': val_orig,
        'val_missing': val_missing,
        'val_valid': val_orig - val_missing
    }


def verify_fold_weights(fold_num, folds_dir):
    """
    Verify weight integration for a specific fold.
    
    Args:
        fold_num: Fold number
        folds_dir: Directory containing folds
    """
    fold_dir = os.path.join(folds_dir, f"fold-{fold_num}")
    
    train_path = os.path.join(fold_dir, "train_metadata.csv")
    val_path = os.path.join(fold_dir, "val_metadata.csv")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    print(f"\n  [VERIFY] Checking weight column...")
    
    # Check if weight column exists
    if "weight" not in train_df.columns:
        print(f"  ⚠️ ERROR: 'weight' column not found in train_metadata.csv")
    else:
        print(f"  ✓ 'weight' column exists in train_metadata.csv")
    
    if "weight" not in val_df.columns:
        print(f"  ⚠️ ERROR: 'weight' column not found in val_metadata.csv")
    else:
        print(f"  ✓ 'weight' column exists in val_metadata.csv")
    
    # Display weight statistics
    if "weight" in train_df.columns:
        print(f"\n  [STATS] Train weight statistics:")
        print(f"    Min: {train_df['weight'].min():.2f}")
        print(f"    Max: {train_df['weight'].max():.2f}")
        print(f"    Mean: {train_df['weight'].mean():.2f}")
        print(f"    Median: {train_df['weight'].median():.2f}")
    
    if "weight" in val_df.columns:
        print(f"\n  [STATS] Val weight statistics:")
        print(f"    Min: {val_df['weight'].min():.2f}")
        print(f"    Max: {val_df['weight'].max():.2f}")
        print(f"    Mean: {val_df['weight'].mean():.2f}")
        print(f"    Median: {val_df['weight'].median():.2f}")


# =====================================================
# MAIN EXECUTION
# =====================================================

def main():
    """Main function to add weights to all folds"""
    
    print("="*60)
    print("ADD WEIGHT LABELS TO FOLD METADATA")
    print("="*60)
    
    # Load weight labels
    weight_df = load_weight_labels()
    
    # Verify folds directory exists
    if not os.path.exists(FOLDS_DIR):
        raise FileNotFoundError(f"Folds directory not found: {FOLDS_DIR}")
    
    print(f"\n✓ Folds directory: {FOLDS_DIR}")
    
    # Merge weights for each fold
    print("\n" + "="*60)
    print("MERGING WEIGHTS INTO FOLD METADATA")
    print("="*60)
    
    merge_stats = []
    
    for fold_num in range(1, NUM_FOLDS + 1):
        print(f"\n{'='*60}")
        print(f"  FOLD-{fold_num}")
        print(f"{'='*60}")
        
        # Merge weights
        stats = merge_weights_to_fold(fold_num, weight_df, FOLDS_DIR)
        merge_stats.append(stats)
        
        print(f"\n  [MERGE] Merging weight column...")
        print(f"  ✓ Train metadata updated")
        print(f"  ✓ Val metadata updated")
        
        print(f"\n  [STATS] Fold-{fold_num} Summary:")
        print(f"    Train: {stats['train_total']} rows, {stats['train_missing']} missing weights")
        print(f"    Val:   {stats['val_total']} rows, {stats['val_missing']} missing weights")
        
        # Verify integration
        verify_fold_weights(fold_num, FOLDS_DIR)
    
    # Print summary table
    print("\n" + "="*60)
    print("WEIGHT INTEGRATION SUMMARY")
    print("="*60)
    
    stats_df = pd.DataFrame(merge_stats)
    
    print("\nTrain Set:")
    print(stats_df[['fold', 'train_total', 'train_valid', 'train_missing']].to_string(index=False))
    
    print("\nValidation Set:")
    print(stats_df[['fold', 'val_total', 'val_valid', 'val_missing']].to_string(index=False))
    
    # Calculate totals
    total_train = stats_df['train_total'].sum()
    total_train_missing = stats_df['train_missing'].sum()
    total_val = stats_df['val_total'].sum()
    total_val_missing = stats_df['val_missing'].sum()
    
    print("\nOverall Summary:")
    print(f"  Total train samples: {total_train}")
    print(f"  Train missing weights: {total_train_missing} ({100*total_train_missing/total_train:.2f}%)")
    print(f"  Total val samples: {total_val}")
    print(f"  Val missing weights: {total_val_missing} ({100*total_val_missing/total_val:.2f}%)")
    
    # Check if all weights were successfully merged
    if total_train_missing == 0 and total_val_missing == 0:
        print("\n✅ ALL FOLDS UPDATED WITH WEIGHT COLUMN!")
        print("✓ No missing weights")
    else:
        print("\n⚠️ FOLDS UPDATED WITH WARNINGS!")
        print(f"⚠️ {total_train_missing + total_val_missing} total missing weights")
        print("   Please verify the weight CSV contains all filenames")
    
    print("\n" + "="*60)
    print("UPDATED FILES:")
    print("="*60)
    print("\nEach fold now contains:")
    print("  fold-{N}/")
    print("    ├── train_metadata.csv  (with 'weight' column)")
    print("    ├── val_metadata.csv    (with 'weight' column)")
    print("    ├── train_images.npy")
    print("    └── val_images.npy")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
