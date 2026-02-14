"""
Script 7: Complete Fold Verification
This script performs comprehensive verification of all folds including:
- Weight label presence and validity
- Data consistency checks
- Statistical summaries
"""

import os
import pandas as pd
import numpy as np

# =====================================================
# CONFIGURATION
# =====================================================
BASE_DIR = "/content/drive/MyDrive/Human Weight Extimation"
FOLDS_DIR = os.path.join(BASE_DIR, "folds_individual")
NUM_FOLDS = 5

# =====================================================
# FUNCTIONS
# =====================================================

def verify_fold_structure(folds_dir, num_folds=5):
    """Verify that all required fold directories and files exist"""
    print("="*60)
    print("FOLD STRUCTURE VERIFICATION")
    print("="*60)
    
    required_files = [
        "train_images.npy",
        "val_images.npy",
        "train_metadata.csv",
        "val_metadata.csv"
    ]
    
    all_valid = True
    
    for fold_num in range(1, num_folds + 1):
        fold_dir = os.path.join(folds_dir, f"fold-{fold_num}")
        
        print(f"\n[CHECK] Fold-{fold_num}:")
        
        if not os.path.exists(fold_dir):
            print(f"  ❌ Directory not found: {fold_dir}")
            all_valid = False
            continue
        
        print(f"  ✓ Directory exists")
        
        # Check for required files
        for filename in required_files:
            filepath = os.path.join(fold_dir, filename)
            if os.path.exists(filepath):
                # Get file size
                size_mb = os.path.getsize(filepath) / (1024**2)
                print(f"    ✓ {filename:25s} ({size_mb:>8.2f} MB)")
            else:
                print(f"    ❌ {filename:25s} (MISSING)")
                all_valid = False
    
    return all_valid


def verify_fold_data(fold_num, folds_dir):
    """
    Verify data integrity for a specific fold.
    
    Args:
        fold_num: Fold number
        folds_dir: Directory containing folds
    
    Returns:
        Dictionary with verification results
    """
    fold_dir = os.path.join(folds_dir, f"fold-{fold_num}")
    
    # Load data
    train_images = np.load(os.path.join(fold_dir, "train_images.npy"))
    val_images = np.load(os.path.join(fold_dir, "val_images.npy"))
    train_df = pd.read_csv(os.path.join(fold_dir, "train_metadata.csv"))
    val_df = pd.read_csv(os.path.join(fold_dir, "val_metadata.csv"))
    
    # Check image-metadata consistency
    train_consistent = len(train_images) == len(train_df)
    val_consistent = len(val_images) == len(val_df)
    
    # Check for weight column
    train_has_weight = "weight" in train_df.columns
    val_has_weight = "weight" in val_df.columns
    
    # Check for missing weights
    train_missing_weights = 0
    val_missing_weights = 0
    
    if train_has_weight:
        train_missing_weights = train_df["weight"].isna().sum()
    
    if val_has_weight:
        val_missing_weights = val_df["weight"].isna().sum()
    
    return {
        'fold': fold_num,
        'train_images': train_images.shape,
        'val_images': val_images.shape,
        'train_meta': len(train_df),
        'val_meta': len(val_df),
        'train_consistent': train_consistent,
        'val_consistent': val_consistent,
        'train_has_weight': train_has_weight,
        'val_has_weight': val_has_weight,
        'train_missing_weights': train_missing_weights,
        'val_missing_weights': val_missing_weights,
        'train_df': train_df,
        'val_df': val_df
    }


def display_fold_statistics(fold_num, fold_data):
    """Display detailed statistics for a fold"""
    print(f"\n{'='*60}")
    print(f"FOLD-{fold_num} DETAILED STATISTICS")
    print(f"{'='*60}")
    
    # Images
    print(f"\n📊 Images:")
    print(f"  Train: {fold_data['train_images']}")
    print(f"  Val:   {fold_data['val_images']}")
    
    # Metadata
    print(f"\n📊 Metadata:")
    print(f"  Train rows: {fold_data['train_meta']}")
    print(f"  Val rows:   {fold_data['val_meta']}")
    print(f"  Train columns: {len(fold_data['train_df'].columns)}")
    print(f"  Val columns:   {len(fold_data['val_df'].columns)}")
    
    # Consistency
    print(f"\n📊 Consistency:")
    status_train = "✓" if fold_data['train_consistent'] else "❌"
    status_val = "✓" if fold_data['val_consistent'] else "❌"
    print(f"  Train images-metadata match: {status_train}")
    print(f"  Val images-metadata match:   {status_val}")
    
    # Weight column
    print(f"\n📊 Weight Column:")
    weight_train = "✓" if fold_data['train_has_weight'] else "❌"
    weight_val = "✓" if fold_data['val_has_weight'] else "❌"
    print(f"  Train has weight: {weight_train}")
    print(f"  Val has weight:   {weight_val}")
    
    if fold_data['train_has_weight']:
        print(f"  Train missing: {fold_data['train_missing_weights']}")
    
    if fold_data['val_has_weight']:
        print(f"  Val missing:   {fold_data['val_missing_weights']}")
    
    # Weight statistics
    if fold_data['train_has_weight'] and fold_data['train_missing_weights'] == 0:
        train_weights = fold_data['train_df']['weight']
        print(f"\n📊 Train Weight Statistics:")
        print(f"  Min:    {train_weights.min():.2f}")
        print(f"  Max:    {train_weights.max():.2f}")
        print(f"  Mean:   {train_weights.mean():.2f}")
        print(f"  Median: {train_weights.median():.2f}")
        print(f"  Std:    {train_weights.std():.2f}")
    
    if fold_data['val_has_weight'] and fold_data['val_missing_weights'] == 0:
        val_weights = fold_data['val_df']['weight']
        print(f"\n📊 Val Weight Statistics:")
        print(f"  Min:    {val_weights.min():.2f}")
        print(f"  Max:    {val_weights.max():.2f}")
        print(f"  Mean:   {val_weights.mean():.2f}")
        print(f"  Median: {val_weights.median():.2f}")
        print(f"  Std:    {val_weights.std():.2f}")


def create_summary_table(all_fold_data):
    """Create and display a summary table for all folds"""
    summary = []
    
    for fold_data in all_fold_data:
        summary.append({
            'Fold': fold_data['fold'],
            'Train Size': fold_data['train_meta'],
            'Val Size': fold_data['val_meta'],
            'Train Weight': '✓' if fold_data['train_has_weight'] else '❌',
            'Val Weight': '✓' if fold_data['val_has_weight'] else '❌',
            'Train Missing': fold_data['train_missing_weights'],
            'Val Missing': fold_data['val_missing_weights']
        })
    
    summary_df = pd.DataFrame(summary)
    
    print("\n" + "="*60)
    print("ALL FOLDS SUMMARY TABLE")
    print("="*60)
    print(f"\n{summary_df.to_string(index=False)}")
    
    # Calculate totals
    total_train = summary_df['Train Size'].sum()
    total_val = summary_df['Val Size'].sum()
    total_train_missing = summary_df['Train Missing'].sum()
    total_val_missing = summary_df['Val Missing'].sum()
    
    print(f"\n{'='*60}")
    print("OVERALL TOTALS")
    print(f"{'='*60}")
    print(f"Total train samples:        {total_train}")
    print(f"Total val samples:          {total_val}")
    print(f"Total samples:              {total_train + total_val}")
    print(f"Train missing weights:      {total_train_missing}")
    print(f"Val missing weights:        {total_val_missing}")
    print(f"Total missing weights:      {total_train_missing + total_val_missing}")
    
    return summary_df


# =====================================================
# MAIN EXECUTION
# =====================================================

def main():
    """Main verification pipeline"""
    
    print("="*60)
    print("COMPREHENSIVE FOLD VERIFICATION")
    print("="*60)
    
    # Check if folds directory exists
    if not os.path.exists(FOLDS_DIR):
        print(f"\n❌ ERROR: Folds directory not found!")
        print(f"   Expected: {FOLDS_DIR}")
        print("\n   Please run script 05_create_folds.py first")
        return
    
    print(f"\n✓ Folds directory: {FOLDS_DIR}")
    
    # Verify fold structure
    structure_valid = verify_fold_structure(FOLDS_DIR, NUM_FOLDS)
    
    if not structure_valid:
        print("\n❌ Fold structure verification failed!")
        print("   Please check the errors above")
        return
    
    print("\n✓ All fold directories and files exist")
    
    # Verify each fold's data
    print("\n" + "="*60)
    print("VERIFYING FOLD DATA")
    print("="*60)
    
    all_fold_data = []
    all_valid = True
    
    for fold_num in range(1, NUM_FOLDS + 1):
        print(f"\n[VERIFY] Fold-{fold_num}...")
        
        try:
            fold_data = verify_fold_data(fold_num, FOLDS_DIR)
            all_fold_data.append(fold_data)
            
            # Check for issues
            if not fold_data['train_consistent']:
                print(f"  ⚠️ Train images-metadata count mismatch!")
                all_valid = False
            
            if not fold_data['val_consistent']:
                print(f"  ⚠️ Val images-metadata count mismatch!")
                all_valid = False
            
            if not fold_data['train_has_weight']:
                print(f"  ⚠️ Train metadata missing 'weight' column!")
                all_valid = False
            
            if not fold_data['val_has_weight']:
                print(f"  ⚠️ Val metadata missing 'weight' column!")
                all_valid = False
            
            if fold_data['train_missing_weights'] > 0:
                print(f"  ⚠️ Train has {fold_data['train_missing_weights']} missing weights!")
            
            if fold_data['val_missing_weights'] > 0:
                print(f"  ⚠️ Val has {fold_data['val_missing_weights']} missing weights!")
            
            if (fold_data['train_consistent'] and 
                fold_data['val_consistent'] and
                fold_data['train_has_weight'] and 
                fold_data['val_has_weight']):
                print(f"  ✓ Fold-{fold_num} verified successfully")
            
        except Exception as e:
            print(f"  ❌ Error verifying Fold-{fold_num}: {str(e)}")
            all_valid = False
    
    # Display detailed statistics for each fold
    for fold_data in all_fold_data:
        display_fold_statistics(fold_data['fold'], fold_data)
    
    # Create summary table
    summary_df = create_summary_table(all_fold_data)
    
    # Final verdict
    print("\n" + "="*60)
    print("VERIFICATION RESULT")
    print("="*60)
    
    if all_valid:
        print("\n✅ ALL FOLDS VERIFIED SUCCESSFULLY!")
        print("✓ All structures are correct")
        print("✓ All data is consistent")
        print("✓ All weight columns present")
        print("\n🎯 Folds are ready for model training!")
    else:
        print("\n⚠️ VERIFICATION COMPLETED WITH WARNINGS!")
        print("   Please review the issues above")
        print("\n   You may need to:")
        print("   1. Re-run 05_create_folds.py")
        print("   2. Re-run 06_add_weights_to_folds.py")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
