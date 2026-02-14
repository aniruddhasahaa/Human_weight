"""
Script 4: Verify Preprocessed Data
This script loads and verifies the preprocessed images and metadata.
"""

import os
import random
import numpy as np
import pandas as pd

# =====================================================
# CONFIGURATION
# =====================================================
BASE_DIR = "/content/drive/MyDrive/Human Weight Extimation"
PROC_DIR = os.path.join(BASE_DIR, "processed_dataset")

IMG_NPY_PATH = os.path.join(PROC_DIR, "processed_images.npy")
CSV_NEW_PATH = os.path.join(PROC_DIR, "processed_metadata.csv")

# Number of random samples to verify
NUM_SAMPLES = 5
SEED = 42

# =====================================================
# FUNCTIONS
# =====================================================

def load_processed_data():
    """Load preprocessed images and metadata"""
    print("="*60)
    print("LOADING PREPROCESSED DATA")
    print("="*60)
    
    print("\n[LOAD] Loading processed images...")
    print(f"Path: {IMG_NPY_PATH}")
    images = np.load(IMG_NPY_PATH)
    print(f"✓ Images loaded: {images.shape}")
    
    print("\n[LOAD] Loading processed metadata...")
    print(f"Path: {CSV_NEW_PATH}")
    df = pd.read_csv(CSV_NEW_PATH)
    print(f"✓ Metadata loaded: {df.shape}")
    
    return images, df


def verify_data_consistency(images, df):
    """Check if image count matches metadata count"""
    print("\n" + "="*60)
    print("DATA CONSISTENCY CHECK")
    print("="*60)
    
    img_count = len(images)
    meta_count = len(df)
    
    print(f"\nImage count: {img_count}")
    print(f"Metadata count: {meta_count}")
    
    if img_count == meta_count:
        print("\n✓ Image count matches metadata count!")
        return True
    else:
        print(f"\n⚠️ MISMATCH FOUND!")
        print(f"   Difference: {abs(img_count - meta_count)} samples")
        return False


def verify_random_samples(images, df, num_samples=5):
    """Verify random samples from the dataset"""
    print("\n" + "="*60)
    print(f"RANDOM SAMPLE VERIFICATION ({num_samples} samples)")
    print("="*60)
    
    # Set seed for reproducibility
    random.seed(SEED)
    
    # Get random sample indices
    total_samples = len(df)
    if num_samples > total_samples:
        num_samples = total_samples
    
    sample_indices = random.sample(range(total_samples), num_samples)
    
    print(f"\nSelected indices: {sample_indices}\n")
    
    for i, idx in enumerate(sample_indices, 1):
        row = df.iloc[idx]
        filename = row["filename"]
        img = images[idx]
        shape = img.shape
        
        print("-" * 60)
        print(f"SAMPLE {i}/{num_samples}")
        print("-" * 60)
        print(f"Index: {idx}")
        print(f"Filename: {filename}")
        print(f"Image shape: {shape}")
        print(f"Image dtype: {img.dtype}")
        print(f"Image value range: [{img.min()}, {img.max()}]")
        print(f"\nMetadata values:")
        
        # Print metadata in a formatted way
        for col, val in row.items():
            print(f"  {col:25s}: {val}")
        print()
    
    print("-" * 60)


def display_statistics(images, df):
    """Display dataset statistics"""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    print(f"\n📊 Images:")
    print(f"   Total: {len(images)}")
    print(f"   Shape: {images.shape}")
    print(f"   Dtype: {images.dtype}")
    print(f"   Memory: {images.nbytes / (1024**2):.2f} MB")
    
    print(f"\n📊 Metadata:")
    print(f"   Total rows: {len(df)}")
    print(f"   Total columns: {len(df.columns)}")
    print(f"\n   Columns:")
    for col in df.columns:
        print(f"      - {col}")
    
    # Check for missing values
    print(f"\n📊 Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✓ No missing values found")
    else:
        print(f"   ⚠️ Found missing values:")
        for col, count in missing[missing > 0].items():
            print(f"      {col}: {count}")


def verify_image_quality(images, num_checks=10):
    """Verify image quality by checking for corrupted or blank images"""
    print("\n" + "="*60)
    print("IMAGE QUALITY CHECK")
    print("="*60)
    
    issues = []
    
    print(f"\nChecking {num_checks} random images for quality issues...")
    random.seed(SEED)
    check_indices = random.sample(range(len(images)), min(num_checks, len(images)))
    
    for idx in check_indices:
        img = images[idx]
        
        # Check for all-zero images (blank)
        if np.all(img == 0):
            issues.append(f"Index {idx}: All-zero image (blank)")
        
        # Check for constant images
        if img.std() < 1.0:
            issues.append(f"Index {idx}: Very low variance (std={img.std():.4f})")
    
    if not issues:
        print("✓ All checked images appear valid")
    else:
        print(f"⚠️ Found {len(issues)} potential issues:")
        for issue in issues:
            print(f"   - {issue}")
    
    return issues


# =====================================================
# MAIN EXECUTION
# =====================================================

def main():
    """Main verification pipeline"""
    
    print("="*60)
    print("PREPROCESSED DATA VERIFICATION")
    print("="*60)
    
    # Load data
    images, df = load_processed_data()
    
    # Verify consistency
    is_consistent = verify_data_consistency(images, df)
    
    # Display statistics
    display_statistics(images, df)
    
    # Verify random samples
    verify_random_samples(images, df, NUM_SAMPLES)
    
    # Verify image quality
    verify_image_quality(images, num_checks=10)
    
    # Final summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    if is_consistent:
        print("✅ Data verification completed successfully!")
        print("✓ Image count matches metadata")
        print("✓ Random samples verified")
        print("✓ Image quality checked")
    else:
        print("⚠️ Data verification completed with warnings!")
        print("⚠️ Please review the issues above")
    
    print("="*60)
    
    return images, df


if __name__ == "__main__":
    images, df = main()
