"""
Script 2: Image Preprocessing and Dataset Creation
This script processes all images, extracts metadata, and saves the processed dataset.
"""

import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# =====================================================
# CONFIGURATION
# =====================================================
BASE_DIR = "/content/drive/MyDrive/Human Weight Extimation"
CSV_PATH = os.path.join(BASE_DIR, "Input Data/All_features_224.csv")
IMG_DIR = os.path.join(BASE_DIR, "Input Data/Image_224")

# Output directory
SAVE_DIR = os.path.join(BASE_DIR, "processed_dataset")

# Output file paths
SAVE_IMG_NPY = os.path.join(SAVE_DIR, "processed_images.npy")
SAVE_CSV = os.path.join(SAVE_DIR, "processed_metadata.csv")

# Image settings
IMG_SIZE = (224, 224)

# Metadata columns to extract
META_COLS = [
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

# =====================================================
# FUNCTIONS
# =====================================================

def preprocess_image(path):
    """
    Load and preprocess a single image.
    
    Args:
        path: Full path to the image file
        
    Returns:
        Preprocessed image as numpy array (uint8, 0-255)
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img, dtype=np.uint8)
    return img


def process_dataset():
    """
    Main function to process the entire dataset.
    Loads images, extracts metadata, and saves processed data.
    """
    
    # Create output directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"✓ Output directory created: {SAVE_DIR}")
    
    # Load metadata CSV
    print(f"\n[LOAD] Reading metadata from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df["filename"] = df["filename"].astype(str)
    print(f"✓ Loaded {len(df)} metadata rows")
    
    # Verify required columns exist
    missing_cols = [col for col in META_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Initialize lists for processed data
    processed_images = []
    processed_meta = []
    processed_filenames = []
    skipped_files = []
    
    # Process all images
    print(f"\n[PROCESS] Processing images from: {IMG_DIR}")
    for fname in tqdm(df["filename"], desc="Processing images"):
        full_path = os.path.join(IMG_DIR, fname)
        
        # Check if image exists
        if not os.path.exists(full_path):
            skipped_files.append(fname)
            continue
        
        try:
            # Preprocess image
            img = preprocess_image(full_path)
            processed_images.append(img)
            
            # Extract metadata row
            meta_row = df.loc[df["filename"] == fname, META_COLS].values[0]
            processed_meta.append(meta_row)
            
            processed_filenames.append(fname)
            
        except Exception as e:
            print(f"\n⚠️ Error processing {fname}: {str(e)}")
            skipped_files.append(fname)
            continue
    
    # Report skipped files
    if skipped_files:
        print(f"\n⚠️ Skipped {len(skipped_files)} files")
        print("First 10 skipped files:")
        for f in skipped_files[:10]:
            print(f"  - {f}")
    
    # Convert to numpy arrays
    print("\n[CONVERT] Converting to numpy arrays...")
    processed_images = np.array(processed_images, dtype=np.uint8)
    processed_meta = np.array(processed_meta, dtype=np.float32)
    
    print(f"✓ Final images shape: {processed_images.shape}")
    print(f"✓ Final metadata shape: {processed_meta.shape}")
    
    # Save processed images as NPY
    print(f"\n[SAVE] Saving processed images to: {SAVE_IMG_NPY}")
    np.save(SAVE_IMG_NPY, processed_images)
    print("✓ Images saved successfully")
    
    # Save metadata as CSV
    print(f"\n[SAVE] Saving metadata to: {SAVE_CSV}")
    out_df = pd.DataFrame(processed_meta, columns=META_COLS)
    out_df["filename"] = processed_filenames
    out_df.to_csv(SAVE_CSV, index=False)
    print("✓ Metadata saved successfully")
    
    # Summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total files in CSV: {len(df)}")
    print(f"Successfully processed: {len(processed_images)}")
    print(f"Skipped: {len(skipped_files)}")
    print(f"Success rate: {100 * len(processed_images) / len(df):.2f}%")
    print("="*60)
    
    return processed_images, out_df


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    print("Starting image preprocessing pipeline...")
    print("="*60)
    
    processed_images, metadata_df = process_dataset()
    
    print("\n✅ Image preprocessing completed successfully!")
    print(f"\nOutput files:")
    print(f"  - Images: {SAVE_IMG_NPY}")
    print(f"  - Metadata: {SAVE_CSV}")
