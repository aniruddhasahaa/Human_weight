"""
Script 1: Load and Explore Dataset
This script loads the CSV metadata and verifies the image directory.
"""

import pandas as pd
import os

# Configuration
CSV_PATH = "/content/drive/MyDrive/Human Weight Extimation/Input Data/All_features_224.csv"
IMG_DIR = "/content/drive/MyDrive/Human Weight Extimation/Input Data/Image_224"

def load_and_explore():
    """Load CSV and explore the dataset"""
    
    # Load CSV
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    print("✓ CSV Loaded!")
    
    # Display basic info
    print("\n" + "="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    print(f"Total rows: {len(df)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Check images
    print("\n" + "="*50)
    print("IMAGE DIRECTORY CHECK")
    print("="*50)
    
    if os.path.exists(IMG_DIR):
        image_files = os.listdir(IMG_DIR)
        print(f"✓ Total images found: {len(image_files)}")
        print(f"\nFirst 10 image filenames:")
        for img in image_files[:10]:
            print(f"  - {img}")
    else:
        print(f"⚠️ Image directory not found: {IMG_DIR}")
    
    return df

if __name__ == "__main__":
    df = load_and_explore()
    print("\n✅ Data exploration completed!")
