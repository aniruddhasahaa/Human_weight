"""
Script 3: Feature Engineering - Add Derived Physical Features
This script adds engineered features to the processed metadata CSV.
"""

import os
import pandas as pd
import numpy as np

# =====================================================
# CONFIGURATION
# =====================================================
PROC_DIR = "/content/drive/MyDrive/Human Weight Extimation/processed_dataset"
CSV_PATH = os.path.join(PROC_DIR, "processed_metadata.csv")
CSV_OUTPUT_PATH = os.path.join(PROC_DIR, "processed_metadata_with_features.csv")

# You can set this to overwrite the original file instead:
# CSV_OUTPUT_PATH = CSV_PATH

# =====================================================
# FEATURE ENGINEERING FUNCTIONS
# =====================================================

def add_derived_features(df):
    """
    Add derived physical features based on body measurements.
    
    Args:
        df: DataFrame with original features
        
    Returns:
        DataFrame with additional engineered features
    """
    eps = 1e-6  # Small constant to prevent division by zero
    
    # Extract original features
    sw = df["shoulder_width_px"].values
    ww = df["waist_width_px"].values
    hw = df["hip_width_px"].values
    tw = df["thigh_width_px"].values
    area = df["Area (px)"].values
    
    print("[ENGINEER] Creating derived features...")
    
    # ----------------------------------
    # Height proxy (scale normalization)
    # ----------------------------------
    print("  - Height estimation features...")
    max_width = np.maximum.reduce([sw, ww, hw, tw, np.ones_like(sw)])
    height_est = area / (max_width + eps)
    df["height_est_px"] = height_est
    
    # ----------------------------------
    # Density & shape features
    # ----------------------------------
    print("  - Logarithmic area...")
    df["log_area"] = np.log(area + 1.0)
    
    # ----------------------------------
    # Width-to-height ratios
    # ----------------------------------
    print("  - Width-to-height ratios...")
    df["waist_to_height"] = ww / (height_est + eps)
    df["hip_to_height"] = hw / (height_est + eps)
    df["thigh_to_height"] = tw / (height_est + eps)
    df["shoulder_to_height"] = sw / (height_est + eps)
    
    # ----------------------------------
    # Area-based features
    # ----------------------------------
    print("  - Area-based features...")
    df["area_over_height2"] = area / (height_est**2 + eps)
    df["area_over_shoulder_height"] = area / ((sw * height_est) + eps)
    
    # ----------------------------------
    # Body proportion ratios
    # ----------------------------------
    print("  - Body proportion ratios...")
    df["waist_to_hip"] = ww / (hw + eps)
    df["thigh_to_hip"] = tw / (hw + eps)
    
    # ----------------------------------
    # Mean-width based features
    # ----------------------------------
    print("  - Mean-width based features...")
    mean_width = (sw + ww + hw + tw) / 4.0
    df["mean_width_px"] = mean_width
    df["area_over_mean_width"] = area / (mean_width + eps)
    
    # ----------------------------------
    # Cleanup: Handle inf/nan values
    # ----------------------------------
    print("  - Cleaning invalid values...")
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    
    return df


def verify_features(df, new_cols):
    """
    Verify that new features were added correctly.
    
    Args:
        df: DataFrame with new features
        new_cols: List of new column names to verify
    """
    print("\n[VERIFY] Checking new features...")
    
    # Check if all columns exist
    missing = [col for col in new_cols if col not in df.columns]
    if missing:
        print(f"⚠️ Missing columns: {missing}")
        return False
    
    print("✓ All new columns present")
    
    # Check for any remaining invalid values
    for col in new_cols:
        inf_count = np.isinf(df[col]).sum()
        nan_count = df[col].isna().sum()
        
        if inf_count > 0 or nan_count > 0:
            print(f"⚠️ {col}: {inf_count} inf values, {nan_count} NaN values")
    
    # Display statistics
    print("\n[STATS] Feature statistics:")
    print(df[new_cols].describe())
    
    return True


# =====================================================
# MAIN EXECUTION
# =====================================================

def main():
    """Main function to run feature engineering pipeline"""
    
    # Load metadata
    print("="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    print(f"\n[LOAD] Reading metadata from: {CSV_PATH}")
    
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Metadata file not found: {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    print(f"✓ Loaded {len(df)} rows")
    print(f"✓ Original columns: {len(df.columns)}")
    
    # Display original columns
    print("\n[INFO] Original features:")
    for col in df.columns:
        print(f"  - {col}")
    
    # Add derived features
    print(f"\n[PROCESS] Adding derived features...")
    df = add_derived_features(df)
    print(f"✓ Features added")
    print(f"✓ Total columns now: {len(df.columns)}")
    
    # List of new features
    new_cols = [
        "height_est_px", 
        "log_area",
        "waist_to_height", 
        "hip_to_height",
        "thigh_to_height", 
        "shoulder_to_height",
        "area_over_height2", 
        "area_over_shoulder_height",
        "waist_to_hip", 
        "thigh_to_hip",
        "mean_width_px", 
        "area_over_mean_width"
    ]
    
    # Verify features
    verify_features(df, new_cols)
    
    # Display sample rows
    print("\n[SAMPLE] First 5 rows of new features:")
    print(df[new_cols].head())
    
    # Save updated CSV
    print(f"\n[SAVE] Saving enhanced metadata to: {CSV_OUTPUT_PATH}")
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print("✓ File saved successfully")
    
    # Summary
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    print(f"Original features: {len(df.columns) - len(new_cols)}")
    print(f"New features added: {len(new_cols)}")
    print(f"Total features: {len(df.columns)}")
    print(f"Total samples: {len(df)}")
    print("="*60)
    
    print("\n[INFO] New features list:")
    for i, col in enumerate(new_cols, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\n✅ Feature engineering completed successfully!")
    print(f"\nOutput file: {CSV_OUTPUT_PATH}")
    
    return df


if __name__ == "__main__":
    df_enhanced = main()
