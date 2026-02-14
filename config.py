"""
Configuration File
Shared settings and paths for all preprocessing scripts.
"""

import os

# =====================================================
# BASE DIRECTORIES
# =====================================================
BASE_DIR = "/content/drive/MyDrive/Human Weight Extimation"
INPUT_DIR = os.path.join(BASE_DIR, "Input Data")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_dataset")

# =====================================================
# INPUT PATHS
# =====================================================
CSV_PATH = os.path.join(INPUT_DIR, "All_features_224.csv")
IMG_DIR = os.path.join(INPUT_DIR, "Image_224")

# =====================================================
# OUTPUT PATHS
# =====================================================
SAVE_IMG_NPY = os.path.join(OUTPUT_DIR, "processed_images.npy")
SAVE_CSV = os.path.join(OUTPUT_DIR, "processed_metadata.csv")
SAVE_CSV_ENHANCED = os.path.join(OUTPUT_DIR, "processed_metadata_with_features.csv")

# =====================================================
# IMAGE SETTINGS
# =====================================================
IMG_SIZE = (224, 224)
IMG_DTYPE = "uint8"  # Store images as uint8 (0-255) to save space

# =====================================================
# METADATA COLUMNS
# =====================================================
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
# DERIVED FEATURE NAMES
# =====================================================
DERIVED_FEATURES = [
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

# =====================================================
# PROCESSING PARAMETERS
# =====================================================
EPSILON = 1e-6  # Small constant to prevent division by zero

# =====================================================
# DISPLAY SETTINGS
# =====================================================
SEPARATOR = "=" * 60


def print_config():
    """Print current configuration settings"""
    print(SEPARATOR)
    print("CONFIGURATION SETTINGS")
    print(SEPARATOR)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"\nImage Settings:")
    print(f"  Size: {IMG_SIZE}")
    print(f"  Data Type: {IMG_DTYPE}")
    print(f"\nMetadata Columns: {len(META_COLS)}")
    print(f"Derived Features: {len(DERIVED_FEATURES)}")
    print(SEPARATOR)


if __name__ == "__main__":
    print_config()
