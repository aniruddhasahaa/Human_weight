# Human Weight Estimation - Complete Project Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Complete Pipeline](#complete-pipeline)
6. [Individual Modules](#individual-modules)
7. [Configuration](#configuration)
8. [Outputs](#outputs)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This project implements an end-to-end deep learning pipeline for human weight estimation from images and body measurements. It includes:

- **Data preprocessing** with feature engineering
- **5-fold cross-validation** setup
- **XGBoost metadata-only model**
- **CNN image-only models** (EfficientNet, MobileNet, ResNet)
- **Comprehensive evaluation** and visualization

### Key Features
- ✅ Fully automated pipeline
- ✅ Multiple model architectures
- ✅ Reproducible experiments (fixed random seeds)
- ✅ GPU acceleration support
- ✅ Comprehensive logging and visualization

---

## 🚀 Quick Start

### Option 1: Run Complete Pipeline (Recommended)

```bash
# Run everything with one command
python MASTER_PIPELINE.py
```

This will execute all stages in order:
1. Data preprocessing 
2. Fold creation  
3. XGBoost training  
4. CNN training  
5. Results analysis  

### Option 2: Run Stages Individually

```bash
# Stage 1: Data Preprocessing
python 01_load_and_explore_data.py
python 02_preprocess_images.py
python 03_feature_engineering.py

# Stage 2: Fold Creation
python 05_create_folds.py
python 06_add_weights_to_folds.py

# Stage 3: Model Training
python run_xgboost_cv.py  # Metadata-only
python run_cnn_cv.py       # Image-only

# Stage 4: Analysis
python analyze_results.py
```

### Option 3: Quick Test (Single Fold)

```bash
# Test XGBoost on one fold
python train_xgboost_meta.py

# Test CNN on one fold
python train_cnn_image.py
```

---

## 📁 Project Structure

```
project/
├── MASTER_PIPELINE.py              ⭐ Main pipeline orchestrator
├── README_COMPLETE.md              📖 This file
│
├── Data Preprocessing Scripts (Phase 1)
│   ├── 01_load_and_explore_data.py
│   ├── 02_preprocess_images.py
│   ├── 03_feature_engineering.py
│   └── 04_verify_preprocessed_data.py
│
├── Fold Creation Scripts (Phase 2)
│   ├── 05_create_folds.py
│   ├── 06_add_weights_to_folds.py
│   └── 07_verify_folds.py
│
├── XGBoost Training Scripts (Phase 3)
│   ├── config_training.py          # Configuration
│   ├── data_loader.py               # Data loading utilities
│   ├── train_xgboost_meta.py        # Single experiment
│   └── run_xgboost_cv.py            # 5-fold CV
│
├── CNN Training Scripts (Phase 4)
│   ├── image_data_loader.py         # Image data loading
│   ├── cnn_models.py                # Model architectures
│   ├── train_cnn_image.py           # Single experiment
│   └── run_cnn_cv.py                # 5-fold CV
│
├── Analysis Scripts (Phase 5)
│   └── analyze_results.py
│
└── Documentation
    ├── README.md                     # General overview
    ├── README_TRAINING.md            # XGBoost training guide
    ├── README_CNN_TRAINING.md        # CNN training guide
    └── requirements.txt              # Python dependencies
```

---

## 💾 Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (8GB+ VRAM recommended for CNN training)
- Google Colab or local machine with GPU

### Install Dependencies

```bash
# Core dependencies
pip install pandas numpy scikit-learn
pip install tensorflow>=2.8.0
pip install xgboost
pip install pillow tqdm

# Visualization (optional)
pip install matplotlib seaborn

# Or install all at once
pip install -r requirements.txt
```

### Google Colab Setup

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to project directory
%cd /content/drive/MyDrive/Human\ Weight\ Extimation/

# Install dependencies
!pip install -q xgboost tensorflow pillow tqdm
```

---

## 🔄 Complete Pipeline

The `CODE_PIPELINE.py` script orchestrates the entire workflow:

### Pipeline Stages

#### **Phase 1: Data Preprocessing** (30-60 min)
```
01. Load and explore raw data
02. Resize images to 224x224, normalize
03. Create derived features (height estimates, ratios, etc.)
04. Verify data integrity
```

**Outputs:**
- `processed_dataset/processed_images.npy`
- `processed_dataset/processed_metadata.csv`
- `processed_dataset/processed_metadata_with_features.csv`

#### **Phase 2: Fold Creation** (10-20 min)
```
05. Split data into 5 folds (stratified)
06. Add weight labels to each fold
07. Verify all folds
```

**Outputs:**
- `folds_individual/fold-1/` to `fold-5/`
  - Each contains: `train_images.npy`, `val_images.npy`
  - `train_metadata.csv`, `val_metadata.csv`

#### **Phase 3: XGBoost Training** (20-40 min)
```
08. Train XGBoost model on metadata (9 features)
09. 5-fold cross-validation
10. Evaluate and save predictions
```

**Outputs:**
- `fusion_results_v1/meta_only_results.csv`
- `fusion_results_v1/meta_only_predictions.csv`

#### **Phase 4: CNN Training** (6-12 hours)
```
11. Train EfficientNet-B0 on images
12. 5-fold cross-validation
13. Evaluate and save models
```

**Outputs:**
- `fusion_results_v1/image_test_fold_1/` to `fold_5/`
  - Each contains: `best_model.keras`, `predictions.csv`
- `fusion_results_v1/cnn_results_summary.csv`

#### **Phase 5: Results Analysis** (5-10 min)
```
14. Generate performance plots
15. Statistical summaries
16. Model comparison
```

**Outputs:**
- `fusion_results_v1/fold_comparison.png`
- `fusion_results_v1/prediction_scatter.png`
- `fusion_results_v1/error_distribution.png`

### Customizing Pipeline Stages

Edit `MASTER_PIPELINE.py`:

```python
class PipelineConfig:
    # Enable/disable stages
    RUN_PREPROCESSING = True
    RUN_FOLD_CREATION = True
    RUN_VERIFICATION = True
    RUN_XGBOOST_TRAINING = True
    RUN_CNN_TRAINING = False  # Skip CNN (long training)
    RUN_ANALYSIS = True
    
    # Choose CNN model
    CNN_MODEL = 'efficientnetB0'  # or 'mobilenet', 'resnet', 'custom'
```

---

## 📦 Individual Modules

### 1. Data Preprocessing

#### Load and Explore
```bash
python 01_load_and_explore_data.py
```
- Loads CSV and images
- Displays dataset statistics
- Verifies file counts

#### Preprocess Images
```bash
python 02_preprocess_images.py
```
- Resizes images to 224x224
- Converts to RGB
- Saves as numpy array

#### Feature Engineering
```bash
python 03_feature_engineering.py
```
- Creates 12 derived features
- Height estimation, ratios, BMI-like features
- Saves enhanced metadata

### 2. XGBoost Training

#### Single Experiment
```python
from train_xgboost_meta import run_meta_experiment

metrics, preds = run_meta_experiment(
    train_folds=[1, 2, 3, 4],
    test_fold=5,
    exp_name="test_experiment"
)
```

#### 5-Fold Cross-Validation
```bash
python run_xgboost_cv.py
```
- Trains 5 models automatically
- Saves all results
- ~30 minutes total

### 3. CNN Training

#### Model Selection
Available in `cnn_models.py`:
- **EfficientNet-B0** (default, best accuracy)
- **MobileNet-V2** (fastest)
- **ResNet-50** (deepest)
- **Custom CNN** (no pre-training)

#### Single Experiment
```python
from train_cnn_image import run_image_experiment

metrics, preds = run_image_experiment(
    train_folds=[1, 2, 3, 4],
    test_fold=5,
    exp_name="test_experiment"
)
```

#### 5-Fold Cross-Validation
```bash
python run_cnn_cv.py
```
- Trains 5 CNN models
- ~2 hours per fold
- Total: 6-12 hours

---

## ⚙️ Configuration

### Global Settings

Edit `config_training.py`:

```python
# Paths
BASE_DIR = "/content/drive/MyDrive/Human Weight Extimation"

# Image settings
IMG_SHAPE = (224, 224, 3)

# Training parameters
BATCH_SIZE = 8
EPOCHS = 75
PATIENCE = 15
LEARNING_RATE = 1e-4

# Metadata features (choose one)
META_COLS = META_COLS_ORIGINAL  # 9 features (default)
# META_COLS = META_COLS_WITH_DERIVED  # 15 features
# META_COLS = META_COLS_SELECTED  # 14 features

# XGBoost parameters
XGBOOST_PARAMS = {
    'n_estimators': 800,
    'max_depth': 4,
    'learning_rate': 0.03,
    ...
}
```

### Feature Sets

#### Original Features (9) - Default
```python
META_COLS = META_COLS_ORIGINAL
```
- shoulder_width_px, waist_width_px, hip_width_px, thigh_width_px
- W_by_Hip, W_by_Shoulder, W_by_Thigh, Hip_by_Shoulder
- Area (px)

#### Original + All Derived (15)
```python
META_COLS = META_COLS_WITH_DERIVED
```
- All original features
- height_est_px, log_area, waist_to_height, etc.

#### Original + Selected Derived (14)
```python
META_COLS = META_COLS_SELECTED
```
- All original features
- Best 5 derived features

---

## 📊 Outputs

### Directory Structure

```
/content/drive/MyDrive/Human Weight Extimation/
├── processed_dataset/
│   ├── processed_images.npy                (N, 224, 224, 3)
│   ├── processed_metadata.csv              Original features
│   └── processed_metadata_with_features.csv Enhanced features
│
├── folds_individual/
│   ├── fold-1/
│   │   ├── train_images.npy
│   │   ├── val_images.npy
│   │   ├── train_metadata.csv
│   │   └── val_metadata.csv
│   ├── fold-2/ ... fold-5/
│
└── fusion_results_v1/
    ├── XGBoost Results
    │   ├── meta_only_results.csv
    │   ├── meta_only_results.json
    │   ├── meta_only_predictions.csv
    │   └── meta_only_summary.txt
    │
    ├── CNN Results
    │   ├── image_test_fold_1/
    │   │   ├── best_model.keras
    │   │   ├── training_log.csv
    │   │   ├── history.csv
    │   │   └── predictions.csv
    │   ├── ... fold_2 to fold_5/
    │   ├── cnn_experiments_summary.json
    │   ├── cnn_results_summary.csv
    │   └── cnn_all_predictions.csv
    │
    └── Visualizations
        ├── fold_comparison.png
        ├── prediction_scatter.png
        └── error_distribution.png
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM) Error

**Symptoms:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**
```python
# Reduce batch size
BATCH_SIZE = 4  # or even 2

# Use smaller model
model_name='mobilenet'

# Enable GPU memory growth (already done in config)
configure_gpu()
```

#### 2. File Not Found Error

**Symptoms:**
```
FileNotFoundError: processed_metadata.csv not found
```

**Solutions:**
```bash
# Run preprocessing first
python 01_load_and_explore_data.py
python 02_preprocess_images.py
python 03_feature_engineering.py

# Or check paths in config_training.py
BASE_DIR = "/your/correct/path"
```

#### 3. Missing Weight Column

**Symptoms:**
```
KeyError: 'weight'
```

**Solutions:**
```bash
# Add weights to folds
python 06_add_weights_to_folds.py

# Verify weights
python 07_verify_folds.py
```

#### 4. Training is Too Slow

**Solutions:**
- Use MobileNet instead of EfficientNet
- Reduce number of folds for testing
- Use smaller image size (requires code changes)
- Reduce epochs (edit `config_training.py`)

#### 5. Poor Model Performance

**Solutions:**
- Check data quality (run verification scripts)
- Try different feature sets
- Tune hyperparameters
- Add more training data
- Increase epochs
- Try different models

---
### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 4GB VRAM | 8GB+ VRAM |
| RAM | 8GB | 16GB+ |
| Storage | 10GB | 20GB+ |
| CPU | 4 cores | 8+ cores |

---

## 🎓 Advanced Usage

### Custom Model Architecture

Create your own model in `cnn_models.py`:

```python
def build_my_custom_model(img_shape=(224, 224, 3), lr=1e-4):
    inputs = layers.Input(shape=img_shape)
    x = layers.Rescaling(1./255)(inputs)
    
    # Your architecture here
    x = layers.Conv2D(64, 3, activation='relu')(x)
    # ...
    
    outputs = layers.Dense(1, name='weight_output')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model
```

### Hyperparameter Tuning

Use grid search:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.03, 0.05],
    'n_estimators': [500, 800, 1000]
}

model = XGBRegressor()
grid_search = GridSearchCV(model, param_grid, cv=3)
grid_search.fit(X_train, y_train)
```

### Ensemble Predictions

Combine XGBoost and CNN:

```python
# Load predictions
xgb_preds = pd.read_csv('meta_only_predictions.csv')
cnn_preds = pd.read_csv('cnn_all_predictions.csv')

# Merge and ensemble
merged = xgb_preds.merge(cnn_preds, on='filename', suffixes=('_xgb', '_cnn'))
merged['y_pred_ensemble'] = 0.5 * merged['y_pred_xgb'] + 0.5 * merged['y_pred_cnn']

# Evaluate
mae = mean_absolute_error(merged['y_true_xgb'], merged['y_pred_ensemble'])
```

---

## 📝 Notes

- All experiments use **SEED=42** for reproducibility
- GPU memory growth is enabled automatically
- Early stopping prevents overfitting (patience=15)
- Models are saved automatically during training
- Results vary slightly due to data distribution across folds

---

## 🔗 References

- **EfficientNetB0**: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- **XGBoost**: [Chen & Guestrin, 2016](https://arxiv.org/abs/1603.02754)
- **Transfer Learning**: [Fuzhen Zhuang, et al, 2020](https://ieeexplore.ieee.org/abstract/document/9134370)

---

## 📧 Support

If you encounter issues:
1. Check this README and individual READMEs
2. Verify paths in `config_training.py`
3. Run verification scripts
4. Check GPU availability: `nvidia-smi`
5. Review error messages carefully

---

**Good luck with your project! 🚀**
