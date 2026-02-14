# XGBoost Metadata-Only Model Training

This directory contains scripts for training and evaluating XGBoost models using only metadata features for human weight estimation.

## 📁 Files Overview

### Core Scripts

1. **`config_training.py`**
   - Configuration file for model training
   - Contains all hyperparameters and paths
   - XGBoost settings, CV configurations
   - GPU configuration and random seed setup

2. **`data_loader.py`**
   - Data loading utilities
   - Functions to load metadata from folds
   - StandardScaler for feature normalization
   - Data integrity verification

3. **`train_xgboost_meta.py`**
   - XGBoost model training functions
   - Single experiment runner
   - Model evaluation and metrics calculation
   - Feature importance analysis

4. **`run_xgboost_cv.py`**
   - Complete 5-fold cross-validation pipeline
   - Runs all experiments automatically
   - Saves results in multiple formats
   - Main script to execute

5. **`analyze_results.py`**
   - Results analysis and visualization
   - Generate performance plots
   - Statistical summaries
   - Error analysis

## 🚀 Quick Start

### Step 1: Configure Settings

Edit `config_training.py` to set your paths and choose feature set:

```python
# Choose feature set
META_COLS = META_COLS_ORIGINAL  # or META_COLS_WITH_DERIVED or META_COLS_SELECTED

# Adjust XGBoost parameters if needed
XGBOOST_PARAMS = {
    'n_estimators': 800,
    'max_depth': 4,
    'learning_rate': 0.03,
    # ... other parameters
}
```

### Step 2: Run 5-Fold Cross-Validation

```bash
python run_xgboost_cv.py
```

This will:
- Train 5 separate models (one for each fold)
- Evaluate on held-out test fold
- Save all results and predictions
- Take approximately 15-30 minutes total

### Step 3: Analyze Results

```bash
python analyze_results.py
```

This generates:
- Summary statistics
- Performance visualizations
- Error analysis
- Comparison plots

## 📊 Feature Set Options

### Option 1: Original Features Only (Default)
```python
META_COLS = META_COLS_ORIGINAL
```
- 9 original features from the dataset
- `shoulder_width_px`, `waist_width_px`, `hip_width_px`, `thigh_width_px`
- `W_by_Hip`, `W_by_Shoulder`, `W_by_Thigh`, `Hip_by_Shoulder`
- `Area (px)`

### Option 2: Original + All Derived Features
```python
META_COLS = META_COLS_WITH_DERIVED
```
- 15 total features (5 original + 10 derived)
- Includes height estimation, log area, ratios, etc.

### Option 3: Original + Selected Derived Features
```python
META_COLS = META_COLS_SELECTED
```
- 14 total features (9 original + 5 best derived)
- Balanced between information and complexity

## ⚙️ XGBoost Hyperparameters

Current stable configuration (in `config_training.py`):

```python
XGBOOST_PARAMS = {
    'n_estimators': 800,        # Number of boosting rounds
    'max_depth': 4,             # Maximum tree depth
    'learning_rate': 0.03,      # Step size shrinkage
    'subsample': 0.9,           # Fraction of samples per tree
    'colsample_bytree': 0.9,    # Fraction of features per tree
    'reg_alpha': 0.5,           # L1 regularization
    'reg_lambda': 1.5,          # L2 regularization
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': -1                # Use all CPU cores
}
```

## 📈 Output Files

After running `run_xgboost_cv.py`, you'll find in `fusion_results_v1/`:

```
fusion_results_v1/
├── meta_only_results.csv          # Metrics for each fold
├── meta_only_results.json         # Same data in JSON format
├── meta_only_predictions.csv      # All predictions with errors
├── meta_only_summary.txt          # Human-readable summary
├── fold_comparison.png            # Performance comparison plot
├── prediction_scatter.png         # Actual vs Predicted plot
└── error_distribution.png         # Error histogram and boxplot
```

## 📋 Expected Results

Typical performance with original features:

```
MAE:  ~8-12 lb
RMSE: ~10-15 lb
R²:   ~0.85-0.92
```

Performance varies by fold due to data distribution.

## 🔍 Detailed Usage

### Run Single Experiment

```python
from train_xgboost_meta import run_meta_experiment
from config_training import set_random_seeds

set_random_seeds()

metrics, preds = run_meta_experiment(
    train_folds=[1, 2, 3, 4],
    test_fold=5,
    exp_name="test_experiment"
)

print(f"MAE: {metrics['mae']:.4f} lb")
```

### Load and Use Trained Model

```python
from data_loader import load_meta_train_val_from_folds, load_meta_test_fold
from xgboost import XGBRegressor

# Load data
(X_train, y_train), (X_val, y_val), scaler = \
    load_meta_train_val_from_folds([1, 2, 3, 4])

# Train model
model = XGBRegressor(**XGBOOST_PARAMS)
model.fit(X_train, y_train)

# Load test data
X_test, y_test, filenames = load_meta_test_fold(5, scaler)

# Predict
predictions = model.predict(X_test)
```

### Analyze Feature Importance

```python
from train_xgboost_meta import analyze_feature_importance
from config_training import META_COLS

# After training a model
importance_df = analyze_feature_importance(model, META_COLS)
print(importance_df)
```

## 🛠️ Customization

### Change Feature Set

Edit `config_training.py`:
```python
# Use different feature set
META_COLS = META_COLS_WITH_DERIVED

# Or create custom feature list
META_COLS = [
    "shoulder_width_px",
    "waist_width_px",
    "Area (px)",
    "height_est_px"
]
```

### Tune Hyperparameters

Edit `config_training.py`:
```python
XGBOOST_PARAMS = {
    'n_estimators': 1000,      # Try more trees
    'max_depth': 5,            # Deeper trees
    'learning_rate': 0.01,     # Slower learning
    # ... adjust other parameters
}
```

### Change CV Split

Edit `config_training.py`:
```python
# Use different fold combinations
CV_EXPERIMENTS = [
    {"train": [1, 2, 3], "test": 4},  # Use 3 folds for training
    {"train": [2, 3, 4], "test": 1},
    # ... more combinations
]
```

## 🐛 Troubleshooting

### Issue: "Fold directory not found"
**Solution:** Make sure you've run the fold creation scripts first:
```bash
python 05_create_folds.py
python 06_add_weights_to_folds.py
```

### Issue: "Missing weight column"
**Solution:** Run the weight labeling script:
```bash
python 06_add_weights_to_folds.py
```

### Issue: Memory errors
**Solution:** Reduce number of folds loaded simultaneously or reduce batch size

### Issue: Poor performance
**Solution:** 
- Try different feature sets
- Tune hyperparameters
- Check for data quality issues
- Verify feature scaling is applied

## 📊 Metrics Explanation

- **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual weight. Lower is better.
- **RMSE (Root Mean Squared Error):** Square root of average squared errors. Penalizes large errors more. Lower is better.
- **R² Score:** Coefficient of determination. Measures how well predictions fit the data. Higher is better (max 1.0).

## 🔬 Advanced Features

### Grid Search for Hyperparameters

```python
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.03, 0.05],
    'n_estimators': [500, 800, 1000]
}

model = XGBRegressor(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
```

### Early Stopping (if using validation set)

```python
model = XGBRegressor(**XGBOOST_PARAMS)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=True
)
```

## 📝 Notes

- All scripts use `SEED=42` for reproducibility
- Feature scaling (StandardScaler) is applied to all features
- Cross-validation ensures robust performance estimates
- Results may vary slightly due to random initialization
- GPU is not used for XGBoost (CPU only)

## 🎯 Next Steps

After training metadata-only model:
1. Train image-based models
2. Implement fusion models (combining images + metadata)
3. Ensemble predictions from multiple models
4. Deploy best-performing model

## 📄 Dependencies

Required packages:
```
xgboost>=1.7.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

Install with:
```bash
pip install xgboost scikit-learn pandas numpy matplotlib seaborn
```
