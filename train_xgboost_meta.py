"""
XGBoost Metadata-Only Model Training
Train and evaluate XGBoost model using only metadata features.
"""

import time
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import utilities
from config_training import XGBOOST_PARAMS, OUTPUT_DIR
from data_loader import (
    load_meta_train_val_from_folds,
    load_meta_test_fold,
    verify_data_integrity
)

# =====================================================
# XGBOOST TRAINING FUNCTION
# =====================================================

def train_xgboost_model(X_train, y_train, X_val=None, y_val=None):
    """
    Train XGBoost model on metadata features.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional, for monitoring)
        y_val: Validation labels (optional, for monitoring)
    
    Returns:
        Trained XGBoost model
    """
    print("\n[TRAIN] Training XGBoost model...")
    print(f"  Training samples: {len(X_train)}")
    
    # Create model
    model = XGBRegressor(**XGBOOST_PARAMS)
    
    # Train model
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"✓ Training completed in {train_time:.2f} seconds")
    
    # Optional: Evaluate on validation set
    if X_val is not None and y_val is not None:
        val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f"  Validation MAE: {val_mae:.4f} lb")
        print(f"  Validation RMSE: {val_rmse:.4f} lb")
    
    return model, train_time


def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model on test data.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n[EVAL] Evaluating model on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Calculate additional metrics
    abs_errors = np.abs(y_test - y_pred)
    median_ae = np.median(abs_errors)
    max_error = np.max(abs_errors)
    
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'median_ae': float(median_ae),
        'max_error': float(max_error)
    }
    
    print(f"✓ Evaluation complete")
    print(f"  MAE:       {mae:.4f} lb")
    print(f"  RMSE:      {rmse:.4f} lb")
    print(f"  R²:        {r2:.4f}")
    print(f"  Median AE: {median_ae:.4f} lb")
    print(f"  Max Error: {max_error:.4f} lb")
    
    return metrics, y_pred


def create_predictions_dataframe(filenames, y_true, y_pred):
    """
    Create a DataFrame with predictions and errors.
    
    Args:
        filenames: Array of filenames
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        DataFrame with predictions
    """
    preds_df = pd.DataFrame({
        'filename': filenames,
        'y_true': y_true,
        'y_pred': y_pred,
        'abs_error': np.abs(y_true - y_pred),
        'relative_error': np.abs(y_true - y_pred) / y_true * 100
    })
    
    # Sort by absolute error (largest errors first)
    preds_df = preds_df.sort_values('abs_error', ascending=False).reset_index(drop=True)
    
    return preds_df


# =====================================================
# SINGLE EXPERIMENT FUNCTION
# =====================================================

def run_meta_experiment(train_folds, test_fold, exp_name):
    """
    Run a complete metadata-only experiment.
    
    Args:
        train_folds: List of fold numbers to use for training
        test_fold: Fold number to use for testing
        exp_name: Name of the experiment
    
    Returns:
        Tuple of (metrics_dict, predictions_df)
    """
    print(f"\n{'='*60}")
    print(f"[START] Metadata-only experiment: {exp_name}")
    print(f"{'='*60}")
    print(f"Train folds: {train_folds}")
    print(f"Test fold: {test_fold}")
    
    # Start timing
    total_start_time = time.time()
    
    try:
        # -----------------------------
        # Load data
        # -----------------------------
        (X_train, y_train), (X_val, y_val), scaler = \
            load_meta_train_val_from_folds(train_folds)
        
        X_test, y_test, filenames = load_meta_test_fold(test_fold, scaler)
        
        print(f"\n[DATA] Loaded successfully")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val:   {len(X_val)} samples")
        print(f"  Test:  {len(X_test)} samples")
        
        # Verify data integrity
        verify_data_integrity(X_train, y_train, "Training")
        verify_data_integrity(X_test, y_test, "Test")
        
        # -----------------------------
        # Train model
        # -----------------------------
        model, train_time = train_xgboost_model(X_train, y_train, X_val, y_val)
        
        # -----------------------------
        # Evaluate model
        # -----------------------------
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        
        # -----------------------------
        # Create predictions DataFrame
        # -----------------------------
        preds_df = create_predictions_dataframe(filenames, y_test, y_pred)
        
        # -----------------------------
        # Calculate total time
        # -----------------------------
        total_time = time.time() - total_start_time
        
        # Add experiment info to metrics
        metrics.update({
            'experiment': exp_name,
            'test_fold': test_fold,
            'train_folds': str(train_folds),
            'train_time_sec': train_time,
            'total_time_sec': total_time,
            'total_time_min': total_time / 60,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test)
        })
        
        print(f"\n{'='*60}")
        print(f"[RESULT] {exp_name}")
        print(f"{'='*60}")
        print(f"MAE:  {metrics['mae']:.4f} lb")
        print(f"RMSE: {metrics['rmse']:.4f} lb")
        print(f"R²:   {metrics['r2']:.4f}")
        print(f"Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"{'='*60}")
        
        return metrics, preds_df
        
    except Exception as e:
        print(f"\n❌ ERROR in {exp_name}: {str(e)}")
        raise


# =====================================================
# FEATURE IMPORTANCE ANALYSIS
# =====================================================

def analyze_feature_importance(model, feature_names):
    """
    Analyze and display feature importance from trained XGBoost model.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature importance
    """
    print("\n[ANALYSIS] Feature Importance:")
    
    importance = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(importance_df.to_string(index=False))
    
    return importance_df


# =====================================================
# EXAMPLE USAGE
# =====================================================

if __name__ == "__main__":
    from config_training import set_random_seeds
    
    # Set random seeds
    set_random_seeds()
    
    print("="*60)
    print("XGBOOST METADATA-ONLY MODEL - SINGLE EXPERIMENT TEST")
    print("="*60)
    
    # Run single experiment
    train_folds = [1, 2, 3, 4]
    test_fold = 5
    exp_name = "meta_test_fold_5"
    
    try:
        metrics, preds_df = run_meta_experiment(train_folds, test_fold, exp_name)
        
        print("\n[PREDICTIONS] Top 10 largest errors:")
        print(preds_df.head(10)[['filename', 'y_true', 'y_pred', 'abs_error']])
        
        print("\n✅ Single experiment test completed successfully")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
