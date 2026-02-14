"""
CNN Image-Only Training Script
Train CNN models for weight estimation using only images.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger
)

# Import utilities
from config_training import OUTPUT_DIR, BATCH_SIZE, EPOCHS, PATIENCE
from image_data_loader import (
    load_train_val_from_folds_image_only,
    load_test_fold_image_only
)
from cnn_models import build_efficientnetb0_weight_regressor

# =====================================================
# TRAINING CALLBACKS
# =====================================================

def get_training_callbacks(exp_dir, model_name='best_model'):
    """
    Create training callbacks for model training.
    
    Args:
        exp_dir: Experiment directory to save files
        model_name: Name for saved model file
    
    Returns:
        List of Keras callbacks
    """
    
    os.makedirs(exp_dir, exist_ok=True)
    
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=os.path.join(exp_dir, f"{model_name}.keras"),
            monitor='val_mae',
            save_best_only=True,
            mode='min',
            verbose=1,
            save_format='keras'
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_mae',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=6,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),
        
        # Log training history
        CSVLogger(
            filename=os.path.join(exp_dir, 'training_log.csv'),
            append=False
        )
    ]
    
    return callbacks


# =====================================================
# MAIN EXPERIMENT FUNCTION
# =====================================================

def run_image_experiment(train_folds, test_fold, exp_name, model_builder=None, lr=1e-4):
    """
    Run a complete image-only CNN experiment.
    
    Args:
        train_folds: List of fold numbers for training
        test_fold: Fold number for testing
        exp_name: Name of the experiment
        model_builder: Function to build model (default: EfficientNet-B0)
        lr: Learning rate
    
    Returns:
        Tuple of (metrics_dict, predictions_df)
    """
    
    print(f"\n{'='*70}")
    print(f"[START] Experiment: {exp_name}")
    print(f"[MODEL] Image-only CNN Regression")
    print(f"{'='*70}")
    print(f"Train folds: {train_folds}")
    print(f"Test fold: {test_fold}")
    
    total_start_time = time.time()
    
    try:
        # =================================================
        # Load Training and Validation Data
        # =================================================
        print("\n" + "="*70)
        print("LOADING TRAINING DATA")
        print("="*70)
        
        (X_train, y_train), (X_val, y_val) = \
            load_train_val_from_folds_image_only(train_folds)
        
        print(f"\n[DATA] Training set: {len(X_train)} samples")
        print(f"[DATA] Validation set: {len(X_val)} samples")
        
        # =================================================
        # Load Test Data
        # =================================================
        print("\n" + "="*70)
        print("LOADING TEST DATA")
        print("="*70)
        
        X_test, y_test, filenames = load_test_fold_image_only(test_fold)
        
        print(f"\n[DATA] Test set: {len(X_test)} samples")
        
        # =================================================
        # Build Model
        # =================================================
        print("\n" + "="*70)
        print("BUILDING MODEL")
        print("="*70)
        
        if model_builder is None:
            model_builder = build_efficientnetb0_weight_regressor
        
        model = model_builder(img_shape=(224, 224, 3), lr=lr)
        
        # Print model summary
        print("\n[MODEL] Architecture summary:")
        model.summary(print_fn=lambda x: print(x) if len(x) < 100 else None)
        
        # =================================================
        # Setup Callbacks
        # =================================================
        exp_dir = os.path.join(OUTPUT_DIR, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        callbacks = get_training_callbacks(exp_dir, model_name='best_model')
        
        print(f"\n[SETUP] Experiment directory: {exp_dir}")
        print(f"[SETUP] Callbacks configured: {len(callbacks)} callbacks")
        
        # =================================================
        # Train Model
        # =================================================
        print("\n" + "="*70)
        print("TRAINING MODEL")
        print("="*70)
        print(f"Epochs: {EPOCHS}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Learning rate: {lr}")
        
        train_start_time = time.time()
        
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        train_time = time.time() - train_start_time
        
        print(f"\n✓ Training completed in {train_time/60:.2f} minutes")
        
        # =================================================
        # Evaluate Model
        # =================================================
        print("\n" + "="*70)
        print("EVALUATING MODEL")
        print("="*70)
        
        # Make predictions
        y_pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1).reshape(-1)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Additional metrics
        abs_errors = np.abs(y_test - y_pred)
        median_ae = np.median(abs_errors)
        max_error = np.max(abs_errors)
        
        print("\n" + "="*70)
        print(f"RESULTS — {exp_name}")
        print("="*70)
        print(f"MAE:        {mae:.4f} lb")
        print(f"RMSE:       {rmse:.4f} lb")
        print(f"R²:         {r2:.4f}")
        print(f"Median AE:  {median_ae:.4f} lb")
        print(f"Max Error:  {max_error:.4f} lb")
        print("="*70)
        
        # =================================================
        # Save Predictions
        # =================================================
        preds_df = pd.DataFrame({
            'filename': filenames,
            'y_true': y_test,
            'y_pred': y_pred,
            'abs_error': abs_errors,
            'relative_error': (abs_errors / y_test) * 100
        })
        
        # Sort by error
        preds_df = preds_df.sort_values('abs_error', ascending=False).reset_index(drop=True)
        
        preds_csv = os.path.join(exp_dir, 'predictions.csv')
        preds_df.to_csv(preds_csv, index=False)
        print(f"\n[SAVE] Predictions saved: {preds_csv}")
        
        # =================================================
        # Save Training History
        # =================================================
        history_df = pd.DataFrame(history.history)
        history_csv = os.path.join(exp_dir, 'history.csv')
        history_df.to_csv(history_csv, index=False)
        print(f"[SAVE] Training history saved: {history_csv}")
        
        # =================================================
        # Calculate Total Time
        # =================================================
        total_time = time.time() - total_start_time
        
        # =================================================
        # Compile Metrics
        # =================================================
        metrics = {
            'experiment': exp_name,
            'test_fold': test_fold,
            'train_folds': str(train_folds),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'median_ae': float(median_ae),
            'max_error': float(max_error),
            'train_time_sec': train_time,
            'train_time_min': train_time / 60,
            'total_time_sec': total_time,
            'total_time_min': total_time / 60,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'epochs_trained': len(history.history['loss']),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1])
        }
        
        print(f"\n[TIME] Total experiment time: {total_time/60:.2f} minutes")
        
        return metrics, preds_df
        
    except Exception as e:
        print(f"\n❌ ERROR in {exp_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


# =====================================================
# EXAMPLE USAGE
# =====================================================

if __name__ == "__main__":
    from config_training import set_random_seeds, configure_gpu
    
    # Setup
    set_random_seeds()
    configure_gpu()
    
    print("="*70)
    print("CNN IMAGE-ONLY MODEL - SINGLE EXPERIMENT TEST")
    print("="*70)
    
    # Run single experiment
    train_folds = [1, 2, 3, 4]
    test_fold = 5
    exp_name = "test_fold_5"
    
    try:
        metrics, preds_df = run_image_experiment(
            train_folds=train_folds,
            test_fold=test_fold,
            exp_name=exp_name,
            lr=1e-4
        )
        
        print("\n[PREDICTIONS] Top 10 largest errors:")
        print(preds_df.head(10)[['filename', 'y_true', 'y_pred', 'abs_error']])
        
        print("\n✅ Single experiment test completed successfully")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
