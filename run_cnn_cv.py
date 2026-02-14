"""
5-Fold Cross-Validation Pipeline for CNN Image-Only Model
Run complete 5-fold cross-validation for weight estimation using CNN.
"""

import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime

# Import configuration and utilities
from config_training import (
    CV_EXPERIMENTS,
    OUTPUT_DIR,
    set_random_seeds,
    configure_gpu
)
from train_cnn_image import run_image_experiment

# =====================================================
# MAIN CROSS-VALIDATION PIPELINE
# =====================================================

def run_5fold_cnn_cross_validation(model_name='efficientnet', lr=1e-4):
    """
    Run complete 5-fold cross-validation for CNN image-only model.
    
    Args:
        model_name: Model architecture to use ('efficientnet', 'mobilenet', 'resnet', 'custom')
        lr: Learning rate
    """
    
    print("="*80)
    print("CNN IMAGE-ONLY 5-FOLD CROSS-VALIDATION")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {model_name}")
    print(f"Learning rate: {lr}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)
    
    # Track results
    summary = []
    all_predictions = []
    failed_experiments = []
    
    pipeline_start_time = time.time()
    
    # --------------------------------------------------
    # Run experiments
    # --------------------------------------------------
    for idx, exp_config in enumerate(CV_EXPERIMENTS, 1):
        
        train_folds = exp_config['train']
        test_fold = exp_config['test']
        exp_name = f"image_test_fold_{test_fold}"
        
        print("\n" + "="*80)
        print(f"EXPERIMENT {idx}/{len(CV_EXPERIMENTS)} → {exp_name}")
        print(f"Train folds: {train_folds} | Test fold: {test_fold}")
        print("="*80)
        
        exp_start_time = time.time()
        
        try:
            # Import model builder based on model_name
            if model_name.lower() == 'mobilenet':
                from cnn_models import build_mobilenetv2_weight_regressor as model_builder
            elif model_name.lower() == 'resnet':
                from cnn_models import build_resnet50_weight_regressor as model_builder
            elif model_name.lower() == 'custom':
                from cnn_models import build_custom_cnn_weight_regressor as model_builder
            else:  # default to efficientnet
                from cnn_models import build_efficientnetb0_weight_regressor as model_builder
            
            # Run experiment
            metrics, preds = run_image_experiment(
                train_folds=train_folds,
                test_fold=test_fold,
                exp_name=exp_name,
                model_builder=model_builder,
                lr=lr
            )
            
            # Calculate elapsed time
            elapsed = time.time() - exp_start_time
            metrics['elapsed_minutes'] = elapsed / 60
            
            # Store results
            summary.append({
                'experiment': exp_name,
                'test_fold': test_fold,
                'train_folds': train_folds,
                'metrics': metrics,
                'status': 'success'
            })
            
            all_predictions.append(preds)
            
            print(f"\n✅ {exp_name} completed in {elapsed/60:.2f} minutes")
            print(f"   MAE:  {metrics['mae']:.4f} lb")
            print(f"   RMSE: {metrics['rmse']:.4f} lb")
            print(f"   R²:   {metrics['r2']:.4f}")
            
        except Exception as err:
            elapsed = time.time() - exp_start_time
            
            print(f"\n❌ {exp_name} FAILED after {elapsed/60:.2f} minutes")
            print(f"   Error: {err}")
            
            failed_experiments.append({
                'experiment': exp_name,
                'test_fold': test_fold,
                'error': str(err),
                'elapsed_minutes': elapsed / 60
            })
            
            summary.append({
                'experiment': exp_name,
                'test_fold': test_fold,
                'train_folds': train_folds,
                'metrics': None,
                'status': 'failed',
                'error': str(err)
            })
            
            # Print traceback for debugging
            import traceback
            traceback.print_exc()
    
    # --------------------------------------------------
    # Calculate total pipeline time
    # --------------------------------------------------
    pipeline_time = time.time() - pipeline_start_time
    
    # --------------------------------------------------
    # Aggregate Results
    # --------------------------------------------------
    print("\n" + "="*80)
    print("CNN IMAGE-ONLY CROSS-VALIDATION SUMMARY")
    print("="*80)
    
    successful = [s for s in summary if s['status'] == 'success']
    
    if successful:
        # Extract metrics
        mae_vals = [s['metrics']['mae'] for s in successful]
        rmse_vals = [s['metrics']['rmse'] for s in successful]
        r2_vals = [s['metrics']['r2'] for s in successful]
        
        # Calculate statistics
        avg_metrics = {
            'avg_mae': float(np.mean(mae_vals)),
            'std_mae': float(np.std(mae_vals)),
            'min_mae': float(np.min(mae_vals)),
            'max_mae': float(np.max(mae_vals)),
            'avg_rmse': float(np.mean(rmse_vals)),
            'std_rmse': float(np.std(rmse_vals)),
            'avg_r2': float(np.mean(r2_vals)),
            'std_r2': float(np.std(r2_vals)),
            'successful_folds': len(successful),
            'failed_folds': len(failed_experiments),
            'total_time_hours': pipeline_time / 3600,
            'avg_time_per_fold_min': (pipeline_time / len(successful)) / 60 if successful else 0
        }
        
        print("\nOverall Statistics:")
        print(f"MAE:  {avg_metrics['avg_mae']:.4f} ± {avg_metrics['std_mae']:.4f} lb")
        print(f"RMSE: {avg_metrics['avg_rmse']:.4f} ± {avg_metrics['std_rmse']:.4f} lb")
        print(f"R²:   {avg_metrics['avg_r2']:.4f} ± {avg_metrics['std_r2']:.4f}")
        print(f"MAE Range: [{avg_metrics['min_mae']:.4f}, {avg_metrics['max_mae']:.4f}] lb")
        
        print("\nPer-Fold Results:")
        for s in successful:
            m = s['metrics']
            print(f"  Fold {s['test_fold']}: "
                  f"MAE={m['mae']:.4f} lb, "
                  f"RMSE={m['rmse']:.4f} lb, "
                  f"R²={m['r2']:.4f}")
        
        print(f"\nTotal pipeline time: {pipeline_time/3600:.2f} hours")
        print(f"Average time per fold: {avg_metrics['avg_time_per_fold_min']:.2f} minutes")
        
        # Best and worst folds
        best_idx = np.argmin(mae_vals)
        worst_idx = np.argmax(mae_vals)
        
        best = successful[best_idx]
        worst = successful[worst_idx]
        
        print("\n" + "-"*80)
        print("Best Fold:")
        print(f"  Test Fold: {best['test_fold']}")
        print(f"  MAE: {best['metrics']['mae']:.4f} lb")
        print(f"  RMSE: {best['metrics']['rmse']:.4f} lb")
        print(f"  R²: {best['metrics']['r2']:.4f}")
        
        print("\nWorst Fold:")
        print(f"  Test Fold: {worst['test_fold']}")
        print(f"  MAE: {worst['metrics']['mae']:.4f} lb")
        print(f"  RMSE: {worst['metrics']['rmse']:.4f} lb")
        print(f"  R²: {worst['metrics']['r2']:.4f}")
        
    else:
        avg_metrics = None
        print("❌ All experiments failed!")
        return
    
    # --------------------------------------------------
    # Save Results
    # --------------------------------------------------
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define output file paths
    summary_json = os.path.join(OUTPUT_DIR, "cnn_experiments_summary.json")
    avg_metrics_json = os.path.join(OUTPUT_DIR, "cnn_average_metrics.json")
    results_csv = os.path.join(OUTPUT_DIR, "cnn_results_summary.csv")
    all_preds_csv = os.path.join(OUTPUT_DIR, "cnn_all_predictions.csv")
    summary_txt = os.path.join(OUTPUT_DIR, "cnn_summary.txt")
    
    # Save summary JSON
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"✓ Summary saved: {summary_json}")
    
    # Save average metrics
    if avg_metrics is not None:
        with open(avg_metrics_json, 'w') as f:
            json.dump(avg_metrics, f, indent=4)
        print(f"✓ Average metrics saved: {avg_metrics_json}")
    
    # Save results CSV
    if successful:
        results_df = pd.DataFrame([
            {
                'experiment': s['experiment'],
                'test_fold': s['test_fold'],
                'mae': s['metrics']['mae'],
                'rmse': s['metrics']['rmse'],
                'r2': s['metrics']['r2'],
                'median_ae': s['metrics']['median_ae'],
                'max_error': s['metrics']['max_error'],
                'train_time_min': s['metrics']['train_time_min'],
                'total_time_min': s['metrics']['total_time_min'],
                'epochs_trained': s['metrics']['epochs_trained']
            }
            for s in successful
        ])
        results_df.to_csv(results_csv, index=False)
        print(f"✓ Results CSV saved: {results_csv}")
    
    # Save all predictions
    if all_predictions:
        combined_preds = pd.concat(all_predictions, ignore_index=True)
        combined_preds.to_csv(all_preds_csv, index=False)
        print(f"✓ All predictions saved: {all_preds_csv}")
        print(f"  Total predictions: {len(combined_preds)}")
    
    # Save summary text file
    with open(summary_txt, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CNN IMAGE-ONLY 5-FOLD CROSS-VALIDATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Total time: {pipeline_time/3600:.2f} hours\n\n")
        
        if avg_metrics is not None:
            f.write("Overall Statistics:\n")
            f.write(f"MAE:  {avg_metrics['avg_mae']:.4f} ± {avg_metrics['std_mae']:.4f} lb\n")
            f.write(f"RMSE: {avg_metrics['avg_rmse']:.4f} ± {avg_metrics['std_rmse']:.4f} lb\n")
            f.write(f"R²:   {avg_metrics['avg_r2']:.4f} ± {avg_metrics['std_r2']:.4f}\n\n")
            
            f.write("Per-Fold Results:\n")
            for s in successful:
                m = s['metrics']
                f.write(f"  Fold {s['test_fold']}: MAE={m['mae']:.4f}, "
                       f"RMSE={m['rmse']:.4f}, R²={m['r2']:.4f}\n")
    
    print(f"✓ Summary text saved: {summary_txt}")
    
    # Report failed experiments
    if failed_experiments:
        failed_json = os.path.join(OUTPUT_DIR, "cnn_failed_experiments.json")
        with open(failed_json, 'w') as f:
            json.dump(failed_experiments, f, indent=4)
        
        print(f"\n⚠️ Failed experiments saved: {failed_json}")
        print("Failed experiments:")
        for f in failed_experiments:
            print(f"  • {f['experiment']} → {f['error']}")
    
    # --------------------------------------------------
    # Final Summary
    # --------------------------------------------------
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"✓ Successful experiments: {len(successful)}/{len(CV_EXPERIMENTS)}")
    print(f"✓ Total runtime: {pipeline_time/3600:.2f} hours")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    
    if len(successful) == len(CV_EXPERIMENTS):
        print("\n✅ All experiments completed successfully!")
    else:
        print(f"\n⚠️ {len(failed_experiments)} experiment(s) failed")
    
    print("="*80 + "\n")


# =====================================================
# MAIN EXECUTION
# =====================================================

def main():
    """Main function to run the complete pipeline"""
    
    # Set random seeds for reproducibility
    set_random_seeds()
    print("✓ Random seeds initialized\n")
    
    # Configure GPU
    configure_gpu()
    
    # Run 5-fold cross-validation
    # Change model_name to 'mobilenet', 'resnet', or 'custom' as needed
    run_5fold_cnn_cross_validation(
        model_name='efficientnet',  # or 'mobilenet', 'resnet', 'custom'
        lr=1e-4
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
