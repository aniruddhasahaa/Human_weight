"""
5-Fold Cross-Validation Pipeline for XGBoost Metadata-Only Model
Run complete 5-fold cross-validation and save results.
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
from train_xgboost_meta import run_meta_experiment

# =====================================================
# MAIN CROSS-VALIDATION PIPELINE
# =====================================================

def run_5fold_cross_validation():
    """
    Run complete 5-fold cross-validation for metadata-only model.
    """
    
    print("="*80)
    print("METADATA-ONLY 5-FOLD CROSS-VALIDATION")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)
    
    # Track results
    results = []
    all_predictions = []
    failed = []
    
    pipeline_start_time = time.time()
    
    # --------------------------------------------------
    # Run experiments
    # --------------------------------------------------
    for idx, exp_config in enumerate(CV_EXPERIMENTS, 1):
        
        train_folds = exp_config['train']
        test_fold = exp_config['test']
        exp_name = f"meta_test_fold_{test_fold}"
        
        print("\n" + "-"*80)
        print(f"EXPERIMENT {idx}/{len(CV_EXPERIMENTS)} → {exp_name}")
        print(f"Train folds: {train_folds} | Test fold: {test_fold}")
        print("-"*80)
        
        exp_start_time = time.time()
        
        try:
            # Run experiment
            metrics, preds = run_meta_experiment(
                train_folds=train_folds,
                test_fold=test_fold,
                exp_name=exp_name
            )
            
            # Calculate elapsed time
            elapsed = time.time() - exp_start_time
            metrics['elapsed_time_sec'] = elapsed
            metrics['elapsed_time_min'] = elapsed / 60
            
            # Store results
            results.append(metrics)
            all_predictions.append(preds)
            
            print(f"\n✅ {exp_name} completed in {elapsed/60:.2f} minutes")
            print(f"   MAE: {metrics['mae']:.4f} lb")
            print(f"   RMSE: {metrics['rmse']:.4f} lb")
            print(f"   R²: {metrics['r2']:.4f}")
            
        except Exception as err:
            elapsed = time.time() - exp_start_time
            
            print(f"\n❌ {exp_name} FAILED after {elapsed/60:.2f} minutes")
            print(f"   Error: {err}")
            
            failed.append({
                'experiment': exp_name,
                'test_fold': test_fold,
                'error': str(err),
                'elapsed_time_sec': elapsed
            })
            
            # Print traceback for debugging
            import traceback
            traceback.print_exc()
    
    # --------------------------------------------------
    # Calculate total pipeline time
    # --------------------------------------------------
    pipeline_time = time.time() - pipeline_start_time
    
    # --------------------------------------------------
    # Summary statistics
    # --------------------------------------------------
    print("\n" + "="*80)
    print("METADATA-ONLY CROSS-VALIDATION SUMMARY")
    print("="*80)
    
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        
        # Display results table
        print("\nResults by Fold:")
        display_cols = ['test_fold', 'mae', 'rmse', 'r2', 'elapsed_time_min']
        print(results_df[display_cols].to_string(index=False))
        
        # Calculate statistics
        avg_mae = results_df['mae'].mean()
        std_mae = results_df['mae'].std()
        avg_rmse = results_df['rmse'].mean()
        std_rmse = results_df['rmse'].std()
        avg_r2 = results_df['r2'].mean()
        std_r2 = results_df['r2'].std()
        
        print("\n" + "-"*80)
        print("OVERALL STATISTICS")
        print("-"*80)
        print(f"MAE:  {avg_mae:.4f} ± {std_mae:.4f} lb")
        print(f"RMSE: {avg_rmse:.4f} ± {std_rmse:.4f} lb")
        print(f"R²:   {avg_r2:.4f} ± {std_r2:.4f}")
        print(f"\nTotal pipeline time: {pipeline_time/60:.2f} minutes")
        print(f"Average time per fold: {pipeline_time/len(results)/60:.2f} minutes")
        
        # Best and worst folds
        best_fold = results_df.loc[results_df['mae'].idxmin()]
        worst_fold = results_df.loc[results_df['mae'].idxmax()]
        
        print("\n" + "-"*80)
        print("BEST FOLD")
        print("-"*80)
        print(f"Test Fold: {best_fold['test_fold']}")
        print(f"MAE: {best_fold['mae']:.4f} lb")
        print(f"RMSE: {best_fold['rmse']:.4f} lb")
        print(f"R²: {best_fold['r2']:.4f}")
        
        print("\n" + "-"*80)
        print("WORST FOLD")
        print("-"*80)
        print(f"Test Fold: {worst_fold['test_fold']}")
        print(f"MAE: {worst_fold['mae']:.4f} lb")
        print(f"RMSE: {worst_fold['rmse']:.4f} lb")
        print(f"R²: {worst_fold['r2']:.4f}")
        
    else:
        print("❌ No successful experiments!")
        return
    
    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define output file paths
    results_csv = os.path.join(OUTPUT_DIR, "meta_only_results.csv")
    results_json = os.path.join(OUTPUT_DIR, "meta_only_results.json")
    preds_csv = os.path.join(OUTPUT_DIR, "meta_only_predictions.csv")
    summary_txt = os.path.join(OUTPUT_DIR, "meta_only_summary.txt")
    
    # Save results CSV
    if len(results) > 0:
        results_df.to_csv(results_csv, index=False)
        print(f"✓ Results CSV saved: {results_csv}")
    
    # Save results JSON
    if len(results) > 0:
        with open(results_json, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"✓ Results JSON saved: {results_json}")
    
    # Save predictions
    if len(all_predictions) > 0:
        combined_preds = pd.concat(all_predictions, ignore_index=True)
        combined_preds.to_csv(preds_csv, index=False)
        print(f"✓ Predictions saved: {preds_csv}")
        print(f"  Total predictions: {len(combined_preds)}")
    
    # Save summary text file
    with open(summary_txt, 'w') as f:
        f.write("="*80 + "\n")
        f.write("METADATA-ONLY 5-FOLD CROSS-VALIDATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {pipeline_time/60:.2f} minutes\n\n")
        
        if len(results) > 0:
            f.write("Results by Fold:\n")
            f.write(results_df[display_cols].to_string(index=False) + "\n\n")
            
            f.write("Overall Statistics:\n")
            f.write(f"MAE:  {avg_mae:.4f} ± {std_mae:.4f} lb\n")
            f.write(f"RMSE: {avg_rmse:.4f} ± {std_rmse:.4f} lb\n")
            f.write(f"R²:   {avg_r2:.4f} ± {std_r2:.4f}\n\n")
            
            f.write("Best Fold:\n")
            f.write(f"  Test Fold: {best_fold['test_fold']}\n")
            f.write(f"  MAE: {best_fold['mae']:.4f} lb\n")
            f.write(f"  RMSE: {best_fold['rmse']:.4f} lb\n")
            f.write(f"  R²: {best_fold['r2']:.4f}\n\n")
            
            f.write("Worst Fold:\n")
            f.write(f"  Test Fold: {worst_fold['test_fold']}\n")
            f.write(f"  MAE: {worst_fold['mae']:.4f} lb\n")
            f.write(f"  RMSE: {worst_fold['rmse']:.4f} lb\n")
            f.write(f"  R²: {worst_fold['r2']:.4f}\n")
    
    print(f"✓ Summary text saved: {summary_txt}")
    
    # Report failed experiments
    if len(failed) > 0:
        failed_json = os.path.join(OUTPUT_DIR, "meta_only_failed.json")
        with open(failed_json, 'w') as f:
            json.dump(failed, f, indent=4)
        
        print(f"\n⚠️ Failed experiments saved: {failed_json}")
        print("Failed experiments:")
        for f in failed:
            print(f"  • {f['experiment']} → {f['error']}")
    
    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"✓ Successful experiments: {len(results)}/{len(CV_EXPERIMENTS)}")
    print(f"✓ Total runtime: {pipeline_time/60:.2f} minutes")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    
    if len(results) == len(CV_EXPERIMENTS):
        print("\n✅ All experiments completed successfully!")
    else:
        print(f"\n⚠️ {len(failed)} experiment(s) failed")
    
    print("="*80 + "\n")


# =====================================================
# MAIN EXECUTION
# =====================================================

def main():
    """Main function to run the complete pipeline"""
    
    # Set random seeds for reproducibility
    set_random_seeds()
    print("✓ Random seeds initialized\n")
    
    # Configure GPU (if available)
    configure_gpu()
    
    # Run 5-fold cross-validation
    run_5fold_cross_validation()


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
