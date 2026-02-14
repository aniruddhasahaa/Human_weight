"""
Results Analysis and Visualization
Analyze and visualize results from XGBoost metadata-only experiments.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config_training import OUTPUT_DIR

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# =====================================================
# LOAD RESULTS
# =====================================================

def load_results():
    """Load results from saved files"""
    
    results_csv = os.path.join(OUTPUT_DIR, "meta_only_results.csv")
    preds_csv = os.path.join(OUTPUT_DIR, "meta_only_predictions.csv")
    
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"Results file not found: {results_csv}")
    
    results_df = pd.read_csv(results_csv)
    
    preds_df = None
    if os.path.exists(preds_csv):
        preds_df = pd.read_csv(preds_csv)
    
    return results_df, preds_df


# =====================================================
# ANALYSIS FUNCTIONS
# =====================================================

def print_summary_statistics(results_df):
    """Print summary statistics from results"""
    
    print("="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Overall metrics
    metrics = ['mae', 'rmse', 'r2']
    
    print("\nOverall Performance:")
    for metric in metrics:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        min_val = results_df[metric].min()
        max_val = results_df[metric].max()
        
        print(f"\n{metric.upper()}:")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Std:  {std_val:.4f}")
        print(f"  Min:  {min_val:.4f}")
        print(f"  Max:  {max_val:.4f}")
    
    # Per-fold results
    print("\n" + "-"*60)
    print("Results by Fold:")
    print("-"*60)
    display_cols = ['test_fold', 'mae', 'rmse', 'r2']
    print(results_df[display_cols].to_string(index=False))


def analyze_predictions(preds_df):
    """Analyze prediction errors"""
    
    if preds_df is None:
        print("⚠️ No predictions data available")
        return
    
    print("\n" + "="*60)
    print("PREDICTION ANALYSIS")
    print("="*60)
    
    # Error statistics
    print("\nError Statistics:")
    print(f"Mean Absolute Error: {preds_df['abs_error'].mean():.4f} lb")
    print(f"Median Absolute Error: {preds_df['abs_error'].median():.4f} lb")
    print(f"Std Error: {preds_df['abs_error'].std():.4f} lb")
    print(f"Max Error: {preds_df['abs_error'].max():.4f} lb")
    
    # Percentile analysis
    print("\nError Percentiles:")
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(preds_df['abs_error'], p)
        print(f"  {p}th percentile: {val:.4f} lb")
    
    # Error ranges
    print("\nError Distribution:")
    ranges = [(0, 5), (5, 10), (10, 15), (15, 20), (20, float('inf'))]
    for low, high in ranges:
        count = ((preds_df['abs_error'] >= low) & (preds_df['abs_error'] < high)).sum()
        pct = count / len(preds_df) * 100
        label = f"{low}-{high}" if high != float('inf') else f">{low}"
        print(f"  {label:10s} lb: {count:5d} ({pct:5.1f}%)")
    
    # Top 10 worst predictions
    print("\nTop 10 Largest Errors:")
    worst_preds = preds_df.nlargest(10, 'abs_error')
    print(worst_preds[['filename', 'y_true', 'y_pred', 'abs_error']].to_string(index=False))


# =====================================================
# VISUALIZATION FUNCTIONS
# =====================================================

def plot_fold_comparison(results_df, save_path=None):
    """Plot comparison of metrics across folds"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = [
        ('mae', 'Mean Absolute Error (lb)', 'Reds_r'),
        ('rmse', 'Root Mean Squared Error (lb)', 'Oranges_r'),
        ('r2', 'R² Score', 'Greens')
    ]
    
    for ax, (metric, title, cmap) in zip(axes, metrics):
        values = results_df[metric].values
        folds = results_df['test_fold'].values
        
        bars = ax.bar(folds, values, color=sns.color_palette(cmap, len(folds)))
        ax.axhline(values.mean(), color='black', linestyle='--', 
                   label=f'Mean: {values.mean():.4f}')
        
        ax.set_xlabel('Test Fold')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (fold, val) in enumerate(zip(folds, values)):
            ax.text(fold, val, f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Fold comparison plot saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_prediction_scatter(preds_df, save_path=None):
    """Plot scatter plot of predictions vs actual"""
    
    if preds_df is None:
        print("⚠️ No predictions data available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Scatter plot
    ax.scatter(preds_df['y_true'], preds_df['y_pred'], 
               alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(preds_df['y_true'].min(), preds_df['y_pred'].min())
    max_val = max(preds_df['y_true'].max(), preds_df['y_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfect Prediction')
    
    # Labels and title
    ax.set_xlabel('Actual Weight (lb)', fontsize=12)
    ax.set_ylabel('Predicted Weight (lb)', fontsize=12)
    ax.set_title('Actual vs Predicted Weight', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add metrics text
    mae = preds_df['abs_error'].mean()
    rmse = np.sqrt((preds_df['abs_error']**2).mean())
    r2 = 1 - ((preds_df['y_true'] - preds_df['y_pred'])**2).sum() / \
             ((preds_df['y_true'] - preds_df['y_true'].mean())**2).sum()
    
    textstr = f'MAE: {mae:.2f} lb\nRMSE: {rmse:.2f} lb\nR²: {r2:.4f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5), fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Prediction scatter plot saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_error_distribution(preds_df, save_path=None):
    """Plot distribution of prediction errors"""
    
    if preds_df is None:
        print("⚠️ No predictions data available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(preds_df['abs_error'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(preds_df['abs_error'].mean(), color='red', 
                    linestyle='--', linewidth=2, label=f"Mean: {preds_df['abs_error'].mean():.2f} lb")
    axes[0].axvline(preds_df['abs_error'].median(), color='green', 
                    linestyle='--', linewidth=2, label=f"Median: {preds_df['abs_error'].median():.2f} lb")
    axes[0].set_xlabel('Absolute Error (lb)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Absolute Errors')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    axes[1].boxplot(preds_df['abs_error'], vert=True)
    axes[1].set_ylabel('Absolute Error (lb)')
    axes[1].set_title('Box Plot of Absolute Errors')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Error distribution plot saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


# =====================================================
# MAIN ANALYSIS FUNCTION
# =====================================================

def analyze_results(save_plots=True):
    """Run complete analysis on results"""
    
    print("="*60)
    print("XGBOOST METADATA-ONLY RESULTS ANALYSIS")
    print("="*60)
    
    try:
        # Load results
        print("\n[LOAD] Loading results...")
        results_df, preds_df = load_results()
        print(f"✓ Results loaded: {len(results_df)} experiments")
        if preds_df is not None:
            print(f"✓ Predictions loaded: {len(preds_df)} samples")
        
        # Print summary statistics
        print_summary_statistics(results_df)
        
        # Analyze predictions
        if preds_df is not None:
            analyze_predictions(preds_df)
        
        # Generate plots
        if save_plots:
            print("\n[PLOT] Generating visualizations...")
            
            plot_fold_comparison(results_df, 
                                os.path.join(OUTPUT_DIR, "fold_comparison.png"))
            
            if preds_df is not None:
                plot_prediction_scatter(preds_df,
                                       os.path.join(OUTPUT_DIR, "prediction_scatter.png"))
                plot_error_distribution(preds_df,
                                       os.path.join(OUTPUT_DIR, "error_distribution.png"))
            
            print("✓ All plots saved")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    analyze_results(save_plots=True)
