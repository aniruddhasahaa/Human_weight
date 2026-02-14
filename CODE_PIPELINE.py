"""
Master Pipeline for Human Weight Estimation Project
====================================================

This script orchestrates the complete end-to-end pipeline:
1. Data preprocessing and feature engineering
2. Fold creation and weight labeling
3. Data verification
4. Model training (XGBoost metadata-only)
5. Model training (CNN image-only)
6. Results analysis

Run this script to execute the entire pipeline automatically.
"""

import os
import sys
import time
from datetime import datetime
import subprocess

# =====================================================
# PIPELINE CONFIGURATION
# =====================================================

class PipelineConfig:
    """Configuration for the complete pipeline"""
    
    # Base directory
    BASE_DIR = "/content/drive/MyDrive/Human Weight Extimation"
    
    # Pipeline stages to run (set to False to skip a stage)
    RUN_PREPROCESSING = True
    RUN_FOLD_CREATION = True
    RUN_VERIFICATION = True
    RUN_XGBOOST_TRAINING = True
    RUN_CNN_TRAINING = True
    RUN_ANALYSIS = True
    
    # Model selection for CNN training
    CNN_MODEL = 'efficientnet'  # Options: 'efficientnet', 'mobilenet', 'resnet', 'custom'
    
    # Script paths (relative to current directory)
    SCRIPTS = {
        # Phase 1: Data Preprocessing
        'load_explore': '01_load_and_explore_data.py',
        'preprocess': '02_preprocess_images.py',
        'feature_eng': '03_feature_engineering.py',
        
        # Phase 2: Verification
        'verify_data': '04_verify_preprocessed_data.py',
        
        # Phase 3: Fold Creation
        'create_folds': '05_create_folds.py',
        'add_weights': '06_add_weights_to_folds.py',
        'verify_folds': '07_verify_folds.py',
        
        # Phase 4: XGBoost Training
        'train_xgboost': 'run_xgboost_cv.py',
        
        # Phase 5: CNN Training
        'train_cnn': 'run_cnn_cv.py',
        
        # Phase 6: Analysis
        'analyze': 'analyze_results.py'
    }


# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def print_header(title, char='='):
    """Print a formatted header"""
    width = 80
    print('\n' + char * width)
    print(title.center(width))
    print(char * width + '\n')


def print_phase_header(phase_num, phase_name):
    """Print a phase header"""
    print('\n' + '='*80)
    print(f'PHASE {phase_num}: {phase_name}'.center(80))
    print('='*80 + '\n')


def run_script(script_path, description, required=True):
    """
    Run a Python script and handle errors.
    
    Args:
        script_path: Path to the script
        description: Description of what the script does
        required: Whether the script is required for pipeline continuation
    
    Returns:
        True if successful, False otherwise
    """
    print(f'\n{"-"*80}')
    print(f'Running: {description}')
    print(f'Script: {script_path}')
    print(f'{"-"*80}\n')
    
    if not os.path.exists(script_path):
        print(f'❌ ERROR: Script not found: {script_path}')
        if required:
            print(f'   This is a required script. Pipeline cannot continue.')
            return False
        else:
            print(f'   This is optional. Skipping...')
            return True
    
    start_time = time.time()
    
    try:
        # Run the script as a subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f'\n✅ {description} completed successfully')
        print(f'   Time: {elapsed/60:.2f} minutes')
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f'\n❌ {description} FAILED')
        print(f'   Error code: {e.returncode}')
        print(f'   Time: {elapsed/60:.2f} minutes')
        
        if required:
            print(f'   This is a required script. Pipeline cannot continue.')
            return False
        else:
            print(f'   This is optional. Continuing...')
            return True
            
    except KeyboardInterrupt:
        print(f'\n\n⚠️ Script interrupted by user')
        raise
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f'\n❌ {description} FAILED with unexpected error')
        print(f'   Error: {str(e)}')
        print(f'   Time: {elapsed/60:.2f} minutes')
        
        if required:
            return False
        else:
            return True


def check_prerequisites():
    """Check if required directories and files exist"""
    print_header('CHECKING PREREQUISITES')
    
    checks = {
        'Base directory': PipelineConfig.BASE_DIR,
        'Input data directory': os.path.join(PipelineConfig.BASE_DIR, 'Input Data'),
    }
    
    all_passed = True
    
    for name, path in checks.items():
        if os.path.exists(path):
            print(f'✓ {name}: {path}')
        else:
            print(f'❌ {name} NOT FOUND: {path}')
            all_passed = False
    
    return all_passed


def display_pipeline_plan():
    """Display which stages will be run"""
    print_header('PIPELINE EXECUTION PLAN')
    
    stages = [
        ('Phase 1: Data Preprocessing', PipelineConfig.RUN_PREPROCESSING),
        ('Phase 2: Fold Creation', PipelineConfig.RUN_FOLD_CREATION),
        ('Phase 3: Data Verification', PipelineConfig.RUN_VERIFICATION),
        ('Phase 4: XGBoost Training', PipelineConfig.RUN_XGBOOST_TRAINING),
        ('Phase 5: CNN Training', PipelineConfig.RUN_CNN_TRAINING),
        ('Phase 6: Results Analysis', PipelineConfig.RUN_ANALYSIS),
    ]
    
    print('Stages to execute:')
    for stage, enabled in stages:
        status = '✓ ENABLED' if enabled else '✗ DISABLED'
        print(f'  {status:12s} - {stage}')
    
    if PipelineConfig.RUN_CNN_TRAINING:
        print(f'\nCNN Model: {PipelineConfig.CNN_MODEL}')


# =====================================================
# MAIN PIPELINE EXECUTION
# =====================================================

def run_complete_pipeline():
    """Execute the complete end-to-end pipeline"""
    
    # Start timing
    pipeline_start_time = time.time()
    start_datetime = datetime.now()
    
    print_header('HUMAN WEIGHT ESTIMATION - COMPLETE PIPELINE', '=')
    print(f'Start time: {start_datetime.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Base directory: {PipelineConfig.BASE_DIR}')
    
    # Check prerequisites
    if not check_prerequisites():
        print('\n❌ Prerequisites check failed. Please fix the issues above.')
        return False
    
    print('\n✅ Prerequisites check passed')
    
    # Display execution plan
    display_pipeline_plan()
    
    # Confirmation
    print('\n' + '='*80)
    response = input('Do you want to proceed with the pipeline? (yes/no): ')
    if response.lower() not in ['yes', 'y']:
        print('Pipeline cancelled by user.')
        return False
    
    # Track results
    results = {
        'start_time': start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        'stages_completed': [],
        'stages_failed': [],
        'stages_skipped': []
    }
    
    # ====================================================================
    # PHASE 1: DATA PREPROCESSING
    # ====================================================================
    if PipelineConfig.RUN_PREPROCESSING:
        print_phase_header(1, 'DATA PREPROCESSING')
        
        # Step 1.1: Load and explore
        if not run_script(
            PipelineConfig.SCRIPTS['load_explore'],
            'Load and explore dataset',
            required=True
        ):
            results['stages_failed'].append('Load and explore')
            return results
        results['stages_completed'].append('Load and explore')
        
        # Step 1.2: Preprocess images
        if not run_script(
            PipelineConfig.SCRIPTS['preprocess'],
            'Preprocess images and metadata',
            required=True
        ):
            results['stages_failed'].append('Preprocess images')
            return results
        results['stages_completed'].append('Preprocess images')
        
        # Step 1.3: Feature engineering
        if not run_script(
            PipelineConfig.SCRIPTS['feature_eng'],
            'Feature engineering',
            required=True
        ):
            results['stages_failed'].append('Feature engineering')
            return results
        results['stages_completed'].append('Feature engineering')
        
        print('\n✅ Phase 1 completed: Data preprocessing')
    else:
        print_phase_header(1, 'DATA PREPROCESSING (SKIPPED)')
        results['stages_skipped'].append('Data preprocessing')
    
    # ====================================================================
    # PHASE 2: DATA VERIFICATION
    # ====================================================================
    if PipelineConfig.RUN_VERIFICATION:
        print_phase_header(2, 'DATA VERIFICATION')
        
        if not run_script(
            PipelineConfig.SCRIPTS['verify_data'],
            'Verify preprocessed data',
            required=False
        ):
            results['stages_failed'].append('Verify data')
        else:
            results['stages_completed'].append('Verify data')
        
        print('\n✅ Phase 2 completed: Data verification')
    else:
        print_phase_header(2, 'DATA VERIFICATION (SKIPPED)')
        results['stages_skipped'].append('Data verification')
    
    # ====================================================================
    # PHASE 3: FOLD CREATION
    # ====================================================================
    if PipelineConfig.RUN_FOLD_CREATION:
        print_phase_header(3, 'FOLD CREATION')
        
        # Step 3.1: Create folds
        if not run_script(
            PipelineConfig.SCRIPTS['create_folds'],
            'Create 5-fold cross-validation splits',
            required=True
        ):
            results['stages_failed'].append('Create folds')
            return results
        results['stages_completed'].append('Create folds')
        
        # Step 3.2: Add weight labels
        if not run_script(
            PipelineConfig.SCRIPTS['add_weights'],
            'Add weight labels to folds',
            required=True
        ):
            results['stages_failed'].append('Add weights')
            return results
        results['stages_completed'].append('Add weights')
        
        # Step 3.3: Verify folds
        if not run_script(
            PipelineConfig.SCRIPTS['verify_folds'],
            'Verify all folds',
            required=False
        ):
            results['stages_failed'].append('Verify folds')
        else:
            results['stages_completed'].append('Verify folds')
        
        print('\n✅ Phase 3 completed: Fold creation')
    else:
        print_phase_header(3, 'FOLD CREATION (SKIPPED)')
        results['stages_skipped'].append('Fold creation')
    
    # ====================================================================
    # PHASE 4: XGBOOST TRAINING
    # ====================================================================
    if PipelineConfig.RUN_XGBOOST_TRAINING:
        print_phase_header(4, 'XGBOOST METADATA-ONLY TRAINING')
        
        if not run_script(
            PipelineConfig.SCRIPTS['train_xgboost'],
            'Train XGBoost metadata-only model (5-fold CV)',
            required=False
        ):
            results['stages_failed'].append('XGBoost training')
        else:
            results['stages_completed'].append('XGBoost training')
        
        print('\n✅ Phase 4 completed: XGBoost training')
    else:
        print_phase_header(4, 'XGBOOST TRAINING (SKIPPED)')
        results['stages_skipped'].append('XGBoost training')
    
    # ====================================================================
    # PHASE 5: CNN TRAINING
    # ====================================================================
    if PipelineConfig.RUN_CNN_TRAINING:
        print_phase_header(5, 'CNN IMAGE-ONLY TRAINING')
        print(f'Model: {PipelineConfig.CNN_MODEL}\n')
        
        # Note: Model selection is done inside run_cnn_cv.py
        # User should edit that file to change the model
        if not run_script(
            PipelineConfig.SCRIPTS['train_cnn'],
            f'Train CNN image-only model ({PipelineConfig.CNN_MODEL}, 5-fold CV)',
            required=False
        ):
            results['stages_failed'].append('CNN training')
        else:
            results['stages_completed'].append('CNN training')
        
        print('\n✅ Phase 5 completed: CNN training')
    else:
        print_phase_header(5, 'CNN TRAINING (SKIPPED)')
        results['stages_skipped'].append('CNN training')
    
    # ====================================================================
    # PHASE 6: RESULTS ANALYSIS
    # ====================================================================
    if PipelineConfig.RUN_ANALYSIS:
        print_phase_header(6, 'RESULTS ANALYSIS')
        
        if not run_script(
            PipelineConfig.SCRIPTS['analyze'],
            'Analyze and visualize results',
            required=False
        ):
            results['stages_failed'].append('Results analysis')
        else:
            results['stages_completed'].append('Results analysis')
        
        print('\n✅ Phase 6 completed: Results analysis')
    else:
        print_phase_header(6, 'RESULTS ANALYSIS (SKIPPED)')
        results['stages_skipped'].append('Results analysis')
    
    # ====================================================================
    # PIPELINE COMPLETION
    # ====================================================================
    pipeline_end_time = time.time()
    end_datetime = datetime.now()
    total_time = pipeline_end_time - pipeline_start_time
    
    results['end_time'] = end_datetime.strftime("%Y-%m-%d %H:%M:%S")
    results['total_time_hours'] = total_time / 3600
    
    print_header('PIPELINE COMPLETED', '=')
    print(f'Start time:  {results["start_time"]}')
    print(f'End time:    {results["end_time"]}')
    print(f'Total time:  {total_time/3600:.2f} hours ({total_time/60:.2f} minutes)')
    
    print(f'\n✅ Stages completed: {len(results["stages_completed"])}')
    for stage in results['stages_completed']:
        print(f'   ✓ {stage}')
    
    if results['stages_skipped']:
        print(f'\n⊘ Stages skipped: {len(results["stages_skipped"])}')
        for stage in results['stages_skipped']:
            print(f'   ⊘ {stage}')
    
    if results['stages_failed']:
        print(f'\n❌ Stages failed: {len(results["stages_failed"])}')
        for stage in results['stages_failed']:
            print(f'   ✗ {stage}')
    
    # Final status
    print('\n' + '='*80)
    if not results['stages_failed']:
        print('🎉 PIPELINE COMPLETED SUCCESSFULLY!')
        print('='*80)
        print('\nNext steps:')
        print('1. Check results in fusion_results_v1/ directory')
        print('2. Compare XGBoost vs CNN performance')
        print('3. Consider building fusion models (images + metadata)')
        print('4. Deploy the best-performing model')
    else:
        print('⚠️ PIPELINE COMPLETED WITH ERRORS')
        print('='*80)
        print('\nSome stages failed. Please check the errors above.')
        print('You may need to:')
        print('1. Fix the errors in failed stages')
        print('2. Re-run specific stages manually')
        print('3. Check data paths and configurations')
    
    print('\n' + '='*80 + '\n')
    
    return results


# =====================================================
# MAIN ENTRY POINT
# =====================================================

def main():
    """Main entry point"""
    
    try:
        results = run_complete_pipeline()
        
        # Exit with appropriate code
        if results and not results.get('stages_failed'):
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure
            
    except KeyboardInterrupt:
        print('\n\n⚠️ Pipeline interrupted by user')
        sys.exit(130)
        
    except Exception as e:
        print(f'\n\n❌ Pipeline failed with unexpected error:')
        print(f'   {str(e)}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
