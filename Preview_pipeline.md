# Human Weight Estimation - Quick Start Guide

## 🎯 One-Command Start

```bash
python CODE_PIPELINE.py
```

This runs the **complete pipeline** automatically (~7-14 hours total).

---

## 📋 What You Need

### Files Required
- ✅ Images in `Input Data/Image_224/`
- ✅ `All_features_224.csv` (metadata)
- ✅ `All_labels_224.csv` (weight labels)

### Software Required
```bash
pip install tensorflow xgboost pandas numpy scikit-learn pillow tqdm
```

---

## 🚀 Three Ways to Run

### 1. Complete Pipeline (Automated)
```bash
python CODE_PIPELINE.py
```
**Time:** ~7-14 hours | **Best for:** Production runs

### 2. Stage-by-Stage (Manual Control)
```bash
# Preprocessing (1 hour)
python 01_load_and_explore_data.py
python 02_preprocess_images.py
python 03_feature_engineering.py

# Fold Creation (20 min)
python 05_create_folds.py
python 06_add_weights_to_folds.py

# Training
python run_xgboost_cv.py    # 30 min
python run_cnn_cv.py         # 8-12 hours

# Analysis
python analyze_results.py    # 5 min
```

### 3. Quick Test (One Fold)
```bash
python train_xgboost_meta.py   # 5 min
python train_cnn_image.py      # 1.5-2 hours
```
**Best for:** Testing setup

---

## ⚙️ Configuration

### Skip Stages
Edit `CODE_PIPELINE.py`:
```python
RUN_PREPROCESSING = True
RUN_FOLD_CREATION = True
RUN_VERIFICATION = True
RUN_XGBOOST_TRAINING = True
RUN_CNN_TRAINING = False  # ← Set to False to skip CNN
RUN_ANALYSIS = True
```

### Change CNN Model
Edit `run_cnn_cv.py`:
```python
run_5fold_cnn_cross_validation(
    model_name='efficientnetB0'
    lr=1e-4
)
```

### Change Features
Edit `config_training.py`:
```python
META_COLS = META_COLS_ORIGINAL       # 9 features (default)
```


## 📁 Output Files

```
fusion_results_v1/
├── meta_only_results.csv        # XGBoost metrics
├── meta_only_predictions.csv    # XGBoost predictions
├── cnn_results_summary.csv      # CNN metrics
├── cnn_all_predictions.csv      # CNN predictions
└── image_test_fold_1/           # Trained CNN models
    └── best_model.keras
```

---

## 🐛 Common Issues

### Out of Memory
```python
BATCH_SIZE = 4  # Reduce in config_training.py
```

### File Not Found
```bash
# Run preprocessing first
python 02_preprocess_images.py
```

### Missing Weights
```bash
python 06_add_weights_to_folds.py
```

### Training Too Slow
```python
model_name='mobilenet'  # Use faster model
```

---

## 📚 Documentation

| File | Description |
|------|-------------|
| `README.md` | Complete documentation |
| `README_TRAINING.md` | XGBoost training guide |
| `README_CNN_TRAINING.md` | CNN training guide |

---

## 🎓 Workflow

```
Raw Data
    ↓
01-03: Preprocessing → processed_dataset/
    ↓
05-06: Fold Creation → folds_individual/
    ↓
├─→ XGBoost Training → meta_only_results.csv
│
└─→ CNN Training → cnn_results_summary.csv
    ↓
Analysis → Plots and summaries
```

---

## 💡 Pro Tips

1. **Start small:** Test one fold first
2. **Monitor progress:** Check CSV logs
3. **Use MobileNet** for faster experiments
4. **Use EfficientNet** for best accuracy
5. **Be patient:** Deep learning takes hours!

---

## ✅ Checklist

Before running:
- [ ] Data files in correct location
- [ ] Python dependencies installed
- [ ] GPU available (`nvidia-smi`)
- [ ] Enough disk space (20GB+)
- [ ] Paths configured in `config_training.py`

---

## 🚀 Next Steps

1. Run pipeline: `python CODE_PIPELINE.py`
2. Check results in `fusion_results_v1/`
3. Compare XGBoost vs CNN
4. Build fusion models (optional)
5. Deploy best model

---

**Questions?** Check `README.md` for detailed documentation.

**Good luck! 🎉**
