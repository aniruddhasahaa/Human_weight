"""
CNN Image-Only Model Training - README
Complete guide for training CNN models for weight estimation.
"""

# CNN Image-Only Model Training

Scripts for training deep learning CNN models using only images for human weight estimation.

## 📁 Files Overview

### Core Scripts

1. **`image_data_loader.py`**
   - Load images from folds
   - Prepare data for CNN training
   - Optional data augmentation support
   - Sample weights for imbalanced data

2. **`cnn_models.py`**
   - CNN model architectures
   - 4 pre-built models:
     - **EfficientNet-B0** (Default, best accuracy)
     - **MobileNet-V2** (Lightweight, faster)
     - **ResNet-50** (Deeper, more parameters)
     - **Custom CNN** (From scratch, no pre-training)
   - Transfer learning with ImageNet weights

3. **`train_cnn_image.py`**
   - Single experiment training function
   - Training callbacks (EarlyStopping, ModelCheckpoint, etc.)
   - Model evaluation and metrics
   - Prediction saving

4. **`run_cnn_cv.py`** ⭐ **MAIN SCRIPT**
   - Complete 5-fold cross-validation
   - Automatic experiment management
   - Comprehensive result tracking
   - Multi-format output saving

## 🚀 Quick Start

### Basic Usage

Run 5-fold cross-validation with default settings (EfficientNet-B0):

```bash
python run_cnn_cv.py
```

This will:
- ✅ Train 5 CNN models (one per fold)
- ✅ Use EfficientNet-B0 architecture
- ✅ Save all results and predictions
- ⏱️ Takes ~6-12 hours (depends on GPU)

### Choose Different Model

Edit `run_cnn_cv.py` main() function:

```python
def main():
    run_5fold_cnn_cross_validation(
        model_name='mobilenet',  # Options: 'efficientnet', 'mobilenet', 'resnet', 'custom'
        lr=1e-4
    )
```

### Single Experiment Test

Test on one fold first:

```bash
python train_cnn_image.py
```

## 📊 Available Models

### 1. EfficientNet-B0 (Default) ⭐
```python
model_name='efficientnet'
```
- **Best accuracy** - Recommended
- Pre-trained on ImageNet
- ~5M parameters
- Training time: ~1.5-2 hours per fold

### 2. MobileNet-V2
```python
model_name='mobilenet'
```
- **Fastest training** - Good for quick experiments
- Pre-trained on ImageNet
- ~3.5M parameters
- Training time: ~1-1.5 hours per fold

### 3. ResNet-50
```python
model_name='resnet'
```
- **Deeper model** - Potentially higher accuracy
- Pre-trained on ImageNet
- ~25M parameters
- Training time: ~2-3 hours per fold

### 4. Custom CNN
```python
model_name='custom'
```
- **No pre-training** - Train from scratch
- ~5M parameters
- Training time: ~2-3 hours per fold
- May require more epochs

## ⚙️ Configuration

### Training Parameters

In `config_training.py`:

```python
# Image settings
IMG_SHAPE = (224, 224, 3)

# Training settings
BATCH_SIZE = 8          # Increase if you have more GPU memory
EPOCHS = 75             # Maximum epochs (early stopping will stop sooner)
PATIENCE = 15           # Early stopping patience
LEARNING_RATE = 1e-4    # Initial learning rate
```

### Callbacks

Automatically configured in `train_cnn_image.py`:

- **ModelCheckpoint**: Saves best model based on validation MAE
- **EarlyStopping**: Stops if no improvement for 15 epochs
- **ReduceLROnPlateau**: Reduces LR by 0.5 if no improvement for 6 epochs
- **CSVLogger**: Logs training metrics to CSV

## 📂 Output Structure

After running, you'll find in `fusion_results_v1/`:

```
fusion_results_v1/
├── image_test_fold_1/
│   ├── best_model.keras              # Saved model
│   ├── training_log.csv              # Training metrics per epoch
│   ├── history.csv                   # Full training history
│   └── predictions.csv               # Test predictions
├── image_test_fold_2/
├── ... (folds 3-5)
│
├── cnn_experiments_summary.json      # All experiment details
├── cnn_average_metrics.json          # Aggregated metrics
├── cnn_results_summary.csv           # Metrics per fold
├── cnn_all_predictions.csv           # All predictions combined
└── cnn_summary.txt                   # Human-readable summary
```

## 📈 Expected Performance

### EfficientNet-B0 (Typical Results)
```
MAE:  ~6-10 lb
RMSE: ~8-13 lb
R²:   ~0.88-0.94
```

### MobileNet-V2
```
MAE:  ~7-11 lb
RMSE: ~9-14 lb
R²:   ~0.86-0.92
```

Performance varies by fold due to data distribution.

## 🎯 Advanced Usage

### Custom Learning Rate

```python
from train_cnn_image import run_image_experiment
from cnn_models import build_efficientnetb0_weight_regressor

metrics, preds = run_image_experiment(
    train_folds=[1, 2, 3, 4],
    test_fold=5,
    exp_name="custom_lr_experiment",
    model_builder=build_efficientnetb0_weight_regressor,
    lr=5e-5  # Custom learning rate
)
```

### Load and Use Trained Model

```python
import tensorflow as tf
import numpy as np

# Load saved model
model = tf.keras.models.load_model('fusion_results_v1/image_test_fold_1/best_model.keras')

# Load test images
from image_data_loader import load_test_fold_image_only
X_test, y_test, filenames = load_test_fold_image_only(fold_index=1)

# Make predictions
predictions = model.predict(X_test, batch_size=8)

print(f"Predictions shape: {predictions.shape}")
```

### Data Augmentation (Optional)

Edit `train_cnn_image.py` to add data augmentation:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create data generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# Use in model.fit()
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    ...
)
```

### Fine-tune More Layers

Edit `cnn_models.py` to unfreeze more layers:

```python
# Fine-tune more layers for potentially better performance
base_model.trainable = True
for layer in base_model.layers[:-40]:  # Unfreeze last 40 layers instead of 20
    layer.trainable = False
```

## 🔍 Monitoring Training

### TensorBoard (Optional)

Add TensorBoard callback:

```python
from tensorflow.keras.callbacks import TensorBoard

callbacks.append(
    TensorBoard(
        log_dir=os.path.join(exp_dir, 'tensorboard'),
        histogram_freq=1
    )
)

# Then run: tensorboard --logdir=fusion_results_v1/image_test_fold_1/tensorboard
```

### Training Progress

Monitor `training_log.csv` in real-time:

```bash
tail -f fusion_results_v1/image_test_fold_1/training_log.csv
```

## 🐛 Troubleshooting

### Issue: Out of Memory (OOM) Error

**Solutions:**
1. Reduce batch size in `config_training.py`:
   ```python
   BATCH_SIZE = 4  # or even 2
   ```

2. Use MobileNet instead of EfficientNet:
   ```python
   model_name='mobilenet'
   ```

3. Enable GPU memory growth (already in config):
   ```python
   configure_gpu()  # Already called
   ```

### Issue: Training is Too Slow

**Solutions:**
1. Use MobileNet (fastest model)
2. Reduce image size (edit model architectures)
3. Use fewer training folds for testing
4. Reduce epochs

### Issue: Poor Performance

**Solutions:**
1. Try different models (EfficientNet usually best)
2. Increase epochs (75 → 100)
3. Adjust learning rate (1e-4 → 5e-5)
4. Add data augmentation
5. Fine-tune more layers
6. Check data quality

### Issue: Model Not Improving

**Solutions:**
1. Check if early stopping is triggering too early
2. Increase patience (15 → 25)
3. Adjust learning rate schedule
4. Verify data normalization

## 📊 Comparing Models

Run experiments with different models and compare:

```bash
# Run EfficientNet
python run_cnn_cv.py  # Edit to use 'efficientnet'

# Run MobileNet
python run_cnn_cv.py  # Edit to use 'mobilenet'

# Compare results
python compare_model_results.py  # Create this script
```

## 💡 Best Practices

1. **Start Small**: Test on 1 fold first
2. **Monitor Training**: Check training_log.csv regularly
3. **Save Checkpoints**: Models are auto-saved, don't worry
4. **GPU Memory**: Start with batch_size=8, adjust if needed
5. **Patience**: Deep learning takes time (hours, not minutes)
6. **Reproducibility**: Random seed is set automatically

## 🔬 Model Architecture Details

### EfficientNet-B0 Architecture

```
Input (224x224x3)
    ↓
Rescaling (0-1)
    ↓
EfficientNet-B0 Base (ImageNet)
    ↓
Dense(512) + Dropout(0.3) + L2
    ↓
Dense(256) + Dropout(0.2) + L2
    ↓
Dense(128)
    ↓
Dense(1) - Weight Output
```

### Transfer Learning Strategy

1. Load ImageNet pre-trained weights
2. Freeze early layers (feature extraction)
3. Fine-tune last 20 layers
4. Add custom regression head
5. Train end-to-end

## 📝 Notes

- **GPU Required**: Training on CPU will be extremely slow
- **Disk Space**: Each model ~20-100MB, 5 folds = ~100-500MB
- **Memory**: 8GB GPU RAM recommended for batch_size=8
- **Time**: Full 5-fold CV = 6-15 hours depending on model
- **Results Vary**: Performance depends on data quality and fold split

## 🎓 Next Steps

After training CNN models:
1. Compare with XGBoost metadata-only results
2. Build fusion models (images + metadata)
3. Ensemble predictions from multiple models
4. Deploy best-performing model

## 📄 Dependencies

```bash
pip install tensorflow>=2.8.0
pip install numpy pandas scikit-learn
pip install matplotlib seaborn  # For visualization
```

## 🔗 Related Scripts

- `train_xgboost_meta.py` - Metadata-only XGBoost model
- `run_xgboost_cv.py` - XGBoost 5-fold CV
- `analyze_results.py` - Results analysis and visualization

---

**Remember**: Deep learning requires patience. Let the model train, monitor the logs, and trust the process! 🚀
