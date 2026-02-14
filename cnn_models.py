"""
CNN Model Architectures for Weight Estimation
Image-only regression models using transfer learning.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import (
    EfficientNetB0,
    MobileNetV2,
    ResNet50,
    DenseNet121
)

# =====================================================
# EFFICIENTNET-B0 MODEL
# =====================================================

def build_efficientnetb0_weight_regressor(img_shape=(224, 224, 3), lr=1e-4):
    """
    Build EfficientNet-B0 based weight regressor.
    
    Args:
        img_shape: Input image shape (height, width, channels)
        lr: Learning rate
    
    Returns:
        Compiled Keras model
    """
    
    print("\n[MODEL] Building EfficientNet-B0 weight regressor...")
    
    # Input layer
    inputs = layers.Input(shape=img_shape, name='image_input')
    
    # Normalize inputs to [0, 1]
    x = layers.Rescaling(1./255)(inputs)
    
    # EfficientNet-B0 backbone (pretrained on ImageNet)
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=img_shape,
        pooling='avg'
    )
    
    # Fine-tune last layers
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    x = base_model(x, training=False)
    
    # Regression head
    x = layers.Dense(512, activation='relu', 
                     kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    
    # Output layer (weight prediction)
    outputs = layers.Dense(1, activation='linear', name='weight_output')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name='EfficientNetB0_WeightRegressor')
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mean_squared_error',
        metrics=[
            'mae',
            tf.keras.metrics.RootMeanSquaredError(name='rmse')
        ]
    )
    
    print(f"✓ Model built: {model.name}")
    print(f"  Total params: {model.count_params():,}")
    print(f"  Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    
    return model


# =====================================================
# MOBILENET-V2 MODEL
# =====================================================

def build_mobilenetv2_weight_regressor(img_shape=(224, 224, 3), lr=1e-4):
    """
    Build MobileNet-V2 based weight regressor.
    Lightweight model suitable for faster training.
    
    Args:
        img_shape: Input image shape
        lr: Learning rate
    
    Returns:
        Compiled Keras model
    """
    
    print("\n[MODEL] Building MobileNet-V2 weight regressor...")
    
    inputs = layers.Input(shape=img_shape, name='image_input')
    x = layers.Rescaling(1./255)(inputs)
    
    # MobileNet-V2 backbone
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=img_shape,
        pooling='avg'
    )
    
    # Fine-tune
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    x = base_model(x, training=False)
    
    # Regression head
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    
    outputs = layers.Dense(1, activation='linear', name='weight_output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='MobileNetV2_WeightRegressor')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mean_squared_error',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    
    print(f"✓ Model built: {model.name}")
    print(f"  Total params: {model.count_params():,}")
    
    return model


# =====================================================
# RESNET-50 MODEL
# =====================================================

def build_resnet50_weight_regressor(img_shape=(224, 224, 3), lr=1e-4):
    """
    Build ResNet-50 based weight regressor.
    Deeper model for potentially better accuracy.
    
    Args:
        img_shape: Input image shape
        lr: Learning rate
    
    Returns:
        Compiled Keras model
    """
    
    print("\n[MODEL] Building ResNet-50 weight regressor...")
    
    inputs = layers.Input(shape=img_shape, name='image_input')
    x = layers.Rescaling(1./255)(inputs)
    
    # ResNet-50 backbone
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=img_shape,
        pooling='avg'
    )
    
    # Fine-tune
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    x = base_model(x, training=False)
    
    # Regression head
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    
    outputs = layers.Dense(1, activation='linear', name='weight_output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='ResNet50_WeightRegressor')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mean_squared_error',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    
    print(f"✓ Model built: {model.name}")
    print(f"  Total params: {model.count_params():,}")
    
    return model


# =====================================================
# CUSTOM CNN MODEL (Lightweight)
# =====================================================

def build_custom_cnn_weight_regressor(img_shape=(224, 224, 3), lr=1e-4):
    """
    Build custom CNN from scratch for weight regression.
    Lightweight alternative without transfer learning.
    
    Args:
        img_shape: Input image shape
        lr: Learning rate
    
    Returns:
        Compiled Keras model
    """
    
    print("\n[MODEL] Building custom CNN weight regressor...")
    
    inputs = layers.Input(shape=img_shape, name='image_input')
    x = layers.Rescaling(1./255)(inputs)
    
    # Convolutional blocks
    # Block 1
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)
    
    # Block 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)
    
    # Block 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 4
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    
    outputs = layers.Dense(1, activation='linear', name='weight_output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='CustomCNN_WeightRegressor')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mean_squared_error',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    
    print(f"✓ Model built: {model.name}")
    print(f"  Total params: {model.count_params():,}")
    
    return model


# =====================================================
# MODEL FACTORY
# =====================================================

def get_model(model_name='efficientnet', img_shape=(224, 224, 3), lr=1e-4):
    """
    Factory function to get model by name.
    
    Args:
        model_name: 'efficientnet', 'mobilenet', 'resnet', or 'custom'
        img_shape: Input image shape
        lr: Learning rate
    
    Returns:
        Compiled Keras model
    """
    
    model_dict = {
        'efficientnet': build_efficientnetb0_weight_regressor,
        'mobilenet': build_mobilenetv2_weight_regressor,
        'resnet': build_resnet50_weight_regressor,
        'custom': build_custom_cnn_weight_regressor
    }
    
    if model_name.lower() not in model_dict:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Choose from {list(model_dict.keys())}")
    
    return model_dict[model_name.lower()](img_shape, lr)


# =====================================================
# EXAMPLE USAGE
# =====================================================

if __name__ == "__main__":
    print("="*60)
    print("CNN MODEL ARCHITECTURES - TEST")
    print("="*60)
    
    # Test building each model
    models_to_test = ['efficientnet', 'mobilenet', 'resnet', 'custom']
    
    for model_name in models_to_test:
        print(f"\n[TEST] Building {model_name} model...")
        try:
            model = get_model(model_name, lr=1e-4)
            print(f"✓ {model_name} model built successfully")
            print(f"  Input shape: {model.input_shape}")
            print(f"  Output shape: {model.output_shape}")
            
            # Clean up
            del model
            tf.keras.backend.clear_session()
            
        except Exception as e:
            print(f"❌ Failed to build {model_name}: {e}")
    
    print("\n✅ Model architecture tests completed!")
