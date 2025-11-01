import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def create_data_generators(train_dir, test_dir, batch_size=32):
    # Balanced data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.15
    )

    # Only rescaling for testing
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Training generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Validation generator
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Test generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator

def create_model(num_classes=6):
    # Use MobileNetV2 - proven to work well with your data
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze base model initially
    base_model.trainable = False

    # Simple and effective classifier
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile with optimal learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def compute_class_weights(generator):
    """Compute class weights for imbalanced dataset"""
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(generator.classes),
        y=generator.classes
    )
    return dict(enumerate(class_weights))

def train_model(model, train_generator, validation_generator, epochs=50, callbacks=None, initial_epoch=0):
    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(train_generator)
    print("Class weights for imbalanced data:", class_weights)
    
    # Train the model with class weights
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    return history

def fine_tune_model(model, train_generator, validation_generator, epochs=30, callbacks=None):
    # Unfreeze the base model for fine-tuning
    model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = 100
    
    # Freeze earlier layers
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Number of trainable layers: {sum([l.trainable for l in model.layers])}")
    
    # Compute class weights again
    class_weights = compute_class_weights(train_generator)

    # Fine-tune the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    return history

def preprocess_image(image):
    """Preprocess a single image for prediction"""
    if isinstance(image, np.ndarray):
        image = tf.keras.preprocessing.image.array_to_img(image)
    
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)