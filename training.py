import os
import time
import glob
import re
import tensorflow as tf
from model import create_model, create_data_generators, train_model, fine_tune_model, compute_class_weights
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

CHECKPOINT_PATH = "Deployment/weights/model.weights.h5"
PAUSE_FLAG_FILE = "pause.flag"

def analyze_dataset(generator, generator_name):
    """Analyze dataset distribution"""
    print(f"\n=== {generator_name} Dataset Analysis ===")
    print(f"Total samples: {generator.samples}")
    print(f"Number of classes: {len(generator.class_indices)}")
    
    class_counts = np.bincount(generator.classes)
    class_names = list(generator.class_indices.keys())
    
    for class_name, count in zip(class_names, class_counts):
        print(f"{class_name}: {count} samples ({count/generator.samples:.1%})")
    
    return class_counts, class_names

def plot_training_history(history, save_path='training_history.png'):
    # Plot training & validation accuracy
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)
    
    # Plot training & validation loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)
    
    # Plot learning rate
    if 'lr' in history.history:
        plt.subplot(1, 3, 3)
        plt.plot(history.history['lr'])
        plt.title('Learning Rate')
        plt.ylabel('Learning Rate')
        plt.xlabel('Epoch')
        plt.yscale('log')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def wait_if_paused():
    while os.path.exists(PAUSE_FLAG_FILE):
        print("Training paused. Waiting for resume...")
        time.sleep(10)

def get_latest_checkpoint():
    """Find the latest checkpoint in the weights directory"""
    weight_files = glob.glob("Deployment/weights/model_*.weights.h5")  # Updated pattern
    if not weight_files:
        return None
    
    # Extract timestamps and find the latest one
    latest_file = max(weight_files, key=os.path.getmtime)
    return latest_file

def get_initial_epoch():
    """Get the initial epoch from the checkpoint filename"""
    latest_checkpoint = get_latest_checkpoint()
    if latest_checkpoint is None:
        return 0
    
    try:
        # Try to extract epoch number from filename if it exists
        epoch = int(re.search(r'epoch_(\d+)', latest_checkpoint).group(1))
        return epoch
    except (AttributeError, ValueError):
        return 0

def main():
    # Set memory growth for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU memory growth error: {e}")

    # Set paths
    train_dir = "Data/Train"
    test_dir = "Data/Test"
    
    # Create weights directory if it doesn't exist
    os.makedirs("Deployment/weights", exist_ok=True)
    os.makedirs("Deployment/models", exist_ok=True)
    
    # Create data generators with smaller batch size
    print("Creating data generators...")
    train_generator, validation_generator, test_generator = create_data_generators(
        train_dir, test_dir, batch_size=12  # Reduced batch size for stability
    )
    
    # Analyze datasets
    train_counts, class_names = analyze_dataset(train_generator, "Training")
    val_counts, _ = analyze_dataset(validation_generator, "Validation")
    test_counts, _ = analyze_dataset(test_generator, "Test")
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create and compile model
    print("\nCreating model...")
    model = create_model(num_classes=len(train_generator.class_indices))
    
    # Print model summary
    model.summary()
    
    # Get the latest checkpoint and initial epoch
    latest_checkpoint = get_latest_checkpoint()
    initial_epoch = get_initial_epoch()
    
    if latest_checkpoint:
        print(f"Found checkpoint: {latest_checkpoint}")
        print(f"Resuming from epoch {initial_epoch}")
        try:
            model.load_weights(latest_checkpoint)
            print("Successfully loaded previous weights")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Starting training from scratch...")
            initial_epoch = 0
    else:
        print("No checkpoint found. Starting training from scratch...")
        initial_epoch = 0
    
    # Define enhanced callbacks - FIXED: Use .weights.h5 extension
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"Deployment/weights/model_{timestamp}_epoch_{{epoch:03d}}.weights.h5",
        save_weights_only=True,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    
    best_model_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"Deployment/weights/best_model_{timestamp}.weights.h5",
        save_weights_only=True,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        mode='max',
        verbose=1
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=25,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )
    
    # Train the model
    print("\n=== Starting Initial Training ===")
    wait_if_paused()
    history = train_model(
        model, 
        train_generator, 
        validation_generator,
        epochs=100,  # Increased epochs
        initial_epoch=initial_epoch,
        callbacks=[checkpoint_cb, best_model_cb, lr_scheduler, early_stopping]
    )
    
    # Plot training history
    plot_training_history(history, f'training_history_{timestamp}.png')
    
    # Fine-tune the model
    print("\n=== Starting Fine-Tuning ===")
    wait_if_paused()
    fine_tune_history = fine_tune_model(
        model, 
        train_generator, 
        validation_generator,
        epochs=50,  # Increased fine-tuning epochs
        callbacks=[checkpoint_cb, best_model_cb, lr_scheduler, early_stopping]
    )
    
    # Plot fine-tuning history
    plot_training_history(fine_tune_history, f'fine_tuning_history_{timestamp}.png')
    
    # Save the complete model for OpenCV
    print("\nSaving complete model for OpenCV...")
    model.save(f"Deployment/models/model_complete_{timestamp}.h5")
    
    # Save the final weights - FIXED: Use .weights.h5 extension
    model.save_weights(f"Deployment/weights/model_final_{timestamp}.weights.h5")
    
    # Evaluate on test set
    print("\n=== Final Evaluation on Test Set ===")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")
    print(f"Test loss: {test_loss:.4f}")
    
    # Detailed classification report
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    # Get predictions
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # Classification report
    print("\n=== Detailed Classification Report ===")
    print(classification_report(true_classes, predicted_classes, 
                              target_names=class_names, digits=4))
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_classes, predicted_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save evaluation results
    with open(f'evaluation_results_{timestamp}.txt', 'w') as f:
        f.write(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})\n")
        f.write(f"Test loss: {test_loss:.4f}\n")
        f.write(f"Training completed at: {datetime.now()}\n")
        f.write(f"Total epochs trained: {len(history.history['accuracy']) + len(fine_tune_history.history['accuracy'])}\n")
        f.write("\nClass Distribution:\n")
        for class_name, count in zip(class_names, train_counts):
            f.write(f"{class_name}: {count} samples\n")
    
    print(f"\n=== Training Completed ===")
    print(f"Final model saved as: Deployment/models/model_complete_{timestamp}.h5")
    print(f"Evaluation results saved as: evaluation_results_{timestamp}.txt")

if __name__ == "__main__":
    main()