import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
import time
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

class CustomLearningRateScheduler(Callback):
    def __init__(self, model, patience=5, min_lr=1e-5):
        super().__init__()
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = 0.41
        self.wait = 0
        self.model = model
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                new_lr = max(current_lr / 2, self.min_lr)
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                print(f"Reducing learning rate to {new_lr}.")
                if current_lr <= self.min_lr:
                    print("Early stopping triggered.")
                    self.model.stop_training = True
                else:
                    print("Restoring model weights from the end of the best epoch.")
                    self.model.set_weights(self.best_weights)
                self.wait = 0

def train(model, train_dataset, valid_dataset, test_dataset, pretrained=False, initial_lr=1e-3, momentum=0.9, weight_decay=1e-4, epochs=10):
    # Check if a previously trained model exists
    model_path = f'models/{model.model}_best.h5'
    if os.path.exists(model_path) and pretrained:
        w, h, c = model.input_size
        model.build((None, w, h, c))        
        print("Loading weights from previously trained model.")
        model.load_weights(model_path)

    # Model checkpoint
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, save_weights_only=True ,verbose=1)

    # Compile the model with SGD optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=momentum, decay=weight_decay)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Custom learning rate scheduler and early stopping
    lr_scheduler = CustomLearningRateScheduler(model)

    # Training history
    history = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': [], 'test_loss': [], 'test_acc': [], 'epoch_times': []}

    for epoch in range(epochs):
        start_time = time.time()
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train the model
        history_train = model.fit(train_dataset, validation_data=valid_dataset, epochs=1, callbacks=[model_checkpoint, lr_scheduler])
        history['train_loss'].extend(history_train.history['loss'])
        history['train_acc'].extend(history_train.history['accuracy'])
        history['valid_loss'].extend(history_train.history['val_loss'])
        history['valid_acc'].extend(history_train.history['val_accuracy'])

        # Evaluate on test set
        test_loss, test_acc = model.evaluate(test_dataset)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

        end_time = time.time()
        epoch_time = end_time - start_time
        history['epoch_times'].append(epoch_time)
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")

    # Calculate and add average time per epoch
    average_epoch_time = sum(history['epoch_times']) / len(history['epoch_times'])
    history['average_epoch_time'] = average_epoch_time
    print(f"Average time per epoch: {average_epoch_time:.2f} seconds")
    # 在训练循环中，记录最小的训练、验证和测试loss以及对应的epoch
    min_train_loss = min(history['train_loss'])
    min_valid_loss = min(history['valid_loss'])
    min_test_loss = min(history['test_loss'])
    best_train_acc = max(history['train_acc'])
    best_valid_acc = max(history['valid_acc'])
    best_test_acc = max(history['test_acc'])

    print(f"Minimum Train Loss: {min_train_loss}")
    print(f"Minimum Validation Loss: {min_valid_loss}")
    print(f"Minimum Test Loss: {min_test_loss}")
    print(f"Best Train Accuracy: {best_train_acc}")
    print(f"Best Validation Accuracy: {best_valid_acc}")
    print(f"Best Test Accuracy: {best_test_acc}")

    # Plot and save training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['valid_loss'], label='Validation Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title(f'{model.model} Loss over epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['valid_acc'], label='Validation Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title(f'{model.model} Accuracy over epochs')
    plt.legend()

    plt.savefig(f'models/{model.model}_training_history{start_time}.png')
    plt.show()

    # Save history to a file
    with open(f'models/{model.model}_training_history{start_time}.json', 'w') as f:
        json.dump(history, f)

    return history


