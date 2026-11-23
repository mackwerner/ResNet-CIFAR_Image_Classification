import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10, cifar100 #changed to CIFAR-100 for that training
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU,
    Add, AveragePooling2D, Flatten, Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]
np.random.seed(0)
tf.random.set_seed(0)

# Model label for this run
model_name = "ResNet Baseline"

# Load and preprocess CIFAR-10/100
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0

num_classes = 10

y_train_oh = to_categorical(y_train, num_classes)
y_test_oh  = to_categorical(y_test, num_classes)

# Residual block
def residual_block(input_layer, filters, stride=1, use_projection=False):
    shortcut = input_layer

    # First 3x3 Conv2D
    layer = Conv2D(
        filters,
        (3, 3),
        strides=stride,
        padding="same",
        use_bias=False
    )(input_layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    # Second 3x3 Conv2D
    layer = Conv2D(
        filters,
        (3, 3),
        strides=1,
        padding="same",
        use_bias=False
    )(layer)
    layer = BatchNormalization()(layer)

    # Projection shortcut (1x1 Conv2D, stride=2 when needed)
    if use_projection:
        shortcut = Conv2D(
            filters,
            (1, 1),
            strides=stride,
            padding="same",
            use_bias=False
        )(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Residual addition
    layer = Add()([layer, shortcut])
    layer = ReLU()(layer)

    return layer

# Input layer
input_layer = Input(shape=(32, 32, 3))

# Initial convolution
layer = Conv2D(
    16,
    (3, 3),
    strides=1,
    padding="same",
    use_bias=False
)(input_layer)
layer = BatchNormalization()(layer)
layer = ReLU()(layer)

# Residual Block Group 1: 16 filters, n = 3
for _ in range(3):
    layer = residual_block(layer, filters=16, stride=1, use_projection=False)

# Residual Block Group 2: 32 filters, n = 3
# First block uses stride=2 and 1x1 projection shortcut
layer = residual_block(layer, filters=32, stride=2, use_projection=True)
for _ in range(2):
    layer = residual_block(layer, filters=32, stride=1, use_projection=False)

# Residual Block Group 3: 64 filters, n = 3
# First block uses stride=2 and 1x1 projection shortcut
layer = residual_block(layer, filters=64, stride=2, use_projection=True)
for _ in range(2):
    layer = residual_block(layer, filters=64, stride=1, use_projection=False)

# Average pooling and classifier
layer = AveragePooling2D(pool_size=(8, 8))(layer)
layer = Flatten()(layer)
output_layer = Dense(num_classes, activation="softmax")(layer)

# Final model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Print model summary
model.summary()

# Train model
history = model.fit(
    X_train, y_train_oh,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)

# Final metrics
train_acc  = history.history["accuracy"][-1]
val_acc    = history.history["val_accuracy"][-1]
train_loss = history.history["loss"][-1]
val_loss   = history.history["val_loss"][-1]

best_val_acc_epoch = int(np.argmax(history.history["val_accuracy"])) + 1
best_val_acc       = float(np.max(history.history["val_accuracy"]))

overfit_gap = train_acc - val_acc

print(f"MODEL: {model_name}")
print(f"Final Train Accuracy   : {train_acc:.4f}")
print(f"Final Validation Acc   : {val_acc:.4f}")
print(f"Final Test Accuracy    : {test_acc:.4f}")
print(f"Final Train Loss       : {train_loss:.4f}")
print(f"Final Validation Loss  : {val_loss:.4f}")
print(f"Final Test Loss        : {test_loss:.4f}")
print(f"Best Val Accuracy      : {best_val_acc:.4f}")
print(f"Best Val Accuracy Epoch: {best_val_acc_epoch}")
print(f"Overfitting Gap        : {overfit_gap:.4f}")

epochs = list(range(1, len(history.history["accuracy"]) + 1))
print("\nEPOCH DATA")
print(f"Model Name   : {model_name}")
print(f"Epochs       : {epochs}")
print(f"Train Acc    : {[round(x, 4) for x in history.history['accuracy']]}")
print(f"Val Acc      : {[round(x, 4) for x in history.history['val_accuracy']]}")
print(f"Train Loss   : {[round(x, 4) for x in history.history['loss']]}")
print(f"Val Loss     : {[round(x, 4) for x in history.history['val_loss']]}")

# Plot training and validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(epochs, history.history["accuracy"], label="Training Accuracy")
plt.plot(epochs, history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"{model_name} Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(epochs, history.history["loss"], label="Training Loss")
plt.plot(epochs, history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"{model_name} Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
