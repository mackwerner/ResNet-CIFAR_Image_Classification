import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    Input, Dense, GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]
np.random.seed(0)
tf.random.set_seed(0)

model_name = "Transfer Learning - ResNet50"

# Load and preprocess CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype("float32")
X_test  = X_test.astype("float32")

num_classes = 10

y_train_oh = to_categorical(y_train, num_classes)
y_test_oh  = to_categorical(y_test, num_classes)

# ResNet50 preprocessing
X_train = preprocess_input(X_train)
X_test  = preprocess_input(X_test)

# Input layer
input_layer = Input(shape=(32, 32, 3))

# Pre-trained ResNet50 base
base_model = ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=input_layer
)

# Freeze convolutional base
base_model.trainable = False

# Per instructions — ONLY Retrain last three layers:
# Average pooling layer, FC layer, Softmax
layer = GlobalAveragePooling2D()(base_model.output)
layer = Dense(256, activation="relu")(layer)
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

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)

# Metrics
train_acc  = history.history["accuracy"][-1]
val_acc    = history.history["val_accuracy"][-1]
overfit_gap = train_acc - val_acc

print(f"MODEL: {model_name}")
print(f"Final Train Accuracy   : {train_acc:.4f}")
print(f"Final Validation Acc   : {val_acc:.4f}")
print(f"Final Test Accuracy    : {test_acc:.4f}")
print(f"Overfitting Gap        : {overfit_gap:.4f}")
