import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
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

model_name = "Model 2 - Wider ResNet (32-64-128)"

# Load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0

num_classes = 10
y_train_oh = to_categorical(y_train, num_classes)
y_test_oh  = to_categorical(y_test, num_classes)

# Residual block
def residual_block(input_layer, filters, stride=1, use_projection=False):
    shortcut = input_layer

    layer = Conv2D(filters, (3,3), strides=stride, padding="same", use_bias=False)(input_layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    layer = Conv2D(filters, (3,3), strides=1, padding="same", use_bias=False)(layer)
    layer = BatchNormalization()(layer)

    if use_projection:
        shortcut = Conv2D(filters, (1,1), strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    layer = Add()([layer, shortcut])
    layer = ReLU()(layer)

    return layer

# Input
input_layer = Input(shape=(32,32,3))

# wider initial conv (32 filters instead of 16)
layer = Conv2D(32, (3,3), strides=1, padding="same", use_bias=False)(input_layer)
layer = BatchNormalization()(layer)
layer = ReLU()(layer)

# Stage 1 (32 filters instead of 16)
for _ in range(3):
    layer = residual_block(layer, filters=32, stride=1, use_projection=False)

# Stage 2 (64 filters instead of 32)
layer = residual_block(layer, filters=64, stride=2, use_projection=True)
for _ in range(2):
    layer = residual_block(layer, filters=64, stride=1, use_projection=False)

# Stage 3 (128 filters instead of 64)
layer = residual_block(layer, filters=128, stride=2, use_projection=True)
for _ in range(2):
    layer = residual_block(layer, filters=128, stride=1, use_projection=False)

# Classifier
layer = AveragePooling2D(pool_size=(8,8))(layer)
layer = Flatten()(layer)
output_layer = Dense(num_classes, activation="softmax")(layer)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train
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
