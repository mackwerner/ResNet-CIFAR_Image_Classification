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
from tensorflow.keras.regularizers import l2

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]
np.random.seed(0)
tf.random.set_seed(0)

model_name = "Model 5 - L2 Weight Decay"

# Load and preprocess CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0

num_classes = 10

y_train_oh = to_categorical(y_train, num_classes)
y_test_oh  = to_categorical(y_test, num_classes)

# L2 regularization strength
l2_reg = 1e-4

# Residual block
def residual_block(input_layer, filters, stride=1, use_projection=False):
    shortcut = input_layer

    # First 3x3 Conv2D
    layer = Conv2D(
        filters,
        (3, 3),
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(l2_reg)
    )(input_layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    # Second 3x3 Conv2D
    layer = Conv2D(
        filters,
        (3, 3),
        strides=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(l2_reg)
    )(layer)
    layer = BatchNormalization()(layer)

    # Projection shortcut
    if use_projection:
        shortcut = Conv2D(
            filters,
            (1, 1),
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_regularizer=l2(l2_reg)
        )(shortcut)
        shortcut = BatchNormalization()(shortcut)

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
    use_bias=False,
    kernel_regularizer=l2(l2_reg)
)(input_layer)
layer = BatchNormalization()(layer)
layer = ReLU()(layer)

# Residual Block Group 1: 16 filters, n = 3
for _ in range(3):
    layer = residual_block(layer, filters=16, stride=1, use_projection=False)

# Residual Block Group 2: 32 filters, n = 3
layer = residual_block(layer, filters=32, stride=2, use_projection=True)
for _ in range(2):
    layer = residual_block(layer, filters=32, stride=1, use_projection=False)

# Residual Block Group 3: 64 filters, n = 3
layer = residual_block(layer, filters=64, stride=2, use_projection=True)
for _ in range(2):
    layer = residual_block(layer, filters=64, stride=1, use_projection=False)

# Average pooling and classifier
layer = AveragePooling2D(pool_size=(8, 8))(layer)
layer = Flatten()(layer)
output_layer = Dense(
    num_classes,
    activation="softmax",
    kernel_regularizer=l2(l2_reg)
)(layer)

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