import os
import numpy as np

print("Loading EMNIST dataset...")
from emnist import extract_training_samples, extract_test_samples

X_train, y_train = extract_training_samples('balanced')
X_test, y_test = extract_test_samples('balanced')

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Number of classes: {len(np.unique(y_train))}")

def fix_emnist_orientation(images):
    fixed = []
    for img in images:
        img_rot = np.rot90(img, k=3)
        img_flip = np.fliplr(img_rot)
        fixed.append(img_flip)
    return np.array(fixed)

print("Fixing EMNIST orientation (rotate 90° CW + flip horizontal)...")
X_train = fix_emnist_orientation(X_train)
X_test = fix_emnist_orientation(X_test)

X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

import tensorflow as tf
from tensorflow.keras import layers, models

NUM_CLASSES = 47

y_train_cat = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_cat = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\nTraining model for 5 epochs...")
history = model.fit(
    X_train, y_train_cat,
    epochs=5,
    batch_size=128,
    validation_data=(X_test, y_test_cat),
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

model_path = os.path.join(os.path.dirname(__file__), 'character_model.h5')
model.save(model_path)
print(f"\nModel saved to: {model_path}")
print("Training complete! You can now start the Flask server with: python main.py")
