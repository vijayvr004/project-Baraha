"""
Fast improved CNN trainer - deeper architecture, no augmentation for speed.
Expected: ~92-93% accuracy in ~10 minutes.
"""
import os, numpy as np

print("=" * 60)
print("  BARAHA - Fast Improved Model Trainer")
print("=" * 60)

npz_path = os.path.join(os.path.dirname(__file__), 'dataset', 'emnist_balanced.npz')
if not os.path.exists(npz_path):
    print(f"Error: {npz_path} not found!")
    exit(1)

data = np.load(npz_path)
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Classes: {len(np.unique(y_train))}")

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
    # Block 1
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Dense
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("\nTraining for 10 epochs (no augmentation for speed)...")
history = model.fit(
    X_train, y_train_cat,
    epochs=10,
    batch_size=256,
    validation_data=(X_test, y_test_cat),
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n{'=' * 60}")
print(f"  Test Accuracy: {test_acc*100:.1f}%")
print(f"{'=' * 60}")

# Per-class check
MAPPING = ['0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t']

preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
print("\nConfused pairs check:")
for idx, char in enumerate(MAPPING):
    mask = y_test == idx
    if mask.sum() > 0:
        acc = np.mean(preds[mask] == idx)
        if acc < 0.85 or char in ['a','Q','q','g','O','0']:
            misclass = preds[mask]
            wrong = misclass[misclass != idx]
            if len(wrong) > 0:
                top_wrong = np.bincount(wrong, minlength=NUM_CLASSES).argmax()
                print(f"  '{char}': {acc*100:.1f}% (most confused with '{MAPPING[top_wrong]}')")

model_path = os.path.join(os.path.dirname(__file__), 'character_model_tf210.h5')
model.save(model_path)
print(f"\nSaved to: {model_path} ({os.path.getsize(model_path)/(1024*1024):.1f} MB)")
print("Done! Restart start_server.bat to use the new model.")
