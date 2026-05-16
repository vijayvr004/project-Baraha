import os
import cv2
import numpy as np
from emnist import extract_training_samples, extract_test_samples

print("Extracting EMNIST Balanced dataset from cache...")
X_train, y_train = extract_training_samples('balanced')
X_test, y_test = extract_test_samples('balanced')

# 1. Save NPZ
os.makedirs('dataset', exist_ok=True)
npz_path = os.path.join('dataset', 'emnist_balanced.npz')
print(f"Saving highly compressed raw dataset to {npz_path}...")
np.savez_compressed(npz_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
print(f"Dataset successfully saved! Size: {os.path.getsize(npz_path) / (1024*1024):.1f} MB")

# 2. Save Samples
os.makedirs('dataset_samples', exist_ok=True)
print("Generating sample PNGs for teacher presentation...")

EMNIST_BALANCED_MAPPING = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]

# Save one image for every class
for class_idx, char in enumerate(EMNIST_BALANCED_MAPPING):
    # Find the first image of this class
    img_idx = np.where(y_train == class_idx)[0][0]
    img_array = X_train[img_idx]
    
    # Optional: Images in EMNIST are white-on-black (28x28). 
    # For a presentation, it might look better to invert it to black-on-white.
    inverted_img = 255 - img_array
    
    # Save image
    safe_char = char if char.isalnum() else f"char_{class_idx}"
    if char.isupper():
        filename = f"Uppercase_{safe_char}.png"
    elif char.islower():
        filename = f"Lowercase_{safe_char}.png"
    else:
        filename = f"Digit_{safe_char}.png"
        
    cv2.imwrite(os.path.join('dataset_samples', filename), inverted_img)

print("Saved 47 sample images into 'dataset_samples' folder!")
