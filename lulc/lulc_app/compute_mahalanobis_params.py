import numpy as np
import tensorflow as tf
from scipy.spatial.distance import mahalanobis
from tensorflow.keras.models import load_model
from PIL import Image
import pickle
import os

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/resnet50_eurosat_latest.h5")
model = load_model(MODEL_PATH)

# Extract features from the second-last layer
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)

# ✅ Provide the absolute path to your dataset
TRAINING_IMAGE_DIR = r"C:\Users\46rin\Downloads\EuroSAT\2750"

# Get all image file paths
training_images = []
for root, _, files in os.walk(TRAINING_IMAGE_DIR):  # Recursively search all subfolders
    for file in files:
        if file.endswith((".jpg", ".png", ".tif")):
            training_images.append(os.path.join(root, file))

# Extract feature vectors
feature_vectors = []
print(f"Processing {len(training_images)} training images...")

for img_path in training_images:
    img = Image.open(img_path).convert("RGB").resize((128, 128))  # Resize to model input size
    img = np.array(img).astype("float32") / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)

    features = feature_extractor.predict(img)
    feature_vectors.append(features.flatten())  # Flatten to 1D vector

# Convert to NumPy array
feature_vectors = np.array(feature_vectors)

# ✅ Check if feature_vectors is 2D
print("Feature Vectors Shape:", feature_vectors.shape)

# Compute mean and covariance matrix
mean_vector = np.mean(feature_vectors, axis=0)
covariance_matrix = np.cov(feature_vectors, rowvar=False)  # Compute covariance along features

# ✅ Check covariance matrix shape
print("Covariance Matrix Shape:", covariance_matrix.shape)

# Save computed values
PARAMS_PATH = os.path.join(os.path.dirname(__file__), "../models/mahalanobis_params.pkl")
with open(PARAMS_PATH, "wb") as f:
    pickle.dump((mean_vector, covariance_matrix), f)

print(f"✅ Mahalanobis parameters computed and saved at {PARAMS_PATH}")
