import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.spatial.distance import mahalanobis
from PIL import Image
import pickle
import os

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/resnet50_eurosat_latest.h5")
model = load_model(MODEL_PATH)

# Load Mahalanobis parameters
PARAMS_PATH = os.path.join(os.path.dirname(__file__), "../models/mahalanobis_params.pkl")
with open(PARAMS_PATH, "rb") as f:
    mean_vector, covariance_matrix = pickle.load(f)

# Extract features from the second-last layer
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)

# Function to compute Mahalanobis distance
def compute_mahalanobis_distance(image_features):
    return mahalanobis(image_features, mean_vector, np.linalg.inv(covariance_matrix))

# Define class labels (same as in training)
CLASS_LABELS = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

def compute_mahalanobis_distance(image_features, mean_vector, covariance_matrix):
    inv_cov_matrix = np.linalg.inv(covariance_matrix)  # Compute inverse covariance matrix
    return mahalanobis(image_features, mean_vector, inv_cov_matrix)

def predict_image(image_path, mahalanobis_threshold=20.0):
    image = Image.open(image_path).convert("RGB").resize((128, 128))
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    # Extract feature vector
    image_features = feature_extractor.predict(image).flatten()

    # Compute Mahalanobis distance
    distance = compute_mahalanobis_distance(image_features, mean_vector, covariance_matrix)

    # If too far from known classes, classify as "Other"
    if distance > mahalanobis_threshold:
        return "Other (Unfamiliar Image)", distance

    # Predict class label
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)

    return CLASS_LABELS[predicted_class], distance
