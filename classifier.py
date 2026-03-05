"""Landmark Classifier App - For Streamlit Deployment"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import json

# -----------------------------------------------------------------------------
# CONFIGURATION (Relative Paths)
# -----------------------------------------------------------------------------
MODEL_PATH = 'C:/Users/Lap.lk/Desktop/ML_VIVA/FineTuned-EfficientNetB0_CNN_Model.h5'
CLASS_NAMES_PATH = 'C:/Users/Lap.lk/Desktop/ML_VIVA/class_names.json'

# -----------------------------------------------------------------------------
# LOCATION MAPPING
# -----------------------------------------------------------------------------
LOCATION_MAP = {
    "Adam's Peak ": "Rathnapura, Sabaragamuwa Province, Sri Lanka ",
    "Ancient City of Polonnaruwa ": "Polonnaruwa, North Central Province, Sri Lanka ",
    "Beruwala Light House ": "Beruwala, Western Province, Sri Lanka ",
    "British War Cemetery ": "Kandy, Central Province, Sri Lanka ",
    "Bundala National Park ": "Hambantota, Southern Province, Sri Lanka ",
    "Delft Island ": "Jaffna, Northern Province, Sri Lanka ",
    "Dowa Rock Temple ": "Bandarawela, Uva Province, Sri Lanka ",
    "Ganagaramaya Temple ": "Colombo, Western Province, Sri Lanka ",
    "Henarathgoda Botanical Gard ": "Gampaha, Western Province, Sri Lanka ",
    "Hortains Plain ": "Nuwara Eliya, Central Province, Sri Lanka ",
    "Independance Square ": "Colombo, Western Province, Sri Lanka ",
    "Jaya Sri Maha Bodhi ": "Anuradhapura, North Central Province, Sri Lanka ",
    "Lotus Tower ": "Colombo, Western Province, Sri Lanka ",
    "Maligawa Buddha Statue ": "Kandy, Central Province, Sri Lanka ",
    "Nine Arches Bridge ": "Ella, Uva Province, Sri Lanka ",
    "Pinnawala Elephant Orphanage ": "Kegalle, Sabaragamuwa Province, Sri Lanka ",
    "Sigiriya ": "Matale, Central Province, Sri Lanka ",
    "Sinharaja Forest ": "Ratnapura, Sabaragamuwa Province, Sri Lanka ",
    "Sri Dalada Maligawa ": "Kandy, Central Province, Sri Lanka ",
    "Star Fort ": "Matara, Southern Province, Sri Lanka ",
    "Turtle Hatchery ": "Kosgoda, Southern Province, Sri Lanka ",
    "Vavuniya Archaeological Museum ": "Vavuniya, Northern Province, Sri Lanka ",
    "Wilapattu National Park ": "Puttalam, North Western Province, Sri Lanka ",
    "Yapahuwa Rock Fortress ": "Yapahuwa, North Western Province, Sri Lanka ",
}

# -----------------------------------------------------------------------------
# CLASSIFIER CLASS
# -----------------------------------------------------------------------------
class LandmarkClassifier:
    def __init__(self, model_path=MODEL_PATH, classes_path=CLASS_NAMES_PATH):
        self.model_path = model_path
        self.classes_path = classes_path
        self.model = None
        self.class_names = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        if not os.path.exists(self.classes_path):
            raise FileNotFoundError(f"Class names file not found at {self.classes_path}")

        self.model = tf.keras.models.load_model(self.model_path)
        with open(self.classes_path, 'r') as f:
            self.class_names = json.load(f)


    def predict(self, image_input):
        img_batch = self.preprocess_image(image_input)
        predictions = self.model.predict(img_batch, verbose=0)
        pred_idx = np.argmax(predictions[0])
        landmark_name = self.class_names[pred_idx]
        location = LOCATION_MAP.get(landmark_name, "Unknown Location")

        return {
            'name': landmark_name,
            'place': location,
            'confidence': float(np.max(predictions[0]))
        }

# -----------------------------------------------------------------------------
# INITIALIZATION
# -----------------------------------------------------------------------------
classifier = None

def init_classifier(model_path=MODEL_PATH, classes_path=CLASS_NAMES_PATH):
    """Initialize the classifier manually"""
    global classifier
    classifier = LandmarkClassifier(model_path, classes_path)
    return classifier

def get_prediction(image_input):
    """Get prediction using initialized classifier"""
    if classifier is None:
        raise RuntimeError("Classifier not initialized. Call init_classifier() first.")
    return classifier.predict(image_input)