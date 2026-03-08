"""Landmark Classifier App - For Streamlit Cloud Deployment"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import json

# -----------------------------------------------------------------------------
# CONFIGURATION (Relative Paths)
# -----------------------------------------------------------------------------
MODEL_PATH = 'FineTuned-EfficientNetB0_CNN_Model.h5'
CLASS_NAMES_PATH = 'class_names.json'

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

        # Load model with safe_mode=False for Lambda layer
        self.model = tf.keras.models.load_model(self.model_path, safe_mode=False)

        with open(self.classes_path, 'r') as f:
            self.class_names = json.load(f)

    def predict(self, image_input):
        # Load image
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img = image_input.convert('RGB')
        else:
            raise ValueError("Input must be file path or PIL Image")

        # Preprocess
        img_array = np.array(img)
        img_resized = cv2.resize(img_array, (290, 290))
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        # Predict
        predictions = self.model.predict(img_batch, verbose=0)
        pred_idx = np.argmax(predictions[0])

        landmark_name = self.class_names[pred_idx]

        # ✅ FIX: Strip spaces from landmark_name BEFORE lookup
        location = LOCATION_MAP.get(landmark_name.strip(), "Unknown Location")

        return {
            'name': landmark_name,
            'place': location
        }


# -----------------------------------------------------------------------------
# INITIALIZATION
# -----------------------------------------------------------------------------
classifier = None

def init_classifier(model_path=MODEL_PATH, classes_path=CLASS_NAMES_PATH):
    global classifier
    classifier = LandmarkClassifier(model_path, classes_path)
    return classifier

def get_prediction(image_input):
    if classifier is None:
        raise RuntimeError("Classifier not initialized. Call init_classifier() first.")
    return classifier.predict(image_input)