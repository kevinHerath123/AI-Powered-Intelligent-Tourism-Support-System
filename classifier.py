# -*- coding: utf-8 -*-
"""Landmark Classifier App - For Streamlit Cloud Deployment"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import json
import tempfile

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
MODEL_PATH = 'FineTuned01-EfficientNetB0_CNN_Model.h5'
CLASS_NAMES_PATH = 'class_names.json'
CONFIDENCE_THRESHOLD = 0.75

# -----------------------------------------------------------------------------
# LOCATION MAPPING
# -----------------------------------------------------------------------------
LOCATION_MAP = {
    "Adams Peak": "Rathnapura, Sabaragamuwa Province, Sri Lanka",
    "Ancient City of Polonnaruwa": "Polonnaruwa, North Central Province, Sri Lanka",
    "Beruwala Light House": "Beruwala, Western Province, Sri Lanka",
    "British War Cemetery": "Kandy, Central Province, Sri Lanka",
    "Bundala National Park": "Hambantota, Southern Province, Sri Lanka",
    "Delft Island": "Jaffna, Northern Province, Sri Lanka",
    "Dowa Rock Temple": "Bandarawela, Uva Province, Sri Lanka",
    "Ganagaramaya Temple": "Colombo, Western Province, Sri Lanka",
    "Henarathgoda Botanical Gard": "Gampaha, Western Province, Sri Lanka",
    "Hortains Plain": "Nuwara Eliya, Central Province, Sri Lanka",
    "Independance Square": "Colombo, Western Province, Sri Lanka",
    "Jaya Sri Maha Bodhi": "Anuradhapura, North Central Province, Sri Lanka",
    "Lotus Tower": "Colombo, Western Province, Sri Lanka",
    "Maligawa Buddha Statue": "Kandy, Central Province, Sri Lanka",
    "Nine Arches Bridge": "Ella, Uva Province, Sri Lanka",
    "Pinnawala Elephant Orphanage": "Kegalle, Sabaragamuwa Province, Sri Lanka",
    "Sigiriya": "Matale, Central Province, Sri Lanka",
    "Sinharaja Forest": "Ratnapura, Sabaragamuwa Province, Sri Lanka",
    "Sri Dalada Maligawa": "Kandy, Central Province, Sri Lanka",
    "Star Fort": "Matara, Southern Province, Sri Lanka",
    "Turtle Hatchery": "Kosgoda, Southern Province, Sri Lanka",
    "Vavuniya Archaeological Museum": "Vavuniya, Northern Province, Sri Lanka",
    "Wilapattu National Park": "Puttalam, North Western Province, Sri Lanka",
    "Yapahuwa Rock Fortress": "Yapahuwa, North Western Province, Sri Lanka",
}

# -----------------------------------------------------------------------------
# GLOBAL VARIABLES FOR LAZY LOADING
# -----------------------------------------------------------------------------
ocr_reader = None
face_cascade = None


def get_ocr_reader():
    """Lazy load EasyOCR only when needed"""
    global ocr_reader
    if ocr_reader is None:
        try:
            import easyocr
            ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        except Exception as e:
            print(f"OCR Load Error: {e}")
            return None
    return ocr_reader


def get_face_cascade():
    """Lazy load Face Detector"""
    global face_cascade
    if face_cascade is None:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def detect_humans(img_array):
    """Returns True if human faces are detected"""
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = get_face_cascade().detectMultiScale(gray, 1.1, 4)
        return len(faces) > 0
    except:
        return False


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

        self.model = tf.keras.models.load_model(self.model_path, safe_mode=False, compile=False)
        with open(self.classes_path, 'r') as f:
            self.class_names = json.load(f)

    def predict(self, image_input):
        try:
            # Load Image
            if isinstance(image_input, str):
                img = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, Image.Image):
                img = image_input.convert('RGB')
            else:
                raise ValueError("Input must be file path or PIL Image")

            img_array = np.array(img)

            # 1. Human Validation (Reject if faces found)
            if detect_humans(img_array):
                return None

            # 2. CNN Prediction
            img_resized = cv2.resize(img_array, (290, 290))
            img_normalized = img_resized / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)

            probs = self.model.predict(img_batch, verbose=0)[0]
            pred_idx = np.argmax(probs)
            confidence = float(np.max(probs))

            # 3. Confidence Validation (Reject objects/non-landmarks)
            if confidence < CONFIDENCE_THRESHOLD:
                return None

            landmark = self.class_names[pred_idx].strip()
            location = LOCATION_MAP.get(landmark, "Unknown Location")

            # 4. OCR Verification (Only if confidence is moderate < 0.90)
            if confidence < 0.90:
                ocr = get_ocr_reader()
                if ocr:
                    try:
                        # Save temp file for OCR
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                        img.save(temp_file.name)

                        # Read text
                        ocr_results = ocr.readtext(temp_file.name, detail=0)
                        detected_text = " ".join(ocr_results).lower()

                        # Cleanup
                        os.unlink(temp_file.name)

                        # Verify text match
                        landmark_keywords = landmark.lower().split()
                        match_found = any(keyword in detected_text for keyword in landmark_keywords if len(keyword) > 3)

                        if not match_found:
                            return None
                    except Exception:
                        # If OCR fails, reject moderate confidence predictions
                        return None
                else:
                    # If OCR reader failed to load, reject moderate confidence
                    return None

            # Only return name and place (NO confidence)
            return {
                'name': landmark,
                'place': location
            }

        except Exception as e:
            print(f"Prediction Error: {e}")
            return None


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
    """Get prediction - with lazy initialization"""
    global classifier
    if classifier is None:
        classifier = LandmarkClassifier()
    return classifier.predict(image_input)