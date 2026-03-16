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
CONFIDENCE_THRESHOLD = 0.65  

# -----------------------------------------------------------------------------
# LOCATION MAPPING (Match your CLASS_NAMES exactly - NO trailing spaces)
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


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
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


def detect_humans(img_array):
    """Returns True if human faces are detected"""
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = get_face_cascade().detectMultiScale(gray, 1.3, 5)
        return len(faces) > 0
    except:
        return False


def flexible_ocr_match(detected_text, landmark_name):
    """
    Flexible matching that handles blurry OCR results.
    Returns True if at least 50% of significant words match.
    """
    if not detected_text:
        return False

    detected_text = detected_text.lower()
    landmark_lower = landmark_name.lower()
    landmark_words = landmark_lower.split()

    # Check for direct match first
    if landmark_lower in detected_text:
        return True

    # Get significant words (longer than 3 chars)
    significant_words = [word for word in landmark_words if len(word) > 3]

    if not significant_words:
        return False

    # Count how many significant words are found in detected text
    matches = sum(1 for word in significant_words if word in detected_text)

    # Accept if at least 50% of significant words match
    return matches >= max(1, len(significant_words) // 2)


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
        """Load model and class names"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        if not os.path.exists(self.classes_path):
            raise FileNotFoundError(f"Class names file not found at {self.classes_path}")

        self.model = tf.keras.models.load_model(self.model_path, safe_mode=False, compile=False)
        with open(self.classes_path, 'r') as f:
            self.class_names = json.load(f)

    def predict(self, image_input):
        """
        Predict landmark from image.
        Returns: {'name': str, 'place': str} OR None if validation fails
        """
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

            # 3. Confidence Validation - Lowered threshold for blurry images
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

                        # Read text with parameters optimized for blurry images
                        ocr_results = ocr.readtext(
                            temp_file.name,
                            detail=0,
                            min_size=15,  # Lower min_size for blurry text
                            contrast_ths=0.03  # Lower contrast threshold
                        )
                        detected_text = " ".join(ocr_results)

                        # Cleanup
                        os.unlink(temp_file.name)

                        # Use flexible matching for blurry text
                        if not flexible_ocr_match(detected_text, landmark):
                            # If OCR found text but no match, be more lenient
                            if confidence < 0.85:
                                return None
                    except Exception as e:
                        print(f"OCR Error: {e}")
                        # If OCR fails completely, rely on CNN confidence
                        # Accept if confidence is reasonably high
                        if confidence < 0.85:
                            return None
                else:
                    # If OCR reader failed to load, accept if confidence is high
                    if confidence < 0.85:
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