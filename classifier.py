# -*- coding: utf-8 -*-
"""Landmark Classifier App - For Streamlit Cloud Deployment"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import json
import easyocr

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
MODEL_PATH = 'FineTuned01-EfficientNetB0_CNN_Model.h5'
CLASS_NAMES_PATH = 'class_names.json'
CONFIDENCE_THRESHOLD = 0.75

# -----------------------------------------------------------------------------
# INITIALIZE OCR READER & FACE DETECTOR
# -----------------------------------------------------------------------------
# Initialize OCR Reader (only load once)
ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)

# Initialize Human Face Detector (OpenCV Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def detect_humans(img_path):
    """Returns True if human faces are detected."""
    try:
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            # Try loading from PIL if cv2.imread fails
            img_pil = Image.open(img_path).convert('RGB')
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return len(faces) > 0
    except Exception:
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

        # ✅ Add compile=False to avoid optimizer issues
        self.model = tf.keras.models.load_model(
            self.model_path,
            safe_mode=False,
            compile=False  # ✅ Don't compile on load (avoids optimizer issues)
        )

        with open(self.classes_path, 'r') as f:
            self.class_names = json.load(f)

    def predict(self, image_input):
        try:
            # Save image temporarily for face detection and OCR
            if isinstance(image_input, str):
                img_path = image_input
                img = Image.open(img_path).convert('RGB')
            elif isinstance(image_input, Image.Image):
                # Save PIL image to temp file
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                image_input.save(temp_file.name)
                img_path = temp_file.name
                img = image_input
            else:
                raise ValueError("Input must be file path or PIL Image")

            # 1. Human Validation (Reject if faces found)
            if detect_humans(img_path):
                return None

            # 2. CNN Prediction
            img_array = np.array(img)
            img_resized = cv2.resize(img_array, (290, 290))
            img_normalized = img_resized / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)

            probs = self.model.predict(img_batch, verbose=0)[0]
            pred_idx = np.argmax(probs)
            confidence = np.max(probs)

            # 3. Confidence Validation (Reject objects/non-landmarks)
            if confidence < CONFIDENCE_THRESHOLD:
                return None

            landmark = self.class_names[pred_idx].strip()
            location = LOCATION_MAP.get(landmark, "Unknown Location")

            # 4. OCR Verification (Optional text check)
            try:
                ocr_results = ocr_reader.readtext(img_path, detail=0)
                detected_text = " ".join(ocr_results).lower()
                landmark_keywords = landmark.lower().split()

                # If confidence is moderate, require text match
                if confidence < 0.90:
                    match_found = any(keyword in detected_text for keyword in landmark_keywords if len(keyword) > 3)
                    if not match_found:
                        return None
            except Exception:
                # If OCR fails, still return result if confidence is high
                if confidence < 0.90:
                    return None

            # Clean up temp file if created
            if isinstance(image_input, Image.Image):
                try:
                    os.unlink(img_path)
                except:
                    pass

            return {
                'name': landmark,
                'place': location
            }

        except Exception as e:
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