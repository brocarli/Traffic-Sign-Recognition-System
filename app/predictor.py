"""
Traffic Sign Predictor
======================
Handles model loading and inference for traffic sign recognition.
"""

import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf


class TrafficSignPredictor:
    """
    Handles loading the trained CNN model and making predictions
    on new traffic sign images.
    """

    def __init__(self, model_path: str, config):
        """
        Initialize the predictor.

        Args:
            model_path: Path to the trained .h5 model file
            config: Flask app configuration object
        """
        self.model_path = model_path
        self.config = config
        self.model = None
        self.class_names = config.CLASS_NAMES
        self.label_map = config.LABEL_MAP
        self.img_size = config.IMG_SIZE
        self._model_metadata = {}
        self._load_model()
        self._load_metadata()

    def _load_model(self):
        """Load the trained Keras model from disk."""
        if not os.path.exists(self.model_path):
            print(f"⚠️  WARNING: Model not found at {self.model_path}")
            print("   Please train the model first: python train_model.py")
            self.model = None
            return

        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded from: {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model = None

    def _load_metadata(self):
        """Load model training metadata if available."""
        metadata_path = os.path.join(
            os.path.dirname(self.model_path), 'model_metadata.json'
        )
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self._model_metadata = json.load(f)
            # Override class names from metadata if available
            if 'class_names' in self._model_metadata:
                self.class_names = self._model_metadata['class_names']
                self.img_size = self._model_metadata.get('img_size', self.img_size)

    def is_ready(self) -> bool:
        """Check if the model is loaded and ready for prediction."""
        return self.model is not None

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess an image for model inference.

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image array of shape (1, img_size, img_size, 3)
        """
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_path: str) -> dict:
        """
        Predict the traffic sign class from an image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with prediction results:
                - predicted_class: Internal class name
                - label: Human-readable label
                - confidence: Confidence percentage
                - top5: Top 5 predictions with confidence scores
                - is_confident: Whether confidence > 70%
        """
        if not self.is_ready():
            return {
                'error': 'Model not loaded. Please train the model first.',
                'predicted_class': None,
                'label': None,
                'confidence': 0,
                'top5': []
            }

        try:
            img_array = self.preprocess_image(image_path)
            predictions = self.model.predict(img_array, verbose=0)[0]

            # Top prediction
            top_idx = int(np.argmax(predictions))
            top_confidence = float(predictions[top_idx]) * 100

            predicted_class = self.class_names[top_idx]
            label = self.label_map.get(predicted_class, predicted_class.replace('_', ' ').title())

            # Top 5 predictions
            top5_indices = np.argsort(predictions)[::-1][:5]
            top5 = []
            for idx in top5_indices:
                cls = self.class_names[idx]
                top5.append({
                    'class': cls,
                    'label': self.label_map.get(cls, cls.replace('_', ' ').title()),
                    'confidence': round(float(predictions[idx]) * 100, 2)
                })

            return {
                'predicted_class': predicted_class,
                'label': label,
                'confidence': round(top_confidence, 2),
                'top5': top5,
                'is_confident': top_confidence >= 70.0,
                'error': None
            }

        except Exception as e:
            return {
                'error': str(e),
                'predicted_class': None,
                'label': None,
                'confidence': 0,
                'top5': []
            }

    def get_model_info(self) -> dict:
        """Return model information and training stats."""
        if not self.is_ready():
            return {'status': 'not_loaded', 'message': 'Model not trained yet'}

        info = {
            'status': 'ready',
            'num_classes': len(self.class_names),
            'img_size': self.img_size,
            'model_path': self.model_path,
        }
        info.update(self._model_metadata)
        return info
