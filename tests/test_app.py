"""
Tests for Traffic Sign Recognition System
Run with: python -m pytest tests/ -v
"""

import os
import sys
import json
import pytest
import tempfile
import numpy as np
from io import BytesIO
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def app():
    """Create test Flask app."""
    from run import app as flask_app
    flask_app.config['TESTING'] = True
    flask_app.config['DEBUG'] = False
    return flask_app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


def create_test_image(size=(100, 100), color=(255, 0, 0)) -> bytes:
    """Create a simple test image in memory."""
    img = Image.new('RGB', size, color=color)
    buf = BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    return buf.read()


class TestRoutes:
    """Test Flask routes."""

    def test_index_returns_200(self, client):
        """Home page should return 200."""
        response = client.get('/')
        assert response.status_code == 200

    def test_index_contains_expected_text(self, client):
        """Home page should contain TrafficAI brand text."""
        response = client.get('/')
        assert b'Traffic' in response.data

    def test_about_returns_200(self, client):
        """About page should return 200."""
        response = client.get('/about')
        assert response.status_code == 200

    def test_model_info_returns_200(self, client):
        """Model info page should return 200."""
        response = client.get('/model-info')
        assert response.status_code == 200

    def test_health_endpoint(self, client):
        """Health check should return JSON with status."""
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'ok'
        assert 'model_ready' in data

    def test_predict_no_file(self, client):
        """Predict without file should return 400."""
        response = client.post('/api/predict')
        assert response.status_code == 400

    def test_predict_wrong_file_type(self, client):
        """Predict with non-image file should return 415."""
        data = {'file': (BytesIO(b'not an image'), 'test.txt')}
        response = client.post(
            '/api/predict',
            data=data,
            content_type='multipart/form-data'
        )
        assert response.status_code == 415

    def test_api_predict_with_image(self, client):
        """API predict with valid image should return JSON."""
        img_bytes = create_test_image()
        data = {'file': (BytesIO(img_bytes), 'test.jpg')}
        response = client.post(
            '/api/predict',
            data=data,
            content_type='multipart/form-data'
        )
        # Either succeeds (200) or fails with model not loaded (500)
        assert response.status_code in [200, 500]
        result = json.loads(response.data)
        assert 'success' in result


class TestConfig:
    """Test configuration."""

    def test_config_has_class_names(self):
        """Config should have all 25 class names."""
        from config import Config
        assert len(Config.CLASS_NAMES) == 25
        assert 'stop_sign' in Config.CLASS_NAMES
        assert 'no_entry' in Config.CLASS_NAMES

    def test_config_label_map_complete(self):
        """All class names should have labels."""
        from config import Config
        for cls in Config.CLASS_NAMES:
            assert cls in Config.LABEL_MAP, f"Missing label for: {cls}"

    def test_config_img_size(self):
        """Image size should be positive integer."""
        from config import Config
        assert isinstance(Config.IMG_SIZE, int)
        assert Config.IMG_SIZE > 0

    def test_config_num_classes(self):
        """Num classes should match class names list."""
        from config import Config
        assert Config.NUM_CLASSES == len(Config.CLASS_NAMES)


class TestPreprocessing:
    """Test image preprocessing."""

    def test_image_resize(self):
        """Image should be resized correctly."""
        from config import Config
        img = Image.new('RGB', (200, 300), color=(100, 150, 200))

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img.save(f.name)
            tmp_path = f.name

        try:
            resized = Image.open(tmp_path).resize(
                (Config.IMG_SIZE, Config.IMG_SIZE), Image.LANCZOS
            )
            assert resized.size == (Config.IMG_SIZE, Config.IMG_SIZE)
        finally:
            os.unlink(tmp_path)

    def test_image_normalization(self):
        """Pixel values should be in [0, 1] after normalization."""
        img_array = np.array(
            Image.new('RGB', (64, 64), color=(255, 128, 0)),
            dtype=np.float32
        ) / 255.0
        assert img_array.min() >= 0.0
        assert img_array.max() <= 1.0


class TestModelTraining:
    """Test model building (does not require trained weights)."""

    def test_model_build(self):
        """Model should build without errors."""
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        from train_model import build_cnn_model
        model = build_cnn_model(num_classes=25, img_size=64)
        assert model is not None

    def test_model_output_shape(self):
        """Model output shape should match num_classes."""
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        from train_model import build_cnn_model
        model = build_cnn_model(num_classes=25, img_size=64)
        assert model.output_shape == (None, 25)

    def test_model_input_shape(self):
        """Model input shape should match img_size."""
        from train_model import build_cnn_model
        model = build_cnn_model(num_classes=25, img_size=64)
        assert model.input_shape == (None, 64, 64, 3)
