"""
Flask Routes - Traffic Sign Recognition
"""

import os
import uuid
from flask import (
    Blueprint, render_template, request,
    jsonify, current_app, redirect, url_for
)
from werkzeug.utils import secure_filename
from app.predictor import TrafficSignPredictor

main_bp = Blueprint('main', __name__)

# Predictor singleton (lazy-loaded)
_predictor = None


def get_predictor() -> TrafficSignPredictor:
    """Get or create the predictor singleton."""
    global _predictor
    if _predictor is None:
        _predictor = TrafficSignPredictor(
            model_path=current_app.config['MODEL_PATH'],
            config=current_app.config['_config_obj']
        )
    return _predictor


def allowed_file(filename: str) -> bool:
    """Check if uploaded file has an allowed extension."""
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower()
        in current_app.config['ALLOWED_EXTENSIONS']
    )


@main_bp.route('/')
def index():
    """Home page - upload interface."""
    return render_template('index.html')


@main_bp.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and return prediction.

    Accepts:
        - Multipart form with 'file' field

    Returns:
        - JSON with prediction results
        - Redirects to result page on success
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file type. Please upload PNG, JPG, JPEG, GIF, BMP, or WEBP.'
        }), 400

    try:
        # Save uploaded file with unique name
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Run prediction
        predictor = get_predictor()
        result = predictor.predict(filepath)

        if result.get('error'):
            return jsonify({'error': result['error']}), 500

        # Relative path for template
        image_url = f"/static/uploads/{unique_filename}"

        # If API request (AJAX), return JSON
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            result['image_url'] = image_url
            return jsonify(result)

        # Otherwise redirect to result page
        return render_template(
            'result.html',
            result=result,
            image_url=image_url,
            filename=secure_filename(file.filename)
        )

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@main_bp.route('/api/predict', methods=['POST'])
def api_predict():
    """
    REST API endpoint for predictions.
    Returns JSON response for programmatic use.
    """
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file field in request'}), 400

    file = request.files['file']

    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'Empty file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'File type not allowed'}), 415

    try:
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"api_{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        predictor = get_predictor()
        result = predictor.predict(filepath)

        if result.get('error'):
            return jsonify({'success': False, 'error': result['error']}), 500

        return jsonify({
            'success': True,
            'predicted_class': result['predicted_class'],
            'label': result['label'],
            'confidence': result['confidence'],
            'is_confident': result['is_confident'],
            'top5': result['top5'],
            'image_url': f"/static/uploads/{unique_filename}"
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@main_bp.route('/model-info')
def model_info():
    """Model information and training stats page."""
    predictor = get_predictor()
    info = predictor.get_model_info()

    # Load training history for charts
    history_path = os.path.join(
        os.path.dirname(current_app.config['MODEL_PATH']),
        'training_history.json'
    )
    history = None
    if os.path.exists(history_path):
        import json
        with open(history_path, 'r') as f:
            history = json.load(f)

    return render_template('model_info.html', info=info, history=history)


@main_bp.route('/about')
def about():
    """About page."""
    from config import Config
    return render_template('about.html', classes=Config.CLASS_NAMES, label_map=Config.LABEL_MAP)


@main_bp.route('/health')
def health():
    """Health check endpoint."""
    predictor = get_predictor()
    return jsonify({
        'status': 'ok',
        'model_ready': predictor.is_ready()
    })
