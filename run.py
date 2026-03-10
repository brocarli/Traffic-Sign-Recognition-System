"""
Traffic Sign Recognition System - Entry Point
=============================================
Run this file to start the Flask development server.

Usage:
    python run.py
    
Or with Gunicorn (production):
    gunicorn -w 4 -b 0.0.0.0:5000 "run:app"
"""

import os
import sys
from config import Config, config

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app


# Patch app config to include the config object for predictor
_cfg_name = os.environ.get('FLASK_ENV', 'development')
if _cfg_name not in ('development', 'production', 'testing'):
    _cfg_name = 'development'

app = create_app(_cfg_name)
# Inject config object into app config for use in routes
app.config['_config_obj'] = Config


if __name__ == '__main__':
    print("=" * 55)
    print("  🚦 Traffic Sign Recognition System")
    print("=" * 55)
    print(f"  Environment: {_cfg_name}")
    print(f"  Model path:  {Config.MODEL_PATH}")
    print(f"  Dataset:     {Config.DATASET_PATH}")
    print()

    # Check model status
    if not os.path.exists(Config.MODEL_PATH):
        print("  ⚠️  WARNING: Trained model not found!")
        print("  Please run: python train_model.py")
        print("  The app will start but predictions won't work.")
    else:
        print("  ✅ Model found and ready.")

    print()
    print("  Starting server at http://127.0.0.1:5000")
    print("=" * 55)

    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=app.config.get('DEBUG', True)
    )
