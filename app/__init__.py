"""
Traffic Sign Recognition - Flask Application Factory
"""

import os
from flask import Flask
from config import config


def create_app(config_name: str = 'default') -> Flask:
    """
    Application factory function.

    Args:
        config_name: Configuration environment name

    Returns:
        Flask application instance
    """
    app = Flask(
        __name__,
        template_folder='../templates',
        static_folder='../static'
    )

    # Load configuration
    app.config.from_object(config[config_name])

    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Register blueprints
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app
