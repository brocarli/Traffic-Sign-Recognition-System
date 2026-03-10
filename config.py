"""
Configuration settings for the Traffic Sign Recognition System.
"""

import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DATASET_PATH = os.path.join(BASE_DIR, os.environ.get('DATASET_PATH', 'dataset'))
    MODEL_PATH = os.path.join(BASE_DIR, os.environ.get('MODEL_PATH', 'model/traffic_sign_model.h5'))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, os.environ.get('UPLOAD_FOLDER', 'static/uploads'))
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    IMG_SIZE = int(os.environ.get('IMG_SIZE', 64))
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

    # Model training settings
    BATCH_SIZE = 32
    EPOCHS = 30
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    LEARNING_RATE = 0.001

    # Class labels - must match your dataset folder names exactly (alphabetical order)
    CLASS_NAMES = [
        'bike_crossing',
        'children_zone',
        'deer_warning',
        'end_restrictions',
        'intersection_warning',
        'keep_left',
        'keep_right',
        'narrow_road',
        'no_entry',
        'no_honking',
        'no_left_turn',
        'no_overtaking',
        'no_overtaking_trucks',
        'no_right_turn',
        'no_stopping',
        'no_straight_ahead',
        'no_swerving',
        'no_trucks',
        'no_turning',
        'no_uturn',
        'no_vehicles',
        'ped_crossing',
        'road_work',
        'roundabout_warning',
        'slippery_road',
        'snow_warning',
        'speed_limit_5',
        'speed_limit_15',
        'speed_limit_20',
        'speed_limit_30',
        'speed_limit_40',
        'speed_limit_50',
        'speed_limit_60',
        'speed_limit_70',
        'speed_limit_80',
        'speed_limit_100',
        'speed_limit_120',
        'steep_down',
        'steep_up',
        'stop',
        'straight_ahead_only',
        'straight_left_only',
        'straight_right_only',
        'traffic_light_ahead',
        'turn_left_mandatory',
        'turn_mandatory',
        'turn_right_mandatory',
        'uneven_road',
        'warning',
        'warning_left_turn',
        'warning_right_turn',
        'warning_winding_road',
        'yield',
    ]

    NUM_CLASSES = len(CLASS_NAMES)

    # Human-readable label map
    LABEL_MAP = {
        'bike_crossing': 'Bike Crossing',
        'children_zone': 'Children Zone',
        'deer_warning': 'Deer Warning',
        'end_restrictions': 'End Restrictions',
        'intersection_warning': 'Intersection Warning',
        'keep_left': 'Keep Left',
        'keep_right': 'Keep Right',
        'narrow_road': 'Narrow Road',
        'no_entry': 'No Entry',
        'no_honking': 'No Honking',
        'no_left_turn': 'No Left Turn',
        'no_overtaking': 'No Overtaking',
        'no_overtaking_trucks': 'No Overtaking (Trucks)',
        'no_right_turn': 'No Right Turn',
        'no_stopping': 'No Stopping',
        'no_straight_ahead': 'No Straight Ahead',
        'no_swerving': 'No Swerving',
        'no_trucks': 'No Trucks',
        'no_turning': 'No Turning',
        'no_uturn': 'No U-Turn',
        'no_vehicles': 'No Vehicles',
        'ped_crossing': 'Pedestrian Crossing',
        'road_work': 'Road Work',
        'roundabout_warning': 'Roundabout Warning',
        'slippery_road': 'Slippery Road',
        'snow_warning': 'Snow Warning',
        'speed_limit_5': 'Speed Limit 5',
        'speed_limit_15': 'Speed Limit 15',
        'speed_limit_20': 'Speed Limit 20',
        'speed_limit_30': 'Speed Limit 30',
        'speed_limit_40': 'Speed Limit 40',
        'speed_limit_50': 'Speed Limit 50',
        'speed_limit_60': 'Speed Limit 60',
        'speed_limit_70': 'Speed Limit 70',
        'speed_limit_80': 'Speed Limit 80',
        'speed_limit_100': 'Speed Limit 100',
        'speed_limit_120': 'Speed Limit 120',
        'steep_down': 'Steep Downhill',
        'steep_up': 'Steep Uphill',
        'stop': 'Stop',
        'straight_ahead_only': 'Straight Ahead Only',
        'straight_left_only': 'Straight or Left Only',
        'straight_right_only': 'Straight or Right Only',
        'traffic_light_ahead': 'Traffic Light Ahead',
        'turn_left_mandatory': 'Turn Left (Mandatory)',
        'turn_mandatory': 'Turn Mandatory',
        'turn_right_mandatory': 'Turn Right (Mandatory)',
        'uneven_road': 'Uneven Road',
        'warning': 'General Warning',
        'warning_left_turn': 'Warning: Left Turn',
        'warning_right_turn': 'Warning: Right Turn',
        'warning_winding_road': 'Warning: Winding Road',
        'yield': 'Yield',
    }


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    FLASK_ENV = 'development'


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    FLASK_ENV = 'production'


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
