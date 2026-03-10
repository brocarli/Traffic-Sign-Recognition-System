# 🚦 Traffic Sign Recognition System

An AI-powered traffic sign recognition system built with a **Convolutional Neural Network (CNN)** using TensorFlow/Keras, with a **Flask** web application as the user interface.

---

## 📸 Features

- Upload traffic sign images via drag-and-drop or file picker
- Real-time CNN inference with confidence scores
- Top-5 prediction breakdown
- Model training with data augmentation
- Training history & confusion matrix visualization
- REST API endpoint for programmatic use

---

## 🗂️ Project Structure

```
traffic-sign-recognition/
├── app/
│   ├── __init__.py         # Flask app factory
│   ├── routes.py           # URL routes & views
│   └── predictor.py        # Model loading & inference
├── static/
│   ├── css/style.css       # Stylesheet
│   ├── js/main.js          # Frontend JS
│   ├── uploads/            # Uploaded images (git-ignored)
│   └── img/                # Training charts (git-ignored)
├── templates/
│   ├── base.html           # Base layout
│   ├── index.html          # Upload page
│   ├── result.html         # Prediction result
│   ├── model_info.html     # Model stats
│   └── about.html          # About page
├── model/                  # Saved model (git-ignored)
├── dataset/                # Your dataset (git-ignored)
├── tests/
│   └── test_app.py         # Unit tests
├── config.py               # App configuration
├── train_model.py          # CNN training script
├── run.py                  # Flask entry point
├── requirements.txt        # Python dependencies
├── setup.sh                # Linux/macOS setup
├── setup.bat               # Windows setup
└── .env.example            # Environment template
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/brocarli/Traffic-Sign-Recognition-System.git
cd traffic-sign-recognition
```

### 2. Set Up Environment

```bash
python -m venv venv
venv\Scripts\activate
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

### 3. Train the Model

```bash
python train_model.py
```
Options:
```bash
python train_model.py --epochs 50 --img-size 64 --batch-size 32
python train_model.py --dataset /path/to/dataset --model-path model/my_model.h5
```

Training produces:
- `model/traffic_sign_model.h5` — Trained model weights
- `model/model_metadata.json` — Training metadata
- `model/training_history.json` — Loss/accuracy history
- `model/classification_report.txt` — Per-class metrics
- `static/img/training_history.png` — Accuracy/loss curves
- `static/img/confusion_matrix.png` — Confusion matrix

### 4. Run the App

```bash
python run.py
```

Open your browser at **http://localhost:5000**

---

## 🧠 Model Architecture

```
Input (64×64×3)
    ↓
Conv Block 1: Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.25)
    ↓
Conv Block 2: Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.25)
    ↓
Conv Block 3: Conv2D(128) → BN → ReLU → Conv2D(128) → BN → ReLU → MaxPool → Dropout(0.30)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256) → BN → ReLU → Dropout(0.50)
    ↓
Dense(53) → Softmax
```

**Training details:**
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Cross-Entropy
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Augmentation: Rotation, shift, zoom, brightness jitter

---

## 🎯 Detectable Classes (53)

| Class | Label |
|-------|-------|
| `bike_crossing` | Bike Crossing |
| `children_zone` | Children Zone |
| `deer_warning` | Deer Warning |
| `end_restrictions` | End Restrictions |
| `intersection_warning` | Intersection Warning |
| `keep_left` | Keep Left |
| `keep_right` | Keep Right |
| `narrow_road` | Narrow Road |
| `no_entry` | No Entry |
| `no_honking` | No Honking |
| `no_left_turn` | No Left Turn |
| `no_overtaking` | No Overtaking |
| `no_overtaking_trucks` | No Overtaking (Trucks) |
| `no_right_turn` | No Right Turn |
| `no_stopping` | No Stopping |
| `no_straight_ahead` | No Straight Ahead |
| `no_swerving` | No Swerving |
| `no_trucks` | No Trucks |
| `no_turning` | No Turning |
| `no_uturn` | No U-Turn |
| `no_vehicles` | No Vehicles |
| `ped_crossing` | Pedestrian Crossing |
| `road_work` | Road Work |
| `roundabout_warning` | Roundabout Warning |
| `slippery_road` | Slippery Road |
| `snow_warning` | Snow Warning |
| `speed_limit_5` | Speed Limit 5 |
| `speed_limit_15` | Speed Limit 15 |
| `speed_limit_20` | Speed Limit 20 |
| `speed_limit_30` | Speed Limit 30 |
| `speed_limit_40` | Speed Limit 40 |
| `speed_limit_50` | Speed Limit 50 |
| `speed_limit_60` | Speed Limit 60 |
| `speed_limit_70` | Speed Limit 70 |
| `speed_limit_80` | Speed Limit 80 |
| `speed_limit_100` | Speed Limit 100 |
| `speed_limit_120` | Speed Limit 120 |
| `steep_down` | Steep Downhill |
| `steep_up` | Steep Uphill |
| `stop` | Stop |
| `straight_ahead_only` | Straight Ahead Only |
| `straight_left_only` | Straight or Left Only |
| `straight_right_only` | Straight or Right Only |
| `traffic_light_ahead` | Traffic Light Ahead |
| `turn_left_mandatory` | Turn Left (Mandatory) |
| `turn_mandatory` | Turn Mandatory |
| `turn_right_mandatory` | Turn Right (Mandatory) |
| `uneven_road` | Uneven Road |
| `warning` | General Warning |
| `warning_left_turn` | Warning: Left Turn |
| `warning_right_turn` | Warning: Right Turn |
| `warning_winding_road` | Warning: Winding Road |
| `yield` | Yield |

---

## 🔌 REST API

```http
POST /api/predict
Content-Type: multipart/form-data
Body: file=<image>
```

**Response:**
```json
{
  "success": true,
  "predicted_class": "stop",
  "label": "Stop",
  "confidence": 98.43,
  "is_confident": true,
  "top5": [
    {"class": "stop", "label": "Stop", "confidence": 98.43},
    ...
  ]
}
```

**Example with curl:**
```bash
curl -X POST http://localhost:5000/api/predict \
     -F "file=@traffic_sign.jpg"
```

---

## 🧪 Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## 🌐 Production Deployment

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 "run:app"
```

---

## 📋 Requirements

- Python 3.9+
- TensorFlow 2.15
- Flask 3.0
- See `requirements.txt` for full list

---
