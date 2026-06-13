# SignSpeak - Yemeni Sign Language Recognition System

<div align="center">

  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white" alt="HTML5">
  <img src="https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white" alt="CSS3">
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black" alt="JavaScript">
  <img src="https://img.shields.io/badge/MediaPipe-FF6F00?logo=mediapipe&logoColor=white" alt="MediaPipe">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white" alt="Scikit-Learn">

  <br>

  <p>
    <strong>An intelligent system for recognizing Yemeni sign language and converting it to text and spoken speech</strong>
  </p>

</div>

---

## Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Requirements](#-system-requirements)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Data Pipeline](#-data-pipeline)
- [Machine Learning Model](#-machine-learning-model)
- [API Documentation](#-api-documentation)
- [Frontend](#-frontend)
- [Configuration](#-configuration)
- [Key Files](#-key-files)
- [Supported Words](#-supported-words)
- [Troubleshooting](#-troubleshooting)

---

##  Overview

**SignSpeak** is a complete system for recognizing Yemeni sign language using computer vision and machine learning techniques. The system consists of three main components:

1. **Frontend:** An interactive web application built with HTML/CSS/JavaScript providing a live camera feed for capturing hand signs.
2. **Backend API:** A FastAPI server that processes video frames, extracts hand landmarks, and predicts the sign.
3. **Machine Learning Model:** An MLP (Multi-Layer Perceptron) neural network trained on hand landmarks extracted using MediaPipe.

### How the System Works

```
Camera → Image Frame → MediaPipe Hands → Hand Landmarks (63×2=126)
                                                    ↓
Arabic Text + Speech ← Prediction ← MLP Model ← Normalized Landmarks
                                                    ↓
                                              Sentence Builder
```

---

##  Features

| Feature | Description |
|---------|-------------|
|  **Live Camera** | Real-time video capture from the browser with hand landmark overlay |
|  **Smart Recognition** | MLP model with prediction smoothing to prevent jitter |
|  **Sentence Building** | Automatically assembles recognized words into meaningful sentences |
|  **Text-to-Speech (TTS)** | Speaks Arabic sentences aloud using Web Speech API |
|  **Dual Hand Support** | Detects one or two hands simultaneously |
|  **Duplicate Removal** | Tools to analyze and clean dataset from duplicate samples |
|  **Data Balancing** | Uses SMOTE to handle class imbalance |
|  **Cross-Validation** | 5-Fold Cross Validation for model performance evaluation |
|  **CORS Enabled** | Supports access from any origin for frontend integration |
|  **Attractive UI** | Glassmorphism design with smooth animations |
|  **Responsive** | Works on desktop, tablet, and mobile devices |
|  **RTL Support** | Full right-to-left Arabic language support |

---

##  System Requirements

### Minimum
- Python 3.8 or newer
- 4 GB RAM
- Webcam

### Recommended
- Python 3.10+
- 8 GB RAM
- GPU (optional, for MediaPipe acceleration)

---

##  Project Structure

```
SignSpeak/
├── data/                    ← Processed datasets
├── dataset/                 ← Original sign images
├── frontend/                ← Web app (index.html, style.css, app.js)
    ├─ assest/
       └─images/
          └─ left_hand.jpg
          └─ right_hand.jpg
    └─ app.js
    └─index.html
    └─style.css

├── models/                  ← Trained ML models
│   ├── encoders/
         └─label_encoder.pkl
│   └── mlp_model/
          └─mlp_best.pkl
          └─scaler.pkl
├── scripts/                 ← Utility scripts
│   ├── data_preparation/    ← Dataset prep scripts
           └─ analyze_duplicates.py
           └─ build_landmark_dataset.py
           └─ check_unique_samples.py
           └─ extract_landmarks.py
           └─ inspect_dataset.py
           └─ remove_dupli.py
           └─ remove_duplicates.py
│   ├── inference/             ← Real-time prediction
           └─ webcam_predict.py
│   └── training/            ← Model training
           └ train_mlp.py
           └ train_random_forest.py
           └ train_xgboost.py
├── .gitignore
├── arabic_mapping.py
├── main.py                  ← FastAPI entry point
├── README.md
├── requirements.txt
└── sign_recognition.py

---

##  Installation

### 1. Clone the Repository

```bash
git clone https://github.com/TheBalqees26Jamil/SignSpeak
cd SignSpeak
```


### 2. stall Requirements

```bash
pip install -r requirements.txt
```

> **Note:** MediaPipe requires some system libraries. If you encounter issues, refer to the [MediaPipe installation guide](https://developers.google.com/mediapipe/framework/getting_started/install).

### 3 Prepare the Model

Ensure the following files exist in the `models/` directory:
- `models/mlp_model/mlp_best.pkl`
- `models/mlp_model/scaler.pkl`
- `models/encoders/label_encoder.pkl`

If they don't exist, train the model first (see [Training the Model])

---

##  Usage

### Run the Backend Server

```bash
python main.py
```

The server will start at:
```
http://localhost:8000
```

You can access the interactive API documentation:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Run the Frontend

Open `index.html` directly in your browser, or use a local web server:



Then open: http://localhost:8080

### Using the Application

1. **Allow camera access** when prompted.
2. **Show a hand sign** in front of the camera.
3. **Watch the result** in the bottom panel.
4. **Sentences build automatically** with each new sign.
5. **Click "مسح" (Clear)** to reset the sentence.

---

##  Data Pipeline

### 1. Extract Landmarks

```bash
python extract_landmarks.py
```

Reads images from `dataset/` and extracts 126 features (2 hands × 21 points × 3 coordinates) per image using MediaPipe Hands.

### 2. Build Dataset

```bash
python build_landmark_dataset.py
```

Combines all CSV files from `data/landmarks_raw/` into a single file `data/landmarks_dataset.csv`.

### 3. Analyze and Remove Duplicates

```bash
# Analyze duplicate samples
python analyze_duplicates.py

# Remove duplicate samples
python remove_duplicates.py

# Check unique samples
python check_unique_samples.py

# Inspect dataset info
python inspect_dataset.py
```

---

##  Machine Learning Model

### Training the Model

```bash
python train_mlp.py
```

### Model Architecture

| Layer | Units | Activation |
|-------|-------|------------|
| Input | 126 | - |
| Hidden Layer 1 | 256 | ReLU |
| Hidden Layer 2 | 128 | ReLU |
| Output | 32 | Softmax |

### Training Steps

1. **Data Split:** 80% training, 20% testing
2. **Data Balancing:** SMOTE on training data
3. **Normalization:** StandardScaler (z-score normalization)
4. **Training:** Adam optimizer, max 500 iterations
5. **Evaluation:** 5-Fold Cross Validation

### Model Saving

The model is automatically saved to:
```
models/mlp_model/mlp_best.pkl
models/mlp_model/scaler.pkl
models/encoders/label_encoder.pkl
```

---

##  API Documentation

### Endpoints

#### `POST /predict`
Send an image frame for recognition.

**Request Body:**
```json
{
  "frame": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**Response:**
```json
{
  "prediction": "صديق",
  "confidence": 0.95,
  "sentence": "أخ صديق",
  "landmarks": [
    [
      {"x": 0.5, "y": 0.3, "z": -0.1},
      ...
    ]
  ]
}
```

#### `POST /clear`
Clear the current sentence.

**Response:**
```json
{
  "success": true,
  "sentence": ""
}
```

#### `GET /sentence`
Get the current sentence.

**Response:**
```json
{
  "sentence": "أخ صديق"
}
```

#### `POST /reset`
Reset the recognition state.

**Response:**
```json
{
  "success": true
}
```

---

##  Frontend

### Technologies Used
- **HTML5** - Page structure
- **CSS3** - Glassmorphism design with blur effects
- **Vanilla JavaScript** - Camera logic and interaction
- **Web Speech API** - Text-to-Speech (TTS)

### Visual Features
-  Square camera (480×480) with rounded corners
-  Real-time hand landmark drawing
-  Animated floating hand background images
-  Responsive design for all screen sizes
-  Full Arabic language (RTL) support

---

##  Configuration

### Recognition Settings (`sign_recognition.py`)

```python
CONFIDENCE_THRESHOLD = 0.70    # Minimum confidence threshold
SMOOTHING_WINDOW = 15            # Smoothing window size
STABLE_THRESHOLD = 10            # Minimum majority vote threshold
```

### Camera Settings (`app.js`)

```javascript
const API_INTERVAL_MS = 200;    # Frame sending interval (ms)
const API_BASE_URL = "http://localhost:8000";  # Server address
```

### MediaPipe Settings (`main.py`)

```python
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
```

---

##  Key Files

### Frontend Files

| File | Description |
|------|-------------|
| `index.html` | Main page with camera and results structure |
| `style.css` | Glassmorphism styles and animations |
| `app.js` | Camera logic, landmark drawing, and frame sending |

### Backend Files

| File | Description |
|------|-------------|
| `main.py` | FastAPI server with prediction and clear endpoints |
| `sign_recognition.py` | Recognition engine: normalization, prediction, sentence building |
| `arabic_mapping.py` | English to Arabic translation dictionary |

### Data Processing Files

| File | Description |
|------|-------------|
| `extract_landmarks.py` | Extract MediaPipe landmarks from sign images |
| `build_landmark_dataset.py` | Combine CSV files into a single dataset |
| `analyze_duplicates.py` | Analyze duplication rate per class |
| `remove_duplicates.py` | Remove duplicate rows from CSV |
| `check_unique_samples.py` | Check unique sample counts |
| `inspect_dataset.py` | Display dataset statistics |

### Training Files

| File | Description |
|------|-------------|
| `train_mlp.py` | Train MLP model with SMOTE and Cross Validation |

---

##  Supported Words

The system includes **32 classes** of Yemeni sign language:

| English | Arabic |
|---------|--------|
| Animals | حيوانات |
| Attentive | مُتيقظ |
| brother | أخ |
| builder | بناء |
| cheap | رخيص |
| Doctor | دكتور |
| excellent | بارع |
| friend | صديق |
| good | رائع |
| him | هو |
| how | كيف |
| husband | شريك |
| mechanic | ميكانيكي |
| morning | صباح |
| plane | طائرة |
| president | رئيس |
| satisfaction | الاكتفاء |
| school | مدرسة |
| six | ستة |
| seven | سبعة |
| teacher | أستاذ |
| translator | مترجم |
| vegetables | خضروات |
| very good | جيد جداً |
| wait | انتظر |
| wash | نظف |
| watch maker | مصلح الساعات |
| what | ماذا |
| when | متى |
| who | من |
| with_god_will | إن شاء الله |
| yours | لك |

---

##  Troubleshooting

### Issue: Camera Not Working
```
 Make sure to allow camera access in the browser
 Ensure the server is running on localhost:8000
 Open Developer Tools (F12) and check Console messages
```

### Issue: Model Not Recognizing Signs
```
Ensure model files exist in the models/ directory
Check that lighting is sufficient and clear
Verify that hand landmarks (purple dots) appear on the camera
```

### Issue: CORS Error
```
Make sure the server (main.py) is running before opening the frontend
Check CORS settings in main.py
```

---

<div align="center">

  <p>
    <strong>Made with ❤️ for the deaf and hard-of-hearing community</strong>
  </p>

  <br>


</div>
