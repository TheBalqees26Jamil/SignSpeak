# SignSpeak

Real-Time Yemeni Sign Language Recognition System.

## Project Overview

SignSpeak is a real-time Yemeni Sign Language recognition system that converts sign gestures into Arabic text and Arabic speech.

The system uses MediaPipe Hand Landmarks and Machine Learning classifiers instead of raw-image deep learning models to improve generalization and reduce overfitting.


## Project Pipeline

Image

↓

MediaPipe Hands

↓

21 Hand Landmarks

↓

63 Numerical Features

↓

Machine Learning Classifier

↓

English Label

↓

Arabic Mapping

↓

Sentence Builder

↓

Arabic Text Display

↓

Arabic Text-To-Speech


## Dataset

* 32 Yemeni Sign Language classes
* 24,245 original images
* Black background images
* Based on the Unified Yemeni Sign Language Dictionary

## Project Structure

SignSpeak/
│
|── app/
│
│── models/
│
├── scripts/
│   │
│   ├── data_preparation/
│   │   ├── extract_landmarks.py
│   │   ├── build_landmark_dataset.py
│   │   ├── inspect_dataset.py
│   │   └── remove_duplicates.py
│   │
│   ├── training/
│   │
│   ├── evaluation/
│   │
│   └── deployment/
│
│── README.md
|
|── .gitignore
│
└── requirements.txt



## Data Preparation

### Landmark Extraction

MediaPipe Hands was used to extract:

* 21 hand landmarks
* x, y, z coordinates

Total Features:

63 Features

---

### Landmark Dataset

After extraction:

* Original Images: 24,245
* Successful Landmark Samples: 19,918
* Failed Detections: 4,327

---

### Data Cleaning

Duplicate landmark vectors were identified and removed.

Results:

* Before Cleaning: 19,918 samples
* After Cleaning: 18,186 samples
* Removed Duplicates: 1,732 samples

Final Dataset:

data/landmarks_dataset_clean.csv

---
### Current Progress

### Completed

* Business Understanding
* Data Understanding
* Data Preparation
* Landmark Extraction
* Dataset Cleaning

### Next Step

* Label Encoding
* Train/Test Split
* Random Forest Training
* Model Evaluation
* Model Comparison
* Real-Time Deployment

---

## Technologies

* Python
* OpenCV
* MediaPipe
* Pandas
* NumPy
* Scikit-learn

---

## Project Status

Data Preparation Completed

Next Phase: Modeling

