import joblib
import numpy as np
from collections import deque, Counter

from arabic_mapping import ARABIC_MAP



# LOAD MODEL
model = joblib.load("models/mlp_model/mlp_best.pkl")
scaler = joblib.load("models/mlp_model/scaler.pkl")
encoder = joblib.load("models/encoders/label_encoder.pkl")


# SETTINGS

CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_WINDOW = 15
STABLE_THRESHOLD = 10

prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
stable_prediction = "Waiting..."
sentence_buffer = []
last_added_word = None
word_locked = False
locked_word = None



def normalize_hand(hand_landmarks):
    
    wrist = hand_landmarks.landmark[0]
    wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

    middle_mcp = hand_landmarks.landmark[9]

    scale = np.sqrt(
        (middle_mcp.x - wrist_x) ** 2 +
        (middle_mcp.y - wrist_y) ** 2 +
        (middle_mcp.z - wrist_z) ** 2
    )

    if scale < 1e-6:
        scale = 1e-6

    features = []
    for lm in hand_landmarks.landmark:
        x = (lm.x - wrist_x) / scale
        y = (lm.y - wrist_y) / scale
        z = (lm.z - wrist_z) / scale
        features.extend([x, y, z])

    return features



def predict_sign(landmarks_list):
    global prediction_buffer, stable_prediction
    global sentence_buffer, last_added_word
    global word_locked, locked_word

    arabic_word = ""
    features = []

    
    features.extend(normalize_hand(landmarks_list[0]))

    
    if len(landmarks_list) >= 2:
        features.extend(normalize_hand(landmarks_list[1]))
    else:
        features.extend([0.0] * 63)

    
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    probabilities = model.predict_proba(features)[0]

    best_idx = np.argmax(probabilities)
    confidence = probabilities[best_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        current_prediction = "Unknown"
    else:
        current_prediction = encoder.inverse_transform([best_idx])[0]

    prediction_buffer.append(current_prediction)

    # Smoothing
    if len(prediction_buffer) > 0:
        most_common = Counter(prediction_buffer).most_common(1)[0]
        majority_prediction = most_common[0]
        majority_count = most_common[1]

        if majority_count >= STABLE_THRESHOLD:
            stable_prediction = majority_prediction

            # Sentence Builder
            if stable_prediction != "Unknown":
                arabic_word = ARABIC_MAP.get(stable_prediction, stable_prediction)

                
                if word_locked and arabic_word != locked_word:
                    word_locked = False
                    locked_word = None

                
                if not word_locked:
                    sentence_buffer.append(arabic_word)
                    last_added_word = arabic_word
                    word_locked = True
                    locked_word = arabic_word

    display_text = ARABIC_MAP.get(stable_prediction, stable_prediction)
    return display_text, confidence, arabic_word


def reset_on_no_hand():
    global word_locked, locked_word, stable_prediction
    word_locked = False
    locked_word = None
    prediction_buffer.clear()
    stable_prediction = "Waiting..."


def clear_sentence():
    global sentence_buffer, last_added_word, word_locked, locked_word
    sentence_buffer.clear()
    last_added_word = None
    word_locked = False
    locked_word = None
    prediction_buffer.clear()


def get_sentence():
    return " ".join(sentence_buffer)