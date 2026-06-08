import cv2
import joblib
import mediapipe as mp
import numpy as np
from collections import deque, Counter

# =====================================
# LOAD MODEL
# =====================================

model = joblib.load("models/mlp_model/mlp_best.pkl")
scaler = joblib.load("models/mlp_model/scaler.pkl")
encoder = joblib.load("models/encoders/label_encoder.pkl")

# =====================================
# MEDIAPIPE
# =====================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# =====================================
# SMOOTHING BUFFER
# =====================================

pred_buffer = deque(maxlen=10)

# threshold for unknown detection
CONF_THRESHOLD = 0.70

# =====================================
# CAMERA
# =====================================

cap = cv2.VideoCapture(0)

stable_prediction = "Waiting..."

while True:

    success, frame = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    prediction_text = "No Hand"
    confidence_text = ""

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        features = []

        for lm in hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])

        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)

        # =========================
        # PREDICT PROBABILITY
        # =========================
        probs = model.predict_proba(features)[0]

        pred_index = np.argmax(probs)
        confidence = probs[pred_index]

        label = encoder.inverse_transform([pred_index])[0]

        # =========================
        # UNKNOWN DETECTION
        # =========================
        if confidence < CONF_THRESHOLD:
            label = "Unknown"
        else:
            pred_buffer.append(label)

        # =========================
        # TEMPORAL SMOOTHING
        # =========================
        if len(pred_buffer) > 0:
            most_common = Counter(pred_buffer).most_common(1)[0][0]
            stable_prediction = most_common

        prediction_text = label
        confidence_text = f"{confidence:.2f}"

        # draw landmarks
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

    # =====================================
    # UI TEXT
    # =====================================

    cv2.putText(
        frame,
        f"Raw: {prediction_text}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Stable: {stable_prediction}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence_text}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 200, 255),
        2
    )

    cv2.imshow("SignSpeak Pro", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()