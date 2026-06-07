import cv2
import joblib
import mediapipe as mp
import numpy as np

# =====================================
# LOAD MODEL
# =====================================

model = joblib.load(
    "models/mlp_model/mlp_best.pkl"
)

scaler = joblib.load(
    "models/mlp_model/scaler.pkl"
)

encoder = joblib.load(
    "models/encoders/label_encoder.pkl"
)

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
# CAMERA
# =====================================

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    image_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(image_rgb)

    prediction_text = "No Hand"

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        features = []

        for lm in hand_landmarks.landmark:
            features.extend([
                lm.x,
                lm.y,
                lm.z
            ])

        features = np.array(
            features
        ).reshape(1, -1)

        features = scaler.transform(
            features
        )

        prediction = model.predict(
            features
        )[0]

        label = encoder.inverse_transform(
            [prediction]
        )[0]

        prediction_text = label

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

    cv2.putText(
        frame,
        prediction_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "SignSpeak Prediction",
        frame
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()