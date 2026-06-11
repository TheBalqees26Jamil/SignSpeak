import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import cv2
import mediapipe as mp


from sign_recognition import (
    predict_sign,
    reset_on_no_hand,
    clear_sentence,
    get_sentence
)


# ==================================================
# MEDIAPIPE ( For only testing the webcam feed and hand detection )
# ==================================================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils



cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    display_text = "Waiting..."
    confidence_text = ""
    sentence_text = get_sentence()

    if results.multi_hand_landmarks:
        landmarks_list = results.multi_hand_landmarks

        
        display_text, confidence, _ = predict_sign(landmarks_list)
        confidence_text = f"{confidence*100:.1f}%"

        
        for hand_landmarks in landmarks_list:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
    else:
        
        reset_on_no_hand()

    
    cv2.putText(frame, f"Prediction: {display_text}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence_text}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Sentence: {sentence_text}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("SignSpeak Prediction", frame)

    
    key = cv2.waitKey(1)
    if key == ord("c"):
        clear_sentence()
        print("Sentence Cleared")
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()