

import base64
from contextlib import asynccontextmanager
from typing import Optional, List

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sign_recognition import (
    predict_sign,
    reset_on_no_hand,
    clear_sentence,
    get_sentence,
)

mp_hands = mp.solutions.hands
hands_detector: Optional[mp_hands.Hands] = None


# ── Pydantic Models ─────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    frame: str


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    sentence: str
    landmarks: List[List[dict]] = []


class ClearResponse(BaseModel):
    success: bool
    sentence: str


class SentenceResponse(BaseModel):
    sentence: str


class ResetResponse(BaseModel):
    success: bool



@asynccontextmanager
async def lifespan(app: FastAPI):
    global hands_detector
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    print("[Lifespan] MediaPipe Hands initialised.")
    yield
    hands_detector.close()
    print("[Lifespan] MediaPipe Hands closed.")


# ── FastAPI App 
app = FastAPI(title="SignSpeak API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper: decode base64 → OpenCV BGR 
def decode_frame(base64_string: str) -> np.ndarray:
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(base64_string)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 frame data") from exc

    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    return image


# ── Helper: transform Raw MediaPipe Landmarks to JSON
def raw_landmarks_to_json(multi_hand_landmarks):
    
    result = []
    if not multi_hand_landmarks:
        return result

    for hand in multi_hand_landmarks:
        hand_points = []
        for lm in hand.landmark:
            hand_points.append({
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z)
            })
        result.append(hand_points)
    return result


# ── Endpoints 
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    image_bgr = decode_frame(req.frame)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    results = hands_detector.process(image_rgb)

    if results.multi_hand_landmarks:
        
        display_text, confidence, _ = predict_sign(results.multi_hand_landmarks)
        sentence = get_sentence()

        
        raw_landmarks = raw_landmarks_to_json(results.multi_hand_landmarks)

        return PredictResponse(
            prediction=display_text,
            confidence=float(confidence),
            sentence=sentence,
            landmarks=raw_landmarks,
        )
    else:
        reset_on_no_hand()
        sentence = get_sentence()
        return PredictResponse(
            prediction="Waiting...",
            confidence=0.0,
            sentence=sentence,
            landmarks=[],
        )


@app.post("/clear", response_model=ClearResponse)
async def clear():
    clear_sentence()
    return ClearResponse(success=True, sentence=get_sentence())


@app.get("/sentence", response_model=SentenceResponse)
async def sentence():
    return SentenceResponse(sentence=get_sentence())


@app.post("/reset", response_model=ResetResponse)
async def reset():
    reset_on_no_hand()
    return ResetResponse(success=True)