

const API_BASE_URL = "http://localhost:8000";

const videoElement      = document.getElementById('camera-feed');
const canvasElement     = document.getElementById('camera-canvas');
const canvasCtx         = canvasElement.getContext('2d');
const cameraPlaceholder = document.getElementById('camera-placeholder');
const resultLabel       = document.getElementById('result-label');
const sentenceDisplay   = document.getElementById('sentence-display');
const clearBtn          = document.getElementById('clear-btn');

let isCameraRunning = false;
let animationFrameId = null;
let lastApiCall = 0;
const API_INTERVAL_MS = 200;

let currentLandmarks = [];
let isProcessing = false;

const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
    [5,9],[9,13],[13,17]
];


let videoTransform = {
    drawWidth: 0,
    drawHeight: 0,
    offsetX: 0,
    offsetY: 0
};

async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
            audio: false
        });

        videoElement.srcObject = stream;

        videoElement.onloadedmetadata = () => {
            videoElement.play();
            isCameraRunning = true;
            cameraPlaceholder.style.display = 'none';

            const size = 480;
            canvasElement.width = size;
            canvasElement.height = size;

            startLoop();
        };

    } catch (err) {
        console.error("Camera error:", err);
        cameraPlaceholder.style.display = 'block';
        cameraPlaceholder.textContent = "Camera Error - " + err.message;
    }
}

function computeVideoTransform() {
    const size = canvasElement.width;
    const videoAspect = videoElement.videoWidth / videoElement.videoHeight;
    const canvasAspect = 1;

    if (videoAspect > canvasAspect) {
        videoTransform.drawHeight = size;
        videoTransform.drawWidth = size * videoAspect;
        videoTransform.offsetX = (size - videoTransform.drawWidth) / 2;
        videoTransform.offsetY = 0;
    } else {
        videoTransform.drawWidth = size;
        videoTransform.drawHeight = size / videoAspect;
        videoTransform.offsetX = 0;
        videoTransform.offsetY = (size - videoTransform.drawHeight) / 2;
    }
}

function startLoop() {
    function loop(timestamp) {
        if (!isCameraRunning) return;

        const size = canvasElement.width;

        
        canvasCtx.clearRect(0, 0, size, size);

       
        computeVideoTransform();

        
        canvasCtx.save();
        canvasCtx.translate(size, 0);
        canvasCtx.scale(-1, 1);

        canvasCtx.drawImage(
            videoElement,
            videoTransform.offsetX,
            videoTransform.offsetY,
            videoTransform.drawWidth,
            videoTransform.drawHeight
        );
        canvasCtx.restore();

        
        let frameData = null;
        if (timestamp - lastApiCall > API_INTERVAL_MS && !isProcessing) {
            lastApiCall = timestamp;
            frameData = canvasElement.toDataURL('image/jpeg', 0.8);
        }

        
        if (currentLandmarks && currentLandmarks.length > 0) {
            drawLandmarks(currentLandmarks);
        }

        
        if (frameData) {
            sendFrameToBackend(frameData);
        }

        animationFrameId = requestAnimationFrame(loop);
    }

    animationFrameId = requestAnimationFrame(loop);
}


function drawLandmarks(landmarks) {
    if (!landmarks || landmarks.length === 0) return;

    const { drawWidth, drawHeight, offsetX, offsetY } = videoTransform;
    const size = canvasElement.width;

    for (const hand of landmarks) {
        
        
        canvasCtx.strokeStyle = '#ffffff';
        canvasCtx.lineWidth = 3;
        canvasCtx.lineCap = 'round';
        canvasCtx.lineJoin = 'round';

        for (const [start, end] of HAND_CONNECTIONS) {
            if (start < hand.length && end < hand.length) {
                const p1 = hand[start];
                const p2 = hand[end];

                

                const x1 = offsetX + p1.x * drawWidth;
                const y1 = offsetY + p1.y * drawHeight;
                const x2 = offsetX + p2.x * drawWidth;
                const y2 = offsetY + p2.y * drawHeight;

                canvasCtx.beginPath();
                canvasCtx.moveTo(x1, y1);
                canvasCtx.lineTo(x2, y2);
                canvasCtx.stroke();
            }
        }

        
        for (let i = 0; i < hand.length; i++) {
            const point = hand[i];

            const x = offsetX + point.x * drawWidth;
            const y = offsetY + point.y * drawHeight;

            
            canvasCtx.beginPath();
            canvasCtx.arc(x, y, 8, 0, 2 * Math.PI);
            canvasCtx.fillStyle = 'rgba(86, 29, 150, 0.3)';
            canvasCtx.fill();

            
            canvasCtx.beginPath();
            canvasCtx.arc(x, y, 4, 0, 2 * Math.PI);

            if (i === 0) {
                canvasCtx.fillStyle = '#7131ad';      
            } else if (i % 4 === 0) {
                canvasCtx.fillStyle = '#ee6bcd';      
            } else {
                canvasCtx.fillStyle = '#ffffff';      
            }

            canvasCtx.fill();
            canvasCtx.strokeStyle = '#a32171';
            canvasCtx.lineWidth = 1.5;
            canvasCtx.stroke();
        }
    }
}

async function sendFrameToBackend(frameData) {
    if (isProcessing) return;
    isProcessing = true;

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ frame: frameData })
        });

        if (!response.ok) {
            console.warn("[/predict] HTTP error:", response.status);
            currentLandmarks = [];
            return;
        }

        const data = await response.json();

        updateUI(data.prediction, data.confidence);
        updateSentence(data.sentence);

        if (data.landmarks && data.landmarks.length > 0) {
            currentLandmarks = data.landmarks;
        } else {
            currentLandmarks = [];
        }

    } catch (err) {
        console.error("[/predict] Network error:", err);
        currentLandmarks = [];
    } finally {
        isProcessing = false;
    }
}

async function clearSentence() {
    try {
        const response = await fetch(`${API_BASE_URL}/clear`, {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });
        const data = await response.json();
        updateSentence(data.sentence);
        updateUI("", 0.0);
        lastSpokenSentence = "";  // Reset so next sentence will be spoken
    } catch (err) {
        console.error("[/clear] Network error:", err);
    }
}

async function fetchSentence() {
    try {
        const response = await fetch(`${API_BASE_URL}/sentence`);
        const data = await response.json();
        updateSentence(data.sentence);
    } catch (err) {
        console.error("[/sentence] Network error:", err);
    }
}

function updateUI(prediction, confidence) {
    if (prediction && prediction !== "Waiting..." && prediction !== "") {
        const confPercent = Math.round(confidence * 100);
        resultLabel.textContent = `${prediction} (${confPercent}%)`;
        resultLabel.classList.remove('empty');
    } else {
        resultLabel.textContent = "لا يوجد نتيجة";
        resultLabel.classList.add('empty');
    }
}



let lastSpokenSentence = "";

function speakSentence(text) {
    
    window.speechSynthesis.cancel();
    
    if (!text || text.trim() === "") return;
    
    
    if (text.trim() === lastSpokenSentence) return;
    
    lastSpokenSentence = text.trim();
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'ar-SA';  
    utterance.rate = 1.0;      
    utterance.pitch = 1.0;     
    utterance.volume = 1.0;    
    
    window.speechSynthesis.speak(utterance);
}


function updateSentence(text) {
    const sentenceDisplay = document.getElementById('sentence-display');
    
    
    const currentText = sentenceDisplay.textContent;
    if (text && text.trim() !== "" && text.trim() !== currentText.trim()) {
        speakSentence(text);
    }
    
    sentenceDisplay.textContent = text || "";
    
    if (!text || text.trim() === "") {
        sentenceDisplay.classList.add('empty');
        lastSpokenSentence = "";
    } else {
        sentenceDisplay.classList.remove('empty');
    }
}


clearBtn.addEventListener('click', clearSentence);

window.addEventListener('beforeunload', () => {
    if (animationFrameId) cancelAnimationFrame(animationFrameId);
    if (videoElement.srcObject) {
        videoElement.srcObject.getTracks().forEach(track => track.stop());
    }
});

document.addEventListener('DOMContentLoaded', () => {
    initCamera();
    fetchSentence();
});