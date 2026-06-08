import cv2
import numpy as np
import pickle
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import os

# ── Config ────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "yoga_pose_classifier.h5")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
LANDMARKER   = os.path.join(BASE_DIR, "pose_landmarker.task")

# Number of frames to average when smoothing predictions
# Higher = more stable but slower to react to pose changes.
SMOOTHING    = 10

# Minimum confidence requried to display a pose label.
# Predictions below this threshold show "Low confidence ..." instead
CONF_THRESH  = 0.9 


# ──────────────────── Skeleton definition ──────────────────────────────────────
# Eacg tuple is a (start_index, end_index) pair of MediaPipe landmark indices
# Indices follow the MediaPipe Pose topology (0 = nose, 11/12 shouldes)
# Reference: //developers.google.com/mediapipe/solutions/vision/pose_landmarker
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),
    (11,13),(13,15),(12,14),(14,16),
    (15,17),(15,19),(15,21),
    (16,18),(16,20),(16,22),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(27,31),
    (24,26),(26,28),(28,30),(28,32),
]

# ── Load model & encoder ──────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)
print(f"Classes: {list(le.classes_)}")

# ── Load MediaPipe detector ───────────────────────────────
base_options = python.BaseOptions(model_asset_path=LANDMARKER)
options      = vision.PoseLandmarkerOptions(base_options=base_options)
detector     = vision.PoseLandmarker.create_from_options(options)
print("Detector ready!")

# ── Drawing helpers ───────────────────────────────────────
def draw_landmarks(frame, detection_result, min_vis=0.5):
    """
    Draw skeletopn lines and joint circles on the frame. 
    
    Only draws connections where both endpoints have visibility >= mins
    avoiding noisy lines for occluded or uncertain landmarks.

      Args:
        frame           (np.ndarray)              : BGR frame to draw on (modified in-place).
        detection_result (PoseLandmarkerResult)   : Output from detector.detect().
        min_vis         (float)                   : Minimum landmark visibility score (0–1).
    """

    
    h, w = frame.shape[:2]
    for pose_landmarks in detection_result.pose_landmarks:
        #Draw skeleton connections
        for s, e in POSE_CONNECTIONS:
            ls, le_ = pose_landmarks[s], pose_landmarks[e]

            #skip low-confidence landmarks to keep the overlay clean
            if ls.visibility < min_vis or le_.visibility < min_vis:
                continue
            x1, y1 = int(ls.x * w), int(ls.y * h)
            x2, y2 = int(le_.x * w), int(le_.y * h)
            cv2.line(frame, (x1,y1), (x2,y2), (255,255,0), 2)

            # Draw joint circles on top of lines so they appear above the skeleton
        for lm in pose_landmarks:
            if lm.visibility < min_vis:
                continue
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx,cy), 5, (0,255,0), -1) # filled green
            cv2.circle(frame, (cx,cy), 5, (0,0,0),   1) # Black border for contrast

def draw_ui(frame, label, confidence, pose_detected):
    """
    Render the heads-up display (top bar, bottom bar, confidence meter)

    Three display states:
        - Pose detected and confidence >= CONF_THRESH : sshow label + green bar
        - Pose detected but confience too low: Show oragne warning
        - No Pose detected: show red "NO POSE" message


        Args: 
            frame (np.ndarray) : BGR frame to draw on (modified in-place)
            labe (str) : Predicted pose class name. 
            confidence (float) : Smoothed confidence score (0-1)
            pose_detected (bool) " whether MediaPipe found any landmarks this frame    
    
    """

    h, w = frame.shape[:2]

    # Solid black bars provide a consistent backdrop for text regardless of background


    # Top bar
    cv2.rectangle(frame, (0,0), (w,70), (0,0,0), -1)
    # Bottom bar
    cv2.rectangle(frame, (0, h-110), (w,h), (0,0,0), -1)

    if pose_detected and confidence >= CONF_THRESH:
        # Pose name
        cv2.putText(frame, label, (20,48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 3)
        # Confidence bar background
        cv2.rectangle(frame, (20, h-90), (w-20, h-60), (50,50,50), -1)
        # Confidence bar fill: filled portion scales with confidence value
        bar_w = int((w-40) * confidence)
        cv2.rectangle(frame, (20, h-90), (20+bar_w, h-60), (0,200,0), -1)
        # Confidence text
        cv2.putText(frame, f"{confidence:.1%}", (20, h-35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    elif pose_detected:
        # Pose found but model is uncertain - prompt user to adjust their position
        cv2.putText(frame, "Low confidence...", (20,48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,165,255), 2)
    else:
        cv2.putText(frame, "No pose detected", (20,48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    # Persistent quit hit at the bottom
    cv2.putText(frame, "Press Q to quit", (20, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,150,150), 1)

# ── Smoothing ─────────────────────────────────────────────
pred_buffer = deque(maxlen=SMOOTHING)

def smooth_prediction(probs):
    """
    Reduce frame - to - frame label flickering by averaging recent predictions

    Appends the latest probability array to a fixed-size buffer, then returns

    the class with the highest mean probability across all buffered frames

    Args: 
        probs (np.ndarray): Shape (num_class) probability array from model.predict()

    Returns:
        tuple: 
            label (str): Class name with the highest averaged probability.
            confidence (float): Averaged probability for that class (0-1)

    """
    pred_buffer.append(probs)
    avg = np.mean(pred_buffer, axis=0)
    idx = np.argmax(avg)
    return le.classes_[idx], float(avg[idx])

# ── Main loop ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)   # C0 - Default Webcam

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    print("Try changing VideoCapture(0) to VideoCapture(1)")
    exit()

print("Webcam started — press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Mirror effect: Flip horizontally
    frame = cv2.flip(frame, 1)
    # Convert BGR ( Default for Opencv )- RGB ( required by Mediapipe )
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result  = detector.detect(mp_img)

    pose_detected      = False
    label, confidence  = "Unknown", 0.0

    if result.pose_landmarks:
        pose_detected = True

        features = []
        for lm in result.pose_landmarks[0]:
            features += [lm.x, lm.y, lm.z, lm.visibility]

        probs             = model.predict(np.array(features).reshape(1,-1), verbose=0)[0]
        label, confidence = smooth_prediction(probs)

        draw_landmarks(frame, result)

    draw_ui(frame, label, confidence, pose_detected)

    cv2.imshow("Yoga Pose Estimator", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
