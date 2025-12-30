import cv2
import time
import numpy as np
import mediapipe as mp
from typing import Tuple

from typing import Optional

from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    HandLandmarksConnections,
    RunningMode,
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
    PoseLandmarksConnections
)
from mediapipe.tasks import python

# === MODEL SETUP ===

# Provide the path to your downloaded hand_landmarker.task model.
HAND_MODEL_PATH: str = "mediapipe_models/hand_landmarker.task"

# Global variable to store the latest async result.
latest_hand_result: Optional[HandLandmarkerResult] = None

def handle_hand_result(
    result: HandLandmarkerResult,
    output_image: mp.Image,
    timestamp_ms: int,
) -> None:
    """Called when MediaPipe receives results."""
    global latest_hand_result
    latest_hand_result = result

# Configure MediaPipe HandLandmarker options
hand_options: HandLandmarkerOptions = HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=handle_hand_result,
    num_hands=2,
)

# Create the HandLandmarker object
hand_landmarker: HandLandmarker = HandLandmarker.create_from_options(hand_options)

# Pose Landmarker

POSE_MODEL_PATH : str = "mediapipe_models/pose_landmarker_full.task"

latest_pose_result: Optional[PoseLandmarkerResult] = None

def handle_pose_result(
        result,
        output_image: mp.Image,
        timestamp_ms: int,
) -> None:
    global latest_pose_result
    latest_pose_result = result

pose_options : PoseLandmarkerOptions = PoseLandmarkerOptions(
    base_options = mp.tasks.BaseOptions(model_asset_path = POSE_MODEL_PATH),
    running_mode = RunningMode.LIVE_STREAM,
    result_callback = handle_pose_result
)

pose_landmarker = PoseLandmarker.create_from_options(pose_options)

# Color code landmarks

def get_color(idx: int) -> Tuple[int, int, int]:
    if idx in [0, 1, 5, 9, 13, 17]:
        return (255, 255, 255)
    if idx <= 4:
        return (255, 99, 99)
    if idx <= 8:
        return (255, 211, 99)
    if idx <= 12:
        return (149, 212, 123)
    if idx <= 16:
        return (141, 187, 224)
    if idx <= 20: 
        return (108, 93, 156)

# === WEBCAM LOOP ===

cap: cv2.VideoCapture = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

def to_mp_image(frame: np.ndarray) -> mp.Image:
    """
    Convert an OpenCV BGR frame to a MediaPipe Image (RGB).
    Ensures correct format for detection.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame for MediaPipe
        mp_image: mp.Image = to_mp_image(frame)

        # # Flip for mirrored view
        # frame = cv2.flip(frame, 1)

        timestamp_ms: int = int(time.time() * 1000)

        # Send frame async to MediaPipe
        hand_landmarker.detect_async(mp_image, timestamp_ms)
        pose_landmarker.detect_async(mp_image, timestamp_ms)

        # Draw landmarks if available
        if latest_hand_result and latest_hand_result.hand_landmarks:
            height, width = frame.shape[:2]

            for idx, hand_landmarks in enumerate(
                latest_hand_result.hand_landmarks
            ):
                # Handedness label
                hand_label: str = (
                    latest_hand_result.handedness[idx][0].category_name
                )

                # Draw landmark points
                for landmark_idx, lm in enumerate(hand_landmarks):

                    x: int = int(lm.x * width)
                    y: int = int(lm.y * height)
                    cv2.circle(frame, (x, y), 5, get_color(landmark_idx), -1)
                    cv2.putText(
                        frame,
                        str(landmark_idx),
                        (x, y),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.75,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                # Draw connections
                for conn in HandLandmarksConnections.HAND_CONNECTIONS:
                    start_lm = hand_landmarks[conn.start]
                    end_lm = hand_landmarks[conn.end]
                    sx, sy = int(start_lm.x * width), int(start_lm.y * height)
                    ex, ey = int(end_lm.x * width), int(end_lm.y * height)
                    cv2.line(frame, (sx, sy), (ex, ey), get_color(conn.end), 2)

                # Draw label text
                wrist = hand_landmarks[0]
                lx, ly = int(wrist.x * width), int(wrist.y * height) - 10
                cv2.putText(
                    frame,
                    hand_label,
                    (int(lx*1.1), int(ly*0.9)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            # Draw in coordinates
            cv2.putText(
                frame,
                str(latest_hand_result.hand_landmarks[0][0].z),
                (10, 10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # Draw landmarks if available
        if latest_pose_result and latest_pose_result.pose_landmarks:
            height, width = frame.shape[:2]

            for idx, pose_landmarks in enumerate(
                latest_pose_result.pose_landmarks
            ):

                # Draw landmark points
                for landmark_idx, lm in enumerate(pose_landmarks):

                    x: int = int(lm.x * width)
                    y: int = int(lm.y * height)
                    cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)

                # Draw connections
                for conn in PoseLandmarksConnections.POSE_LANDMARKS:
                    start_lm2 = pose_landmarks[conn.start]
                    end_lm2 = pose_landmarks[conn.end]
                    sx2, sy2 = int(start_lm2.x * width), int(start_lm2.y * height)
                    ex2, ey2 = int(end_lm2.x * width), int(end_lm2.y * height)
                    cv2.line(frame, (sx2, sy2), (ex2, ey2), (0, 255, 0), 2)

        cv2.imshow("Landmarks", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    hand_landmarker.close()
    pose_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
