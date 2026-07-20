from pathlib import Path
import sys
import cv2
import numpy as np
import mediapipe as mp

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
import logging

import itertools

# ensure repository root is on sys.path so imports of local modules succeed
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from old import extract_keypoints as ek

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/videos")
OUT_DIR = Path("data/test_processed")

pose = None
hand = None

POSE_JOINTS = set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, # Face
                        11, 12,      # Shoulders
                        13, 14,      # Elbows
                        15, 16,      # Wrists
                        23, 24       # Hips
                        ])

def _init_models():
    global pose, hand
    if pose is None:
        pose = PoseLandmarker.create_from_model_path("mediapipe_models/pose_landmarker_full.task")
    if hand is None:
        # hand = HandLandmarker.create_from_model_path("mediapipe_models/hand_landmarker.task")
        hand_options = HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="mediapipe_models/hand_landmarker.task"),
            num_hands=2,
            running_mode = RunningMode.IMAGE,
        )
        hand = HandLandmarker.create_from_options(hand_options)

def get_hand_connections(n: int):
    """
    Standard hand topology template (thumb, index, middle, ring, pinky).
    
    n = 20 returns all joints, n = 5 only returns wrist and thumb

    Args:
    n -- returns the first n joints in the hand (int)
    """
    # 
    template = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    return [(a, b) for a, b in template if a < n and b < n]

def draw_landmarks_on_frame(frame, pose_res, hand_res):
    """
    Takes a video frame and annotates it with what the pose & hand MediaPipe models output.

    Args:
    frame -- frame of a video in a NumPy array of shape (height, width, 3)
    """
    h, w = frame.shape[:2]
    out = frame.copy()

    # test to see if a pose was detected
    if not getattr(pose_res, "pose_landmarks"):
        raise RuntimeWarning("No pose landmarks detected")
    
    for index, landmark in enumerate(pose_res.pose_landmarks[0]):
        if index not in POSE_JOINTS: continue

        # reverse normalization of landmarks
        x = int(landmark.x * w)
        y = int(landmark.y * h)

        cv2.circle(img = out, center = (x, y), radius = 2, color = (0, 255, 0), thickness = -1)
    
    left, right = ek.normalize_hands(hand_res)

    hand_connections = get_hand_connections(20)

    if left:
        for landmark in left:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(img = out, center = (x, y), radius = 2, color = (255, 255, 0), thickness = -1)
        for a, b in hand_connections:
            x_a = int(left[a].x * w)
            y_a = int(left[a].y * h)
            x_b = int(left[b].x * w)
            y_b = int(left[b].y * h)
            cv2.line(img = out, pt1 = (x_a, y_a), pt2 = (x_b, y_b), color = (0, 255, 255), thickness = 1)

    if right:
        for landmark in right:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(img = out, center = (x, y), radius = 2, color = (255, 255, 0), thickness = -1)
        for a, b in hand_connections:
            x_a = int(right[a].x * w)
            y_a = int(right[a].y * h)
            x_b = int(right[b].x * w)
            y_b = int(right[b].y * h)
            cv2.line(img = out, pt1 = (x_a, y_a), pt2 = (x_b, y_b), color = (0, 255, 255), thickness = 1)
    
    return out
    

def draw_landmarks_on_frames(vid_num = 0):
    video = next(itertools.islice(RAW_DIR.glob("*.mp4"), vid_num, vid_num+1), None)
    print(type(video))
    print(f"Processing {video.name}")

    _init_models()

    cap = cv2.VideoCapture(str(video))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    SIZE = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(f"test_video_{video.name}", fourcc, FPS, SIZE)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        converted_frame = np.ascontiguousarray(converted_frame)

        mp_frame = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data = converted_frame
        )

        pose_res = pose.detect(mp_frame)
        hand_res = hand.detect(mp_frame)

        annotated = draw_landmarks_on_frame(frame, pose_res, hand_res)

        video_writer.write(annotated)

    cap.release()
    video_writer.release()


if __name__ == "__main__":
    draw_landmarks_on_frames(152)

# 22 didn't detect hands??

# TODO:
# Fix flickering when processing
# Fix jittering when processing

# Actual model should use hand_res.hand_world_landmarks and pose_res.pose_world_landmarks




