import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import PoseLandmarker, HandLandmarker

pose = PoseLandmarker.create_from_model_path("mediapipe_models/pose_landmarker_full.task")
hand = HandLandmarker.create_from_model_path("mediapipe_models/hand_landmarker.task")

POSE_JOINTS = 17
HAND_JOINTS = 21
TARGET_FRAMES = 30


def normalize_hands(hand_result):
    left = None
    right = None

    if hand_result.hand_landmarks:
        for lms, handedness in zip(
            hand_result.hand_landmarks,
            hand_result.handedness
        ):
            label = handedness[0].category_name.lower()
            if label == "left":
                left = lms
            elif label == "right":
                right = lms

    return left, right


def sample_frames(total, T):
    if total <= T:
        return list(range(total))
    step = total / T
    return [int(i * step) for i in range(T)]


def extract_video(video_path):
    cap = cv2.VideoCapture(video_path)
    raw_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)

    cap.release()

    indices = sample_frames(len(raw_frames), TARGET_FRAMES)
    processed = []

    for idx in indices:
        frame = raw_frames[idx]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        pose_res = pose.detect(mp_image)
        hand_res = hand.detect(mp_image)

        frame_kp = []

        # ---- POSE ----
        if pose_res.pose_landmarks:
            for lm in pose_res.pose_landmarks[0][:POSE_JOINTS]:
                frame_kp.append([lm.x, lm.y, lm.z])
        else:
            frame_kp.extend([[0.0, 0.0, 0.0]] * POSE_JOINTS)

        # ---- HANDS ----
        left, right = normalize_hands(hand_res)

        if left:
            for lm in left:
                frame_kp.append([lm.x, lm.y, lm.z])
        else:
            frame_kp.extend([[0.0, 0.0, 0.0]] * HAND_JOINTS)

        if right:
            for lm in right:
                frame_kp.append([lm.x, lm.y, lm.z])
        else:
            frame_kp.extend([[0.0, 0.0, 0.0]] * HAND_JOINTS)

        processed.append(frame_kp)

    return np.array(processed, dtype=np.float32)
