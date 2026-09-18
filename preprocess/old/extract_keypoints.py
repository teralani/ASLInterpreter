import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import PoseLandmarker, HandLandmarker
import logging

logger = logging.getLogger(__name__)

pose = None
hand = None

POSE_JOINTS = 17
HAND_JOINTS = 21
TARGET_FRAMES = 30


def _init_models():
    global pose, hand
    if pose is None:
        pose = PoseLandmarker.create_from_model_path("mediapipe_models/pose_landmarker_full.task")
    if hand is None:
        hand = HandLandmarker.create_from_model_path("mediapipe_models/hand_landmarker.task")


def normalize_hands(hand_result):
    """
    Ensures that the hand that is outputed by the MediaPipe model follows the same order each time.

    (None, None)

    (left_landmarks, None)

    (None, right_landmarks)

    (left_landmarks, right_landmarks)
    
    """
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
    if total <= 0:
        return []
    if total <= T:
        # return available frames; caller will pad
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

    if len(raw_frames) == 0:
        logger.debug("No frames read from %s", video_path)
        # return empty array with expected joint dimension (0, J, 3)
        J = POSE_JOINTS + HAND_JOINTS * 2
        return np.zeros((0, J, 3), dtype=np.float32)

    indices = sample_frames(len(raw_frames), TARGET_FRAMES)
    processed = []
    masks = []
    valid_count = 0

    try:
        _init_models()
    except Exception as e:
        logger.exception("Failed to initialize MediaPipe models: %s", e)
        J = POSE_JOINTS + HAND_JOINTS * 2
        return np.zeros((0, J, 3), dtype=np.float32)

    for idx in indices:
        frame = raw_frames[idx]

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = np.ascontiguousarray(frame_rgb)
        except Exception:
            frame_rgb = frame

        try:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )

            pose_res = pose.detect(mp_image)
            hand_res = hand.detect(mp_image)
        except Exception:
            logger.exception("MediaPipe detection failed on frame %s of %s", idx, video_path)
            pose_res = type("R", (), {"pose_landmarks": None})()
            hand_res = type("R", (), {"hand_landmarks": None, "handedness": []})()

        frame_kp = []
        frame_mask = []

        # ---- POSE ----
        if getattr(pose_res, "pose_landmarks", None):
            for lm in pose_res.pose_landmarks[0][:POSE_JOINTS]:
                frame_kp.append([lm.x, lm.y, lm.z])
                frame_mask.append(1.0)
        else:
            frame_kp.extend([[0.0, 0.0, 0.0]] * POSE_JOINTS)
            frame_mask.extend([0.0] * POSE_JOINTS)

        left, right = normalize_hands(hand_res)

        if left:
            for lm in left:
                frame_kp.append([lm.x, lm.y, lm.z])
                frame_mask.append(1.0)
        else:
            frame_kp.extend([[0.0, 0.0, 0.0]] * HAND_JOINTS)
            frame_mask.extend([0.0] * HAND_JOINTS)

        if right:
            for lm in right:
                frame_kp.append([lm.x, lm.y, lm.z])
                frame_mask.append(1.0)
        else:
            frame_kp.extend([[0.0, 0.0, 0.0]] * HAND_JOINTS)
            frame_mask.extend([0.0] * HAND_JOINTS)

        processed.append(frame_kp)
        masks.append(frame_mask)
        if any(any(coord != 0.0 for coord in lm) for lm in frame_kp):
            valid_count += 1

    arr = np.array(processed, dtype=np.float32)
    mask_arr = np.array(masks, dtype=np.float32)  # (F, J)

    if arr.shape[0] > 0:
        # arr (F, J, 3)
        F, J, C = arr.shape

        # valid entries mask per (frame, joint)
        valid = (np.abs(arr).sum(axis=2) > 0)  # (F, J)

        # forward-fill
        for f in range(1, F):
            miss = ~valid[f]
            if miss.any():
                arr[f, miss] = arr[f - 1, miss]
                valid[f, miss] = valid[f - 1, miss]

        # backward-fill
        for f in range(F - 2, -1, -1):
            miss = ~valid[f]
            if miss.any():
                arr[f, miss] = arr[f + 1, miss]
                valid[f, miss] = valid[f + 1, miss]
    logger.debug("Extracted %d/%d valid frames from %s", valid_count, len(indices), video_path)

    return arr, mask_arr
