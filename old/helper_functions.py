"""
2 Mediapipe Models: Pose + Hand

Hands -> 2 hands (21 joints each)

Pose -> 7 joints

NOSE
LEFT_SHOULDER
RIGHT_SHOULDER
LEFT_ELBOW
RIGHT_ELBOW
LEFT_WRIST
RIGHT_WRIST
LEFT_HIP
RIGHT_HIP

Total joints: 51 joints

Pose tensors: X = R[ T x J x C]
Number of frames (T): 30
Number of joints (J): 51
Coordinates per joint (C): 3
"""

import cv2
import numpy as np

POSE_JOINTS = [
    0,      # nose
    11, 12, # shoulders
    13, 14, # elbows
    15, 16, # wrists
    23, 24  # hips
]

HAND_JOINTS = list(range(21))

J = len(POSE_JOINTS) + 2*len(HAND_JOINTS)
    # Pose Joints      Left + Right Hand Joints

def sample_frames(video_path, T=30):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, T).astype(int)
    idx_set = set(indices.tolist())

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i in idx_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        i += 1
    
    cap.release()
    return frames

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

POSE_MODEL_PATH: str = "models/pose_landmarker_full.task"
HAND_MODEL_PATH : str = "models/hand_landmarker.task"

base_options_pose = mp.tasks.BaseOptions(
    model_asset_path = POSE_MODEL_PATH,
    delegate = python.BaseOptions.Delegate.GPU
)

base_options_hand = mp.tasks.BaseOptions(
    model_asset_path = HAND_MODEL_PATH,
    delegate = python.BaseOptions.Delegate.GPU
)

pose_options = vision.PoseLandmarkerOptions(
    base_options = base_options_pose,
    running_mode = vision.RunningMode.IMAGE
)

hand_options = vision.HandLandmarkerOptions(
    base_options = base_options_hand,
    running_mode = vision.RunningMode.IMAGE,
    num_hands = 2
)

pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

def extract_keypoints(frame_rgb):
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    pose_result = pose_landmarker.detect(mp_image)
    hand_result = hand_landmarker.detect(mp_image)

    keypoints = []

    # pose joints

    if pose_result.pose_landmarks:
        lm = pose_result.pose_landmarks[0]
        for idx in POSE_JOINTS:
            p = lm[idx]
            keypoints.append([p.x, p.y, p.z])
    else:
        keypoints.extend([[0, 0, 0]] * len(POSE_JOINTS))

    # hand joints
    hands = hand_result.hand_landmarks or []

    for h in range(2):
        if h < len(hands):
            for p in hands[h]:
                keypoints.append([p.x, p.y, p.z])
        else:
            keypoints.extend([[0, 0, 0]] * len(HAND_JOINTS))
    
    return np.array(keypoints, dtype=np.float32)


"""
Takes the frame from the video using sample_frames()
Extracts keypoints from each frame using extract_keypoints()

Returns the sequence
"""
def process_video(video_path, T=30):
    frames = sample_frames(video_path, T)
    sequence = []

    for frame in frames:
        kp = extract_keypoints(frame)
        sequence.append(kp)

    return np.stack(sequence)  # (T, 51, 3)


import os
from tqdm import tqdm

def preprocess_data(video_dir, output_dir, T=30):
    os.makedirs(output_dir, exist_ok=True)

    for video in tqdm(os.listdir(video_dir)):
        if not video.endswith(".mp4"):
            continue

        path = os.path.join(video_dir, video)
        tensor = process_video(path, T)

        out = os.path.join(output_dir, video.replace(".mp4", ".npy"))
        np.save(out, tensor)    # Outpute: [T x 51 x 3]

if __name__ == '__main__':
    # print(os.listdir("data/videos"))

    preprocess_data("data/videos", "data/processed_data")