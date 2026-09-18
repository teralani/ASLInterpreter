from pathlib import Path
import cv2
import numpy as np
import numpy.typing as npt
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
    PoseLandmarker,
    PoseLandmarkerOptions,
)
import logging
from tqdm import tqdm

# prevents mediapipe warning outputs
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"

import mediapipe as mp

import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from utils.skeleton import NUM_HAND_JOINTS, POSE_JOINT_INDICES, NUM_POSE_JOINTS, WINDOW_DURATION_MS, WINDOW_FRAMES, FRAME_INTERVAL_MS
print(POSE_JOINT_INDICES)

ASL_VIDEOS = Path("data/ASL_Citizen/videos")
WLASL_VIDEOS = Path("data/videos")

RAW_DIR = ASL_VIDEOS
OUT_DIR = Path("data/processed")
FAILED_LOG = Path("data/failed_videos.txt")
REPROCESS_LOG = Path("data/reprocess_log.txt")

logger = logging.getLogger(__name__)
 
POSE_MODEL_PATH = "mediapipe_models/pose_landmarker_full.task"
HAND_MODEL_PATH = "mediapipe_models/hand_landmarker.task"

def _build_pose_options():
    return PoseLandmarkerOptions(
        base_options = mp.tasks.BaseOptions(model_asset_path = "mediapipe_models/pose_landmarker_full.task"),
        running_mode = RunningMode.VIDEO 
    )

def _build_hand_options():
    return HandLandmarkerOptions(
        base_options = mp.tasks.BaseOptions(model_asset_path="mediapipe_models/hand_landmarker.task"),
        num_hands = 2,
        running_mode = RunningMode.VIDEO,
    )

def normalize_hands(hand_result):
    """
    Ensures that the hand that is outputed by the MediaPipe model follows the same order each time.

    (None, None)                 # no hands

    (left_landmarks, None)       # only left hand

    (None, right_landmarks)      # only right hand

    (left_landmarks, right_landmarks)  # both hands
    
    """
    left = None
    right = None

    if hand_result.hand_world_landmarks:
        for lms, handedness in zip(
            hand_result.hand_world_landmarks,
            hand_result.handedness
        ):
            label = handedness[0].category_name.lower()
            if label == "left":
                left = lms
            elif label == "right":
                right = lms

    return left, right

def _extract_frame_keypoints(pose_res, hand_res):
    """
    Builds one (J, 3) row & one (J,) mask row from a single frame's detection results.

    Args:
    pose_res: A result from a MediaPipe pose landmarker for a single frame
    hand_res: A result from a MediaPipe hand landmarker for a single frame
    """
    frame_kp = []
    frame_mask = []
    
    # Pose
    if getattr(pose_res, "pose_world_landmarks", None):
        landmarks = pose_res.pose_world_landmarks[0]
        for j in POSE_JOINT_INDICES:
            lm = landmarks[j]
            frame_kp.append([lm.x, lm.y, lm.z])
            frame_mask.append(1.0)
    else:
        frame_kp.extend([[0.0, 0.0, 0.0]] * NUM_POSE_JOINTS)
        frame_mask.extend([0.0] * NUM_POSE_JOINTS)

    # Hands
    left, right = normalize_hands(hand_res)

    if left:
        for lm in left:
            frame_kp.append([lm.x, lm.y, lm.z])
            frame_mask.append(1.0)
    else:
        frame_kp.extend([[0.0, 0.0, 0.0]] * NUM_HAND_JOINTS)
        frame_mask.extend([0.0] * NUM_HAND_JOINTS)

    if right:
        for lm in right:
            frame_kp.append([lm.x, lm.y, lm.z])
            frame_mask.append(1.0)
    else:
        frame_kp.extend([[0.0, 0.0, 0.0]] * NUM_HAND_JOINTS)
        frame_mask.extend([0.0] * NUM_HAND_JOINTS)
    
    return frame_kp, frame_mask

def _compute_window_start_ms(dense_kp, dense_mask, total_frames, ms_per_frame,
                              window_duration_ms=WINDOW_DURATION_MS,
                              hand_weight=2.0, smooth_frames=5):
    """
    Decides where the `window_duration_ms` sampling window should start within the full video instead of always starting at beginning (t = 0).
    (avoids cutting off crucial motion for a sign)
 
    If no motion is detected anywhere in the clip, falls back to starting at 0.0.
    """
    total_duration_ms = total_frames * ms_per_frame
 
    if total_duration_ms <= window_duration_ms or total_frames < 2:
        return 0.0
 
    kp = np.asarray(dense_kp, dtype=np.float32) # (T, J, 3)
    mask = np.asarray(dense_mask, dtype=np.float32) # (T, J)
 
    weights = np.ones(kp.shape[1], dtype=np.float32)
    weights[NUM_POSE_JOINTS:] = hand_weight 
 
    diff = kp[1:] - kp[:-1] # (T-1, J, 3)
    disp = np.linalg.norm(diff, axis=-1) # (T-1, J)
    pair_mask = mask[1:] * mask[:-1] # both frames must have the joint
 
    weighted = disp * pair_mask * weights
    valid_weight = (pair_mask * weights).sum(axis=1)
    motion = np.divide(
        weighted.sum(axis=1), valid_weight,
        out=np.zeros(kp.shape[0] - 1, dtype=np.float32),
        where=valid_weight > 0,
    )
 
    if smooth_frames > 1:
        kernel = np.ones(smooth_frames, dtype=np.float32) / smooth_frames
        motion = np.convolve(motion, kernel, mode="same")
 
    if motion.sum() <= 0:
        logger.debug("No motion detected; defaulting window start to 0ms")
        return 0.0
 
    pair_times_ms = (np.arange(len(motion)) + 0.5) * ms_per_frame
    center_ms = float(np.average(pair_times_ms, weights=motion))
 
    start_ms = center_ms - window_duration_ms / 2.0
    start_ms = max(0.0, min(start_ms, total_duration_ms - window_duration_ms))
 
    return start_ms
 
 
def _estimate_motion_window_from_pixels(raw_frames, ms_per_frame,
                                         window_duration_ms=WINDOW_DURATION_MS,
                                         downsample_size=(64, 48),
                                         smooth_frames=5, pad_ms=400.0):
    """ 
    Returns `(start_idx, end_idx)`: the range of raw frame indices to run MediaPipe on. 
    If the whole video already fits within one window, or is shorter than `window_duration_ms`, returns (0, total_frames).

    Centers window of `window_duration_ms` around the center of motion, padded by `pad_ms`. 

    Each frame is downsampled into `downsample_size` and made grayscale to store brightness/shape patterns.

    Pixelwise motion is detecting by convolving 2D array image using convolving kernel of length `smooth_frames`

    Center is computed and a rough start/end index are returned based on padding and window size.

    """
    total_frames = len(raw_frames)
    total_duration_ms = total_frames * ms_per_frame
    padded_window_ms = window_duration_ms + 2 * pad_ms
 
    if total_duration_ms <= padded_window_ms or total_frames < 2:
        return 0, total_frames
 
    small_gray = np.empty((total_frames, downsample_size[1], downsample_size[0]), dtype=np.float32)
    for i, frame in enumerate(raw_frames):
        small = cv2.resize(frame, downsample_size, interpolation=cv2.INTER_AREA)
        small_gray[i] = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
 
    motion = np.abs(small_gray[1:] - small_gray[:-1]).mean(axis=(1, 2))
 
    if smooth_frames > 1:
        kernel = np.ones(smooth_frames, dtype=np.float32) / smooth_frames
        motion = np.convolve(motion, kernel, mode="same")
 
    if motion.sum() <= 0:
        return 0, total_frames
 
    pair_times_ms = (np.arange(len(motion)) + 0.5) * ms_per_frame
    center_ms = float(np.average(pair_times_ms, weights=motion))
 
    core_start_ms = center_ms - window_duration_ms / 2.0
    core_start_ms = max(0.0, min(core_start_ms, total_duration_ms - window_duration_ms))
 
    start_ms = max(0.0, core_start_ms - pad_ms)
    end_ms = min(total_duration_ms, core_start_ms + window_duration_ms + pad_ms)
 
    start_idx = int(start_ms / ms_per_frame)
    end_idx = min(total_frames, int(np.ceil(end_ms / ms_per_frame)) + 1)
 
    return start_idx, end_idx


def extract_video(video_path) -> tuple[np.ndarray[tuple[float, float, float], np.dtype[np.float32]], np.ndarray[tuple[int, int], np.dtype[np.float32]]]:
    """
    Extracts and returns a pose & hand landmark numpy array and a numpy mask array using MediaPipe Hand and Pose Landmarker models.
    Both arrays are based on ***world landmarks***, as opposed to 2D landmarks, resulting in ***3D coordinates***. 
    Hence, the landmark array has the dimensions **(T, J, 3)** and the mask array has the dimensions **(T, J)** where T is the number of timesteps and J is the number of joints.
    

    If a joint is not detected for a given time step, its coordinates will be set as (0, 0, 0) and the mask array for the time step will be set as 0.

    """
    J = NUM_POSE_JOINTS + NUM_HAND_JOINTS * 2

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    ms_per_frame = 1000.0 / fps

    raw_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()

    total_frames = len(raw_frames)

    if total_frames == 0:
        logger.debug("No frames read from %s", video_path)
        return (np.zeros((WINDOW_FRAMES, J, 3), dtype = np.float32),
                np.zeros((WINDOW_FRAMES, J),    dtype = np.float32))
    
    try:
        pose_landmarker = PoseLandmarker.create_from_options(_build_pose_options())
        hand_landmarker = HandLandmarker.create_from_options(_build_hand_options())
    except Exception as e:
        logger.exception("Failed to initialize MediaPipe models: %s", e)
        return (np.zeros((WINDOW_FRAMES, J, 3), dtype = np.float32),
                np.zeros((WINDOW_FRAMES, J),    dtype = np.float32))
    
    start_idx, end_idx = _estimate_motion_window_from_pixels(raw_frames, ms_per_frame)
    dense_start_ms = start_idx * ms_per_frame

    if end_idx - start_idx < total_frames:
        logger.debug("Running MediaPipe on frames %d-%d of %d for %s (pixel motion pre-pass)",
                     start_idx, end_idx, total_frames, video_path)
    
    dense_kp = []
    dense_mask = []
    last_timestamp_ms = -1

    with pose_landmarker, hand_landmarker:
        for i in range(start_idx, end_idx):
            frame = raw_frames[i]
            
            timestamp_ms = int(round(i * ms_per_frame))
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = np.ascontiguousarray(frame_rgb)
            except Exception:
                frame_rgb = frame

            try:
                mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame_rgb)
                pose_res = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                hand_res = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception:
                logger.exception("MediaPipe detection failed on frame %s of %s", i, video_path)
                pose_res = type("R", (), {"pose_landmarks" : None, "pose_world_landmarks" : None})()
                hand_res = type("R", (), {"hand_landmarks" : None, "hand_world_landmarks" : None, "handedness" : []})()

            frame_kp, frame_mask = _extract_frame_keypoints(pose_res, hand_res)
            dense_kp.append(frame_kp)
            dense_mask.append(frame_mask)

    local_start_ms = _compute_window_start_ms(dense_kp, dense_mask, len(dense_kp), ms_per_frame)
    start_ms = dense_start_ms + local_start_ms

    # resampling for specified fps
    processed = []
    masks = []
    valid_count = 0
    num_dense = len(dense_kp)

    for k in range(WINDOW_FRAMES):
        target_ms = start_ms + k * FRAME_INTERVAL_MS
        local_target_ms = target_ms - dense_start_ms
        src_idx = int(round(local_target_ms / ms_per_frame))

        if 0 <= src_idx < total_frames:
            frame_kp = dense_kp[src_idx]
            frame_mask = dense_mask[src_idx]
        else:
            frame_kp = [[0.0, 0.0, 0.0]] * J
            frame_mask = [0.0] * J
        
        processed.append(frame_kp)
        masks.append(frame_mask)

        if any(any(coord != 0.0 for coord in lm) for lm in frame_kp):
            valid_count += 1
    
    arr = np.array(processed,  dtype=np.float32)
    mask_arr = np.array(masks, dtype=np.float32)

    logger.debug("Extracted %d/%d valid window slots from %s (window start %.0fms of %.0fms total, "
                 "MediaPipe ran on %d/%d native frames)",
                 valid_count, WINDOW_FRAMES, video_path, start_ms, total_frames * ms_per_frame,
                 num_dense, total_frames)
    
    return arr, mask_arr

def extract_all_videos(dir_from_text = None, out_dir = OUT_DIR, skip_past : str | None = None ):
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = []

    if dir_from_text != None:
        with open(dir_from_text, "r", encoding="utf-8") as file:
            videos_to_process = [Path(f"{RAW_DIR}/{x.strip()}.mp4") for x in file.readlines()]
    else:
        videos_to_process = RAW_DIR.glob("*.mp4")

    hasnt_reached = True

    for video_path in tqdm(videos_to_process, desc="Processing videos"):

        if skip_past and hasnt_reached:
            if video_path.name == skip_past:
                hasnt_reached = False
                print(f"Skipping past {skip_past}")

            continue

        tqdm.write(f"Processing {video_path.name}")

        arr, mask_arr = extract_video(video_path)

        if arr.size == 0 or np.count_nonzero(arr) == 0:
            logger.warning("Extraction produced no keypoints for %s", video_path.name)
            failed.append(video_path.stem + ".mp4")
            continue

        np.save(f"{out_dir}/{video_path.stem}.npy", arr)
        np.save(f"{out_dir}/{video_path.stem}_mask.npy", mask_arr)

        if failed:
            FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)
            with FAILED_LOG.open("w") as f:
                for name in failed:
                    f.write(name + "\n")
            tqdm.write(f"Logged {len(failed)} failed videos to {FAILED_LOG}")

    
    print(f"{len(failed)} failed videos.")

if __name__ == "__main__":
    if input("Task 1? (Y/N): ").lower().strip() == "y":

        # check_length()
        extract_all_videos(out_dir=OUT_DIR, skip_past = "7695266259775881-DIAMOND.mp4")
    else:
        inp = input("File name or video number (1 / 2)? ").strip()
        if inp == "2":
            vid_num = int(input("Input a video number: "))
            import itertools
            video_stem = next(itertools.islice(RAW_DIR.glob("*.mp4"), vid_num, vid_num+1), None)
        # arr, mask_arr = extract_video(video_path)
        elif inp == "1":
            vid_num = input("Input a video number: ")
            video_stem = vid_num

        else:
            print("Please input a valid number")
             
        print(video_stem)

        # video_stem = "18720"

        arr = np.load(f"data/processed/{video_stem}.npy", allow_pickle=True)
        mask_arr = np.load(f"data/processed/{video_stem}_mask.npy", allow_pickle=True)

        print(arr.shape)

        from annotate import keypoints_to_video

        cap = cv2.VideoCapture(f"{RAW_DIR}/{video_stem}.mp4")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        keypoints_to_video(arr, mask_arr, output_path=f"skeleton_{video_stem}.mp4", fps=fps)


