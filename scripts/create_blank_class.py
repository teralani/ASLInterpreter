import sys
import json
import shutil
import random
import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))

import numpy as np
import cv2
import mediapipe as mp

from preprocess.extract import (
    extract_video,
    NUM_POSE_JOINTS,
    NUM_HAND_JOINTS,
    _build_pose_options,
    _build_hand_options,
    _extract_frame_keypoints,
)
from utils.skeleton import WINDOW_FRAMES, FRAME_INTERVAL_MS


MODE = "long_video_slices"

BLANK_VIDEOS_DIR = Path("data/blank_footage_short")
LONG_VIDEOS_DIR = Path("data/blank_footage_long")
WINDOW_STRIDE_FRAMES = WINDOW_FRAMES // 2

OUT_DIR = Path("data/blank_processed")
COMBINED_DICT_PATH = Path("data/combined_dict.json")

BLANK_GLOSS = "blank"
BLANK_ID_PREFIX = "blank_"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42

MOTION_THRESHOLD = 0.02
HAND_WEIGHT = 2.0


def extract_dense_landmarks(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
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

    if not raw_frames:
        return np.empty((0, 0, 3), dtype=np.float32), np.empty((0, 0), dtype=np.float32), ms_per_frame

    from mediapipe.tasks.python.vision import PoseLandmarker, HandLandmarker
    pose_landmarker = PoseLandmarker.create_from_options(_build_pose_options())
    hand_landmarker = HandLandmarker.create_from_options(_build_hand_options())

    dense_kp, dense_mask = [], []
    last_timestamp_ms = -1

    with pose_landmarker, hand_landmarker:
        for i, frame in enumerate(raw_frames):
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
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                pose_res = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                hand_res = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception:
                pose_res = type("R", (), {"pose_landmarks": None, "pose_world_landmarks": None})()
                hand_res = type("R", (), {"hand_landmarks": None, "hand_world_landmarks": None, "handedness": []})()

            frame_kp, frame_mask = _extract_frame_keypoints(pose_res, hand_res)
            dense_kp.append(frame_kp)
            dense_mask.append(frame_mask)

    return np.array(dense_kp, dtype=np.float32), np.array(dense_mask, dtype=np.float32), ms_per_frame


def resample_to_frame_interval(dense_kp: np.ndarray, dense_mask: np.ndarray, ms_per_frame: float):
    total_native_frames = dense_kp.shape[0]
    total_duration_ms = total_native_frames * ms_per_frame
    num_target_frames = int(total_duration_ms // FRAME_INTERVAL_MS)

    J = dense_kp.shape[1]
    out_kp = np.zeros((num_target_frames, J, 3), dtype=np.float32)
    out_mask = np.zeros((num_target_frames, J), dtype=np.float32)

    for k in range(num_target_frames):
        target_ms = k * FRAME_INTERVAL_MS
        src_idx = int(round(target_ms / ms_per_frame))
        if 0 <= src_idx < total_native_frames:
            out_kp[k] = dense_kp[src_idx]
            out_mask[k] = dense_mask[src_idx]

    return out_kp, out_mask


def slice_windows(resampled_kp: np.ndarray, resampled_mask: np.ndarray, stride: int):
    T = resampled_kp.shape[0]
    if T < WINDOW_FRAMES:
        return
    for start in range(0, T - WINDOW_FRAMES + 1, stride):
        yield start, resampled_kp[start:start + WINDOW_FRAMES], resampled_mask[start:start + WINDOW_FRAMES]


def motion_energy(arr: np.ndarray, mask: np.ndarray, hand_weight: float = HAND_WEIGHT) -> float:
    if arr.shape[0] < 2:
        return 0.0

    weights = np.ones(arr.shape[1], dtype=np.float32)
    weights[NUM_POSE_JOINTS:] = hand_weight

    diff = arr[1:] - arr[:-1]
    disp = np.linalg.norm(diff, axis=-1)
    pair_mask = mask[1:] * mask[:-1]

    weighted = disp * pair_mask * weights
    valid_weight = (pair_mask * weights).sum(axis=1)
    per_frame_motion = np.divide(
        weighted.sum(axis=1), valid_weight,
        out=np.zeros(arr.shape[0] - 1, dtype=np.float32),
        where=valid_weight > 0,
    )
    return float(np.mean(per_frame_motion))


def load_combined_dict(path: Path) -> list:
    if not path.exists():
        print(f"WARNING: {path} not found -- starting a new combined_dict list.")
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_combined_dict(data: list, path: Path):
    if path.exists():
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(f"{path.stem}_backup_{stamp}{path.suffix}")
        shutil.copy2(path, backup_path)
        print(f"Backed up existing {path} -> {backup_path}")
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {path}")


def assign_split(rng: random.Random) -> str:
    r = rng.random()
    if r < TRAIN_FRAC:
        return "train"
    elif r < TRAIN_FRAC + VAL_FRAC:
        return "val"
    else:
        return "test"


def process_and_register(new_instances: list):
    if not new_instances:
        print("No new blank instances to add -- combined_dict.json left unchanged.")
        return

    data = load_combined_dict(COMBINED_DICT_PATH)

    existing_entry = next((e for e in data if e.get("gloss") == BLANK_GLOSS), None)
    if existing_entry is None:
        data.append({"gloss": BLANK_GLOSS, "instances": new_instances})
        print(f"Added new '{BLANK_GLOSS}' gloss entry with {len(new_instances)} instances.")
    else:
        existing_ids = {inst["video_id"] for inst in existing_entry["instances"]}
        added = 0
        for inst in new_instances:
            if inst["video_id"] not in existing_ids:
                existing_entry["instances"].append(inst)
                added += 1
        print(f"Merged into existing '{BLANK_GLOSS}' gloss entry: "
              f"{added} new instances added ({len(new_instances) - added} were already present).")

    save_combined_dict(data, COMBINED_DICT_PATH)

    entry = next(e for e in data if e.get("gloss") == BLANK_GLOSS)
    counts = {"train": 0, "val": 0, "test": 0}
    for inst in entry["instances"]:
        counts[inst["split"]] = counts.get(inst["split"], 0) + 1
    print(f"\n'{BLANK_GLOSS}' gloss now has {len(entry['instances'])} total instances: {counts}")


def run_short_clips():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not BLANK_VIDEOS_DIR.exists():
        print(f"BLANK_VIDEOS_DIR does not exist: {BLANK_VIDEOS_DIR}\n"
              f"Create it and add your idle/rest .mp4 clips first.")
        sys.exit(1)

    video_paths = sorted(BLANK_VIDEOS_DIR.glob("*.mp4"))
    if not video_paths:
        print(f"No .mp4 files found in {BLANK_VIDEOS_DIR}.")
        sys.exit(1)

    print(f"Found {len(video_paths)} candidate blank clips in {BLANK_VIDEOS_DIR}\n")

    rng = random.Random(RANDOM_SEED)
    new_instances = []
    kept, rejected_motion, rejected_empty = 0, 0, 0

    for video_path in video_paths:
        video_id = f"{BLANK_ID_PREFIX}{video_path.stem}"
        print(f"Processing {video_path.name} -> {video_id}")

        arr, mask_arr = extract_video(video_path)

        if arr.size == 0 or np.count_nonzero(arr) == 0:
            print(f"  SKIP: extraction produced no keypoints (same failure "
                  f"condition extract_all_videos uses)")
            rejected_empty += 1
            continue

        energy = motion_energy(arr, mask_arr)
        if energy > MOTION_THRESHOLD:
            print(f"  SKIP: motion energy {energy:.4f} exceeds threshold "
                  f"{MOTION_THRESHOLD} -- not actually low-motion")
            rejected_motion += 1
            continue

        np.save(OUT_DIR / f"{video_id}.npy", arr)
        np.save(OUT_DIR / f"{video_id}_mask.npy", mask_arr)
        print(f"  KEEP: motion energy {energy:.4f}, saved to {OUT_DIR}")

        new_instances.append({"split": assign_split(rng), "video_id": video_id})
        kept += 1

    print(f"\n{kept} kept, {rejected_motion} rejected (too much motion), "
          f"{rejected_empty} rejected (empty extraction)")

    process_and_register(new_instances)


def run_long_video_slices():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not LONG_VIDEOS_DIR.exists():
        print(f"LONG_VIDEOS_DIR does not exist: {LONG_VIDEOS_DIR}\n"
              f"Create it and add your longer idle/rest recording(s) first.")
        sys.exit(1)

    video_paths = sorted(LONG_VIDEOS_DIR.glob("*.mp4"))
    if not video_paths:
        print(f"No .mp4 files found in {LONG_VIDEOS_DIR}.")
        sys.exit(1)

    print(f"Found {len(video_paths)} long recording(s) in {LONG_VIDEOS_DIR}\n"
          f"Window: {WINDOW_FRAMES} frames, stride: {WINDOW_STRIDE_FRAMES} frames "
          f"({100 * WINDOW_STRIDE_FRAMES / WINDOW_FRAMES:.0f}% step)\n")

    rng = random.Random(RANDOM_SEED)
    new_instances = []
    kept, rejected_motion = 0, 0

    for video_path in video_paths:
        video_split = assign_split(rng)

        print(f"Processing {video_path.name} (-> split: {video_split})")
        dense_kp, dense_mask, ms_per_frame = extract_dense_landmarks(video_path)

        if dense_kp.size == 0:
            print(f"  SKIP: no frames read or no keypoints extracted at all")
            continue

        resampled_kp, resampled_mask = resample_to_frame_interval(dense_kp, dense_mask, ms_per_frame)
        print(f"  {dense_kp.shape[0]} native frames -> {resampled_kp.shape[0]} resampled frames")

        clip_kept, clip_rejected = 0, 0
        for start_idx, window_kp, window_mask in slice_windows(resampled_kp, resampled_mask, WINDOW_STRIDE_FRAMES):
            energy = motion_energy(window_kp, window_mask)
            if energy > MOTION_THRESHOLD:
                clip_rejected += 1
                continue

            video_id = f"{BLANK_ID_PREFIX}{video_path.stem}_{start_idx:06d}"
            np.save(OUT_DIR / f"{video_id}.npy", window_kp)
            np.save(OUT_DIR / f"{video_id}_mask.npy", window_mask)
            new_instances.append({"split": video_split, "video_id": video_id})
            clip_kept += 1

        print(f"  {clip_kept} windows kept, {clip_rejected} rejected (too much motion)")
        kept += clip_kept
        rejected_motion += clip_rejected

    print(f"\n{kept} total windows kept, {rejected_motion} rejected across all recordings")

    process_and_register(new_instances)


def main():
    if MODE == "short_clips":
        run_short_clips()
    elif MODE == "long_video_slices":
        run_long_video_slices()
    else:
        raise ValueError(f"Unknown MODE: {MODE!r} -- use 'short_clips' or 'long_video_slices'")


if __name__ == "__main__":
    main()