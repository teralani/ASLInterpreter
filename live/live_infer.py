import sys
import time
from collections import deque
from pathlib import Path
 
import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions,
    PoseLandmarker, PoseLandmarkerOptions, RunningMode,
)
 
sys.path.append(str(Path(__file__).parent.parent.resolve()))
 
from preprocess.extract import _extract_frame_keypoints, POSE_MODEL_PATH, HAND_MODEL_PATH
from utils.skeleton import (
    WINDOW_FRAMES, FRAME_INTERVAL_MS, NUM_JOINTS, NUM_POSE_JOINTS,
    LEFT_HAND_START, RIGHT_HAND_START, NUM_HAND_JOINTS, build_skeleton_adjacency,
)
from model.st_pose_model import STPoseModel
from build_gloss import load_gloss_vocab
 
 
# to prioritize finger tip movments (since they move the most)
FINGER_TIP_LOCAL_IDX = [4, 8, 12, 16, 20]
 
 
def _build_motion_weights(pose_weight=0.15, hand_weight=5.0, fingertip_weight=12.0):
    """
    Per-joint weight vector for frame_motion_score using weights provided to prioritize finer movments.
    """
    weights = np.full(NUM_JOINTS, pose_weight, dtype=np.float32)
    for start in (LEFT_HAND_START, RIGHT_HAND_START):
        weights[start:start + NUM_HAND_JOINTS] = hand_weight
        for tip in FINGER_TIP_LOCAL_IDX:
            weights[start + tip] = fingertip_weight
    return weights
 
 
MOTION_WEIGHTS = _build_motion_weights()
 
 
def frame_motion_score(prev_kp, prev_mask, curr_kp, curr_mask, weights=MOTION_WEIGHTS):
    prev_kp = np.asarray(prev_kp, dtype=np.float32)
    curr_kp = np.asarray(curr_kp, dtype=np.float32)
    prev_mask = np.asarray(prev_mask, dtype=np.float32)
    curr_mask = np.asarray(curr_mask, dtype=np.float32)
 
    disp = np.linalg.norm(curr_kp - prev_kp, axis=-1)  # (J,)
    pair_mask = prev_mask * curr_mask                   # (J,)
 
    weighted = disp * pair_mask * weights
    valid_weight = (pair_mask * weights).sum()
    if valid_weight <= 0:
        return 0.0
    return float(weighted.sum() / valid_weight)
 
 
def fit_to_window(frames, masks, window_frames):
    """
    Places a captured, variable-length sign segment into a fixed `window_frames`-length slot, matching how offline clips were always
    resampled to WINDOW_FRAMES.
 
    - Segment shorter than window_frames: place at the start, zero-pad the remainder.
    - Segment longer than window_frames: uniformly subsample down to window_frames indices.
    """
    n = len(frames)
    j = np.asarray(frames[0]).shape[0]
 
    if n <= window_frames:
        out_kp = np.zeros((window_frames, j, 3), dtype=np.float32)
        out_mask = np.zeros((window_frames, j), dtype=np.float32)
        out_kp[:n] = np.asarray(frames, dtype=np.float32)
        out_mask[:n] = np.asarray(masks, dtype=np.float32)
        return out_kp, out_mask
 
    idx = np.linspace(0, n - 1, window_frames).round().astype(int)
    out_kp = np.asarray(frames, dtype=np.float32)[idx]
    out_mask = np.asarray(masks, dtype=np.float32)[idx]
    return out_kp, out_mask
 
 
def draw_hud(frame, state, motion, last_result, display_duration_s=3.0):
    """
    Draws a small status HUD and the most recent recognition result, held on screen for display_duration_s
    """
    h, w = frame.shape[:2]
 
    status_text = f"{state.upper()}  motion={motion:.4f}"
    status_color = (0, 200, 255) if state == "active" else (180, 180, 180)
    cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                status_color, 1, cv2.LINE_AA)
 
    if last_result is not None:
        text, color, ts = last_result
        if time.monotonic() - ts <= display_duration_s:
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
            x = (w - text_w) // 2
            y = h - 40
            cv2.rectangle(frame, (x - 15, y - text_h - 15), (x + text_w + 15, y + 15),
                          (0, 0, 0), -1)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        color, 2, cv2.LINE_AA)
 
 
def build_landmarkers():
    pose = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
    ))
    hand = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        num_hands=2,
        running_mode=RunningMode.VIDEO,
    ))
    return pose, hand
 
 
def load_model(checkpoint_path, adj, num_classes, device):
    model = STPoseModel(adj=adj, num_classes=num_classes, T=WINDOW_FRAMES, num_joints=NUM_JOINTS)
    state = torch.load(checkpoint_path, map_location=device)
 
    ckpt_out_features = state["cls.weight"].shape[0]
    if ckpt_out_features != num_classes:
        raise ValueError(
            f"{ckpt_out_features} output classes, {num_classes} loaded gloss vocab classes."
        )
 
    model.load_state_dict(state)
    model.to(device).eval()
    return model
 
 
def run_live(checkpoint_path, adj, idx_to_gloss, device="cpu",
             onset_threshold=0.02, offset_threshold=0.008,
             min_rest_frames=6, preroll_frames=5, prob_threshold=0.6,
             result_display_s=3.0):
    """
    onset_threshold / offset_threshold: motion-score levels that start/end a segment using motion weights
    min_rest_frames: consecutive low-motion frames required to conclude the sign has ended (debounces brief pauses within one sign).
    preroll_frames: frames of context kept from just before onset was detected so the very start of the sign isn't clipped.
    result_display_s: how long a recognized gloss stays on screen.
    """
    pose_landmarker, hand_landmarker = build_landmarkers()
    model = load_model(checkpoint_path, adj, num_classes=len(idx_to_gloss), device=device)
 
    cap = cv2.VideoCapture(0)
 
    preroll_kp = deque(maxlen=preroll_frames)
    preroll_mask = deque(maxlen=preroll_frames)
 
    prev_kp, prev_mask = None, None
    state = "idle"          # "idle" or "active"
    segment_kp, segment_mask = [], []
    rest_run = 0
    last_result = None
    motion = 0.0
 
    start_time = time.monotonic()
    target_dt = FRAME_INTERVAL_MS / 1000.0
 
    with pose_landmarker, hand_landmarker:
        while True:
            loop_start = time.monotonic()
 
            ret, frame = cap.read()
            if not ret:
                break
 
            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=np.ascontiguousarray(frame_rgb))
 
            pose_res = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_res = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
 
            curr_kp, curr_mask = _extract_frame_keypoints(pose_res, hand_res)
 
            if prev_kp is not None:
                motion = frame_motion_score(prev_kp, prev_mask, curr_kp, curr_mask)
            prev_kp, prev_mask = curr_kp, curr_mask
 
            if state == "idle":
                preroll_kp.append(curr_kp)
                preroll_mask.append(curr_mask)
 
                if motion >= onset_threshold:
                    state = "active"
                    segment_kp = list(preroll_kp)
                    segment_mask = list(preroll_mask)
                    rest_run = 0
 
            else:  # state == "active"
                segment_kp.append(curr_kp)
                segment_mask.append(curr_mask)
 
                if motion < offset_threshold:
                    rest_run += 1
                else:
                    rest_run = 0
 
                segment_done = rest_run >= min_rest_frames or len(segment_kp) >= WINDOW_FRAMES
 
                if segment_done:
                    window_kp, window_mask = fit_to_window(segment_kp, segment_mask, WINDOW_FRAMES)
                    probs = model.predict_window(window_kp, window_mask, device=device)
                    top_prob, top_idx = probs.max(dim=0)
 
                    if top_prob.item() >= prob_threshold:
                        text = f"{idx_to_gloss[top_idx.item()]} ({top_prob.item():.2f})"
                        color = (60, 220, 60)
                    else:
                        text = f"low confidence ({top_prob.item():.2f})"
                        color = (60, 60, 220)
                    last_result = (text, color, time.monotonic())
                    print(text)
 
                    state = "idle"
                    segment_kp, segment_mask = [], []
                    preroll_kp.clear()
                    preroll_mask.clear()
 
            draw_hud(frame, state, motion, last_result, display_duration_s=result_display_s)
            cv2.imshow("ASL live (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
 
            elapsed = time.monotonic() - loop_start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
 
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    adj = build_skeleton_adjacency()


    checkpoint_path = "model_files/best_model_w_blanks.pt"

    from build_gloss import load_gloss_vocab
    _, idx_to_gloss = load_gloss_vocab(Path(__file__).parent / "gloss_vocab.json")  

    run_live(checkpoint_path, adj, idx_to_gloss, device="xpu", prob_threshold=0.4, preroll_frames=8, result_display_s=1.5)

    p = Path("data/processed")
    print(p.resolve())
    print(len(list(p.glob("*.npy"))))
