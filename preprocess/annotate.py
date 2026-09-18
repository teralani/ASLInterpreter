from pathlib import Path
import os
import cv2
import numpy as np

from extract import (POSE_JOINT_INDICES, NUM_POSE_JOINTS, NUM_HAND_JOINTS)


POSE_SLICE = slice(0, NUM_POSE_JOINTS)
LEFT_HAND_SLICE = slice(NUM_POSE_JOINTS, NUM_POSE_JOINTS + NUM_HAND_JOINTS)
RIGHT_HAND_SLICE = slice(NUM_POSE_JOINTS + NUM_HAND_JOINTS, NUM_POSE_JOINTS + NUM_HAND_JOINTS * 2)

POSE_IDX_TO_LOCAL = {orig: local for local, orig in enumerate(POSE_JOINT_INDICES)}
 
LEFT_ELBOW_IDX = POSE_IDX_TO_LOCAL[13]
RIGHT_ELBOW_IDX = POSE_IDX_TO_LOCAL[14]
LEFT_POSE_WRIST_IDX = POSE_IDX_TO_LOCAL[15]
RIGHT_POSE_WRIST_IDX = POSE_IDX_TO_LOCAL[16]
 

LEFT_HAND_WRIST_IDX = LEFT_HAND_SLICE.start
RIGHT_HAND_WRIST_IDX = RIGHT_HAND_SLICE.start
 
POSE_COLOR = (255, 255, 255)# (white)
LEFT_HAND_COLOR = (0, 255, 0) # (green)
RIGHT_HAND_COLOR = (0, 165, 255) # (orange)
 
_FULL_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]
 
# trim unused joints (like feet, legs, etc.)
POSE_CONNECTIONS = [
    (POSE_IDX_TO_LOCAL[a], POSE_IDX_TO_LOCAL[b])
    for a, b in _FULL_POSE_CONNECTIONS
    if a in POSE_IDX_TO_LOCAL and b in POSE_IDX_TO_LOCAL
]
 
HAND_CONNECTIONS = [
    (0, 1), (1, 5), (9, 13), (13, 17), (5, 9), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]
 
 
def align_hands_to_wrists(arr, valid):
    """
    Translates each hand's landmarks so that the hand's own wrist (local index 0) coincides with the wrist reported by the pose landmarks 
    due to differing local indices.

    arr: (T, C, 3) keypoints array.
    valid: (T, C) bool array of which points are valid/present.
 
    Returns a (T, C, 3) array;
    """
    arr = arr.copy()
 
    for hand_slice, hand_wrist_idx, pose_wrist_idx in (
        (LEFT_HAND_SLICE, LEFT_HAND_WRIST_IDX, LEFT_POSE_WRIST_IDX),
        (RIGHT_HAND_SLICE, RIGHT_HAND_WRIST_IDX, RIGHT_POSE_WRIST_IDX),
    ):
        can_align = valid[:, hand_wrist_idx] & valid[:, pose_wrist_idx]
        if not np.any(can_align):
            continue
        offsets = arr[can_align, pose_wrist_idx] - arr[can_align, hand_wrist_idx]  # (n, 3)
        arr[can_align, hand_slice] += offsets[:, None, :]
 
    return arr
 
 
def keypoints_to_video(
    arr,
    mask=None,
    output_path="keypoints.mp4",
    fps=15,
    frame_size=(640, 480),
    point_radius=4,
    line_thickness=2,
    margin_frac=0.1,
    align_hands=True,
):
    """
    arr: np.ndarray, shape (T, C, 3) w/ 3D keypoints (x, y, z) per frame
    mask: optional np.ndarray, shape (T, C) (1.0 = valid; 0.0 = missing).
    output_path: path to write the .mp4 file.
    fps: playback frame rate of the output video.
    frame_size: (width, height) of the output video in pixels.
    point_radius: radius of each drawn keypoint.
    line_thickness: thickness.
    margin_frac: fraction of the frame reserved as empty margin on each side..
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"got {arr.shape} instead of (T, C, 3)")
 
    T, C, _ = arr.shape
    W, H = frame_size
 
    expected_C = NUM_POSE_JOINTS + NUM_HAND_JOINTS * 2
    if C != expected_C:
        raise ValueError()
 
    if mask is None:
        valid = ~np.all(arr == 0.0, axis=-1)
    else:
        valid = np.asarray(mask).astype(bool)
        if valid.shape != (T, C):
            raise ValueError(f"mask shape {valid.shape} doesn't match arr's (T, C) = {(T, C)}")
 
    if align_hands:
        arr = align_hands_to_wrists(arr, valid)
 
    valid_xy = arr[..., :2][valid]
    if valid_xy.size == 0:
        raise ValueError()
 
    x_min, y_min = valid_xy.min(axis=0)
    x_max, y_max = valid_xy.max(axis=0)
 
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    span = max(x_span, y_span)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
 
    usable_w = W * (1 - 2 * margin_frac)
    usable_h = H * (1 - 2 * margin_frac)
    scale = min(usable_w, usable_h) / span
 
    def project(x, y):
        px = int(round(W / 2 + (x - x_center) * scale))
        py = int(round(H / 2 + (y - y_center) * scale))
        return px, py
 
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(os.path.join("data", "skeleton", output_path)), fourcc, fps, (W, H))
    if not writer.isOpened():
        raise IOError(f"Couldn't open VideoWriter for {output_path}")
 
    connection_groups = [
        (POSE_CONNECTIONS, POSE_COLOR, POSE_SLICE.start),
        (HAND_CONNECTIONS, LEFT_HAND_COLOR, LEFT_HAND_SLICE.start),
        (HAND_CONNECTIONS, RIGHT_HAND_COLOR, RIGHT_HAND_SLICE.start),
    ]
 
    for t in range(T):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
 
        for connections, color, offset in connection_groups:
            for a, b in connections:
                ia, ib = offset + a, offset + b
                if valid[t, ia] and valid[t, ib]:
                    pa = project(arr[t, ia, 0], arr[t, ia, 1])
                    pb = project(arr[t, ib, 0], arr[t, ib, 1])
                    cv2.line(frame, pa, pb, color, line_thickness)
 
        for c in range(C):
            if not valid[t, c]:
                continue
            x, y, _ = arr[t, c]
            px, py = project(x, y)
            if 0 <= px < W and 0 <= py < H:
                if POSE_SLICE.start <= c < POSE_SLICE.stop:
                    color = POSE_COLOR
                elif LEFT_HAND_SLICE.start <= c < LEFT_HAND_SLICE.stop:
                    color = LEFT_HAND_COLOR
                else:
                    color = RIGHT_HAND_COLOR
                cv2.circle(frame, (px, py), point_radius, color, -1)
 
        writer.write(frame)
 
    writer.release()
    print(f"{T} frames to {output_path}")