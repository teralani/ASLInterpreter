from pathlib import Path
import os
import cv2
import numpy as np

from extract import (POSE_JOINT_INDICES,
                    NUM_POSE_JOINTS,
                    NUM_HAND_JOINTS)


# Positions within the pose sub-array: 
POSE_SLICE = slice(0, NUM_POSE_JOINTS)
LEFT_HAND_SLICE = slice(NUM_POSE_JOINTS, NUM_POSE_JOINTS + NUM_HAND_JOINTS)
RIGHT_HAND_SLICE = slice(NUM_POSE_JOINTS + NUM_HAND_JOINTS, NUM_POSE_JOINTS + NUM_HAND_JOINTS * 2)
 
# Map each original MediaPipe pose landmark index to its position within our
# trimmed pose sub-array, e.g. POSE_IDX_TO_LOCAL[15] = local index of the left wrist.
POSE_IDX_TO_LOCAL = {orig: local for local, orig in enumerate(POSE_JOINT_INDICES)}
 
LEFT_ELBOW_IDX = POSE_IDX_TO_LOCAL[13]
RIGHT_ELBOW_IDX = POSE_IDX_TO_LOCAL[14]
LEFT_POSE_WRIST_IDX = POSE_IDX_TO_LOCAL[15]
RIGHT_POSE_WRIST_IDX = POSE_IDX_TO_LOCAL[16]
 
# Local index 0 of each hand block is that hand's own wrist landmark.
LEFT_HAND_WRIST_IDX = LEFT_HAND_SLICE.start
RIGHT_HAND_WRIST_IDX = RIGHT_HAND_SLICE.start
 
POSE_COLOR = (255, 255, 255)      # white, BGR
LEFT_HAND_COLOR = (0, 255, 0)     # green
RIGHT_HAND_COLOR = (0, 165, 255)  # orange
 
# --- Skeleton connections ----------------------------------------------------
# Full 33-point MediaPipe PoseLandmarker connection set, taken from
# mediapipe.tasks.python.vision.PoseLandmarksConnections.POSE_LANDMARKS.
_FULL_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]
 
# Keep only connections whose endpoints survive the POSE_JOINT_INDICES trim
# (e.g. legs/feet/knees drop out), remapped to local sub-array positions.
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
    Translates each hand's landmarks so that the hand's own wrist (local
    index 0) coincides with the wrist reported by the pose landmarks.
 
    hand_world_landmarks are reported in a hand-centric coordinate frame,
    independent of pose_world_landmarks' frame, so directly overlaying the
    two produces hands that float in the wrong place relative to the body.
    Since both are in meters, adding a per-frame, per-hand offset (pose
    wrist position minus hand wrist position) is enough to line them up --
    no rotation or scaling needed.
 
    arr: (T, C, 3) keypoints array.
    valid: (T, C) boolean array of which points are valid/present.
 
    Returns a new (T, C, 3) array; a hand is left untouched for frames where
    either its own wrist or the matching pose wrist is invalid (nothing
    reliable to anchor to).
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
    arr: np.ndarray, shape (T, C, 3) -- 3D keypoints (x, y, z) per frame.
    mask: optional np.ndarray, shape (T, C) -- 1.0 = valid point, 0.0 = missing.
          If None, a point is treated as missing when x == y == z == 0.
    output_path: path to write the .mp4 file to.
    fps: playback frame rate of the output video.
    frame_size: (width, height) of the output video, in pixels.
    point_radius: radius, in pixels, of each drawn keypoint.
    line_thickness: thickness, in pixels, of each drawn skeleton connection.
    margin_frac: fraction of the frame reserved as empty margin on each side.
    align_hands: if True (default), translate each hand so its own wrist
                 lines up with the corresponding wrist from the pose
                 landmarks, compensating for hand_world_landmarks' independent
                 coordinate frame. See align_hands_to_wrists().
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected arr of shape (T, C, 3), got {arr.shape}")
 
    T, C, _ = arr.shape
    W, H = frame_size
 
    expected_C = NUM_POSE_JOINTS + NUM_HAND_JOINTS * 2
    if C != expected_C:
        raise ValueError(
            f"arr has {C} joints per frame, but POSE_JOINT_INDICES implies {expected_C} "
            f"({NUM_POSE_JOINTS} pose + {NUM_HAND_JOINTS}*2 hand). Update POSE_JOINT_INDICES "
            f"at the top of this file to match extract.py."
        )
 
    if mask is None:
        # treat exact (0, 0, 0) as "missing", matching extract.py's convention
        valid = ~np.all(arr == 0.0, axis=-1)
    else:
        valid = np.asarray(mask).astype(bool)
        if valid.shape != (T, C):
            raise ValueError(f"mask shape {valid.shape} doesn't match arr's (T, C) = {(T, C)}")
 
    if align_hands:
        arr = align_hands_to_wrists(arr, valid)
 
    # Scale/center using only valid points across the whole clip, so a single
    # frame with a missing joint doesn't skew the projection for every frame.
    valid_xy = arr[..., :2][valid]
    if valid_xy.size == 0:
        raise ValueError("No valid (non-zero) keypoints found in the array.")
 
    x_min, y_min = valid_xy.min(axis=0)
    x_max, y_max = valid_xy.max(axis=0)
 
    # Keep aspect ratio: scale both axes by the same factor (the larger span),
    # centered on the data, so the skeleton isn't stretched.
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
        # flip y: image row 0 is the top, but we want "up" in the data to be up
        py = int(round(H / 2 + (y - y_center) * scale))
        return px, py
 
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(os.path.join("data", "skeleton", output_path)), fourcc, fps, (W, H))
    if not writer.isOpened():
        raise IOError(f"Could not open VideoWriter for {output_path}")
 
    # (connections, color, index-offset into the frame's C axis) for each body part
    connection_groups = [
        (POSE_CONNECTIONS, POSE_COLOR, POSE_SLICE.start),
        (HAND_CONNECTIONS, LEFT_HAND_COLOR, LEFT_HAND_SLICE.start),
        (HAND_CONNECTIONS, RIGHT_HAND_COLOR, RIGHT_HAND_SLICE.start),
    ]
 
    for t in range(T):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
 
        # Draw skeleton lines first so joint dots render on top of them.
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
    print(f"Wrote {T} frames to {output_path}")