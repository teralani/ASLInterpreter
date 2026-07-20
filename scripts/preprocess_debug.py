import numpy as np
from pathlib import Path
import logging
import cv2
import sys

# ensure repository root is on sys.path so imports of local modules succeed
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from old import extract_keypoints as ek

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/videos")
OUT_DIR = Path("data/test_processed")


def get_hand_connections(n):
    # Standard hand topology template (thumb, index, middle, ring, pinky)
    template = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    return [(a, b) for a, b in template if a < n and b < n]


def draw_landmarks_on_frame(frame, pose_res, hand_res):
    h, w = frame.shape[:2]
    out = frame.copy()

    # draw pose landmarks
    if getattr(pose_res, "pose_landmarks", None):
        try:
            for lm in pose_res.pose_landmarks[0][: ek.POSE_JOINTS]:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(out, (x, y), 3, (0, 255, 0), -1)
        except Exception:
            pass

    # draw hands using hand joint count from extract_keypoints
    left, right = ek.normalize_hands(hand_res)
    hand_conns = get_hand_connections(ek.HAND_JOINTS)

    if left:
        for i, lm in enumerate(left):
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(out, (x, y), 3, (255, 0, 0), -1)
        for a, b in hand_conns:
            if a < len(left) and b < len(left):
                xa = int(left[a].x * w)
                ya = int(left[a].y * h)
                xb = int(left[b].x * w)
                yb = int(left[b].y * h)
                cv2.line(out, (xa, ya), (xb, yb), (255, 0, 0), 1)

    if right:
        for i, lm in enumerate(right):
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(out, (x, y), 3, (0, 0, 255), -1)
        for a, b in hand_conns:
            if a < len(right) and b < len(right):
                xa = int(right[a].x * w)
                ya = int(right[a].y * h)
                xb = int(right[b].x * w)
                yb = int(right[b].y * h)
                cv2.line(out, (xa, ya), (xb, yb), (0, 0, 255), 1)

    return out


def test_preprocess():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = []

    for video in RAW_DIR.glob("*.mp4"):
        print(f"Processing {video.name}")

        # run extraction to get .npy keypoints and mask
        kp_res = ek.extract_video(str(video))
        if isinstance(kp_res, (tuple, list)):
            keypoints, mask = kp_res
        else:
            keypoints = kp_res
            mask = None

        out_npy = OUT_DIR / f"{video.stem}.npy"
        out_mask = OUT_DIR / f"{video.stem}_mask.npy"
        out_vid = OUT_DIR / f"{video.stem}_annotated.mp4"

        # skip saving empty or all-zero extractions and log failures
        if keypoints.size == 0 or np.count_nonzero(keypoints) == 0:
            logger.warning("Extraction produced no keypoints for %s", video.name)
            failed.append(video.stem + ".mp4")
            continue

        np.save(out_npy, keypoints)
        if mask is not None:
            np.save(out_mask, mask)

        # now open the video and write an annotated version
        cap = cv2.VideoCapture(str(video))
        frames = []
        # capture fps and framesize while capture is open
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        # release early after reading
        cap.release()

        if len(frames) == 0:
            logger.warning("No frames to annotate for %s", video.name)
            continue

        # determine sampled indices (same as extraction)
        indices = ek.sample_frames(len(frames), ek.TARGET_FRAMES)

        # init models for detection
        try:
            ek._init_models()
        except Exception:
            logger.exception("Failed to init MediaPipe models for %s", video.name)

        # prepare video writer
        h, w = frames[0].shape[:2]
        if fps <= 0:
            # fallback to a reasonable fps if capture didn't provide one
            fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_vid), fourcc, float(fps), (w, h))

        # if mp4v writer couldn't be opened (platform/codec issue), fall back to XVID/.avi
        if not writer.isOpened():
            logger.warning("mp4 writer not available on this platform, falling back to .avi XVID for %s", video.name)
            out_vid_avi = out_vid.with_suffix('.avi')
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(str(out_vid_avi), fourcc, float(fps), (w, h))
            if not writer.isOpened():
                logger.error("Failed to open any VideoWriter for %s; skipping annotated save", video.name)
                continue

        # re-run detection only on sampled frames and draw landmarks
        for i, frame in enumerate(frames):
            if i in indices:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb = np.ascontiguousarray(frame_rgb)
                    mp_image = ek.mp.Image(image_format=ek.mp.ImageFormat.SRGB, data=frame_rgb)
                    pose_res = ek.pose.detect(mp_image)
                    hand_res = ek.hand.detect(mp_image)
                    annotated = draw_landmarks_on_frame(frame, pose_res, hand_res)
                except Exception:
                    annotated = frame
            else:
                annotated = frame

            writer.write(annotated)

        writer.release()

        # only process a single file in debug; remove break to run over all
        break


if __name__ == "__main__":
    test_preprocess()

    