import os
import numpy as np
from pathlib import Path
from old.extract_keypoints import extract_video, POSE_JOINTS, HAND_JOINTS
import logging

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/videos")
OUT_DIR = Path("data/processed")
FAILED_LOG = Path("data/failed_videos.txt")


def preprocess_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = []

    for video in RAW_DIR.glob("*.mp4"):
        print(f"Processing {video.name}")
        kp_res = extract_video(str(video))

        # extract_video now returns (coords, mask)
        if isinstance(kp_res, tuple) or isinstance(kp_res, list):
            coords, mask = kp_res
        else:
            coords = kp_res
            mask = None

        out_path = OUT_DIR / f"{video.stem}.npy"

        # skip saving empty or all-zero extractions and log failures
        if coords.size == 0 or np.count_nonzero(coords) == 0:
            logger.warning("Extraction produced no keypoints for %s", video.name)
            failed.append(video.stem + ".mp4")
            continue

        np.save(out_path, coords)
        # save mask if available
        if mask is not None:
            mask_path = OUT_DIR / f"{video.stem}_mask.npy"
            np.save(mask_path, mask)

    if failed:
        FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)
        with FAILED_LOG.open("w") as f:
            for name in failed:
                f.write(name + "\n")
        print(f"Logged {len(failed)} failed videos to {FAILED_LOG}")


if __name__ == "__main__":
    preprocess_all()

    x = np.load("data/processed/example.npy")
    print("shape: ", x.shape, " ; all_zeroes? ", np.allclose(x, 0))
