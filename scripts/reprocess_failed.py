from pathlib import Path
import numpy as np
import sys
import logging

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from old.extract_keypoints import extract_video

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLD_DIR = Path("data/old_processed")
VIDEO_DIR = Path("data/videos")
OUT_DIR = Path("data/processed")
FAILED_LOG = Path("data/failed_videos_reprocess.txt")


def find_all_zero_files(folder: Path):
    for f in folder.glob("*.npy"):
        try:
            a = np.load(f)
        except Exception:
            yield f, "error_loading"
            continue
        if a.size == 0 or np.count_nonzero(a) == 0:
            yield f, "all_zero"


def reprocess():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = []

    for f, reason in find_all_zero_files(OLD_DIR):
        stem = f.stem
        video_path = VIDEO_DIR / (stem + ".mp4")
        if not video_path.exists():
            logger.warning("Source video not found for %s", stem)
            failed.append(stem)
            continue

        logger.info("Reprocessing %s (reason=%s)", stem, reason)
        kp_res = extract_video(str(video_path))
        if isinstance(kp_res, (tuple, list)):
            kp_coords, kp_mask = kp_res
        else:
            kp_coords = kp_res
            kp_mask = None

        if kp_coords.size == 0 or np.count_nonzero(kp_coords) == 0:
            logger.warning("Re-extraction failed for %s", stem)
            failed.append(stem)
            continue

        out_path = OUT_DIR / (stem + ".npy")
        np.save(out_path, kp_coords)
        if kp_mask is not None:
            np.save(OUT_DIR / (stem + "_mask.npy"), kp_mask)
        logger.info("Saved reprocessed keypoints for %s", stem)

    if failed:
        with FAILED_LOG.open("w") as f:
            for name in failed:
                f.write(name + "\n")
        logger.info("Logged %d failed reprocesses to %s", len(failed), FAILED_LOG)


if __name__ == "__main__":
    reprocess()
