import os
import numpy as np
from pathlib import Path
from extract_keypoints import extract_video

RAW_DIR = Path("data/videos")
OUT_DIR = Path("data/processed")

def preprocess_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for video in RAW_DIR.glob("*.mp4"):
        print(f"Processing {video.name}")
        keypoints = extract_video(str(video))

        out_path = OUT_DIR / f"{video.stem}.npy"
        np.save(out_path, keypoints)

if __name__ == "__main__":
    preprocess_all()

    x = np.load("data/processed/example.npy")
    print(x.shape)
