from pathlib import Path
import cv2
from tqdm import tqdm

from extract import REPROCESS_LOG, RAW_DIR, OUT_DIR
from utils.skeleton import WINDOW_DURATION_MS

def find_lengths(video_paths:str):
    durations_sec = {}

    def get_duration_opencv(video_path:Path):
        video = cv2.VideoCapture(str(video_path))
        
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        video.release()
        
        if fps > 0:
            return frame_count / fps
        return 0

    video = 0
    for video_path in tqdm(Path(video_paths).glob("*.mp4"), desc=f"Video #{video+1}", leave=False):
        duration = round(get_duration_opencv(video_path))

        if duration > 7:
            print(video_path.name)

        if duration in durations_sec:
            durations_sec[duration] += 1
        else:
            durations_sec[duration] = 1

        video += 1

    return durations_sec


def check_length():
    total = 0
    too_long = 0
    REPROCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with REPROCESS_LOG.open("w") as f:
        for arrays in OUT_DIR.glob("*.npy"):
            total += 1
            if "mask" in arrays.stem:
                continue
            video = cv2.VideoCapture(f"{RAW_DIR}/{arrays.stem}.mp4")

            if video.get(cv2.CAP_PROP_FRAME_COUNT) // video.get(cv2.CAP_PROP_FPS) > (WINDOW_DURATION_MS) / 1000 : 
                too_long += 1
                f.write(arrays.stem + "\n")
        
        video.release()
    print(f"{too_long} of {total} are longer than the {WINDOW_DURATION_MS / 1000} seconds")


if __name__ == "__main__":
    durations = find_lengths("data/videos")

    print(durations)
        