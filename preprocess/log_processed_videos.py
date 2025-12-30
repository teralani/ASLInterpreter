from pathlib import Path
import json

RAW_DIR = Path("data/videos")
PROCESSED_DIR = Path("data/processed")
LOG_FILE = Path("data/failed_videos.txt")
JSON_FILE = Path("data/WLASL_v0.3.json")

def log_failed_files():
    if not RAW_DIR.exists():
        print(f"{RAW_DIR} does not exist.")
        return
    if not PROCESSED_DIR.exists():
        print(f"{PROCESSED_DIR} does not exist.")
        return

    raw_files = {f.stem for f in RAW_DIR.glob("*.mp4")}

    processed_files = {f.stem for f in PROCESSED_DIR.glob("*.npy")}

    print(f"{len(raw_files)} raw files and {len(processed_files)} processed files")

    failed_files = sorted(raw_files - processed_files)

    if len(failed_files) == 0:
        print("No failed videos found. All videos were processed.")
        return
    
    with LOG_FILE.open("w") as f:
        for name in failed_files:
            f.write(f"{name}.mp4\n")
    
    print(f"Logged {len(failed_files)} failed videos to {LOG_FILE}")

def log_missing_vids():
    if not JSON_FILE.exists():
        print(f"{JSON_FILE} does not exist")
        return
    if not PROCESSED_DIR.exists():
        print(f"{PROCESSED_DIR} does not exist.")
        return

    try:
        with open(JSON_FILE, "r") as file, LOG_FILE.open("w") as f:
            data = json.load(file)
            d = {f"{vid["video_id"]}" for word in data for vid in word["instances"] }
            processed_files = {f.stem for f in PROCESSED_DIR.glob("*.npy")}
            failed_files = sorted(d - processed_files)
            print(len(failed_files))

            for name in failed_files:
                f.write(f"{name}\n")

            print(f"Logged {len(failed_files)} failed videos to {LOG_FILE}")
            print(f"There were a total of {len(d)} files in {JSON_FILE}")
            print(f"There are {len(processed_files)} files currently processed")

    except FileNotFoundError:
        print("Error: the file was not found")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from the file: {e}")



if __name__ == '__main__':
    # log_failed_files()
    log_missing_vids()
