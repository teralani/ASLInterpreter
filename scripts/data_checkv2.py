import os
from pathlib import Path
import json



processed_dir="data/processed"
wsasl_json="data/WLASL_v0.3.json"

with open(wsasl_json, "r") as f:
    wsasl = json.load(f)

processed_stems = {
    f.stem for f in Path(processed_dir).glob("*.npy") if not f.stem.endswith("_mask")
}

split_counts = {}
for entry in wsasl:
    g = entry["gloss"]
    counts = split_counts.setdefault(g, {"train": 0, "val": 0, "test": 0})
    for inst in entry["instances"]:
        if inst["video_id"] in processed_stems and inst["split"] in counts:
            counts[inst["split"]] += 1

eligible = [
    g for g, c in split_counts.items()
    if c["train"] > 0 and c["val"] > 0 and c["test"] > 0
]
eligible.sort(key=lambda g: sum(split_counts[g].values()), reverse=True)
top_glosses = set(eligible[:100])

raw_video_dir = Path("data/videos")

missing_instances = [
    (entry["gloss"], inst["video_id"])
    for entry in wsasl if entry["gloss"] in top_glosses
    for inst in entry["instances"]
    if inst["video_id"] not in processed_stems
]

print(f"{len(missing_instances)} missing instances to categorize")

no_raw_video = 0
raw_video_present = 0

for gloss, vid in missing_instances:
    if (raw_video_dir / f"{vid}.mp4").exists():
        raw_video_present += 1
    else:
        no_raw_video += 1

print(f"Missing raw video file entirely: {no_raw_video}")
print(f"Raw video present but extraction failed/produced no .npy: {raw_video_present}")