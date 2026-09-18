import csv
import re
import sys
from pathlib import Path
from collections import defaultdict


def normalize_gloss(g: str) -> str:
    g = g.upper()
    g = re.sub(r"[^A-Z0-9]", "", g)
    g = re.sub(r"\d+$", "", g)
    return g


def load_aslc_split(csv_path: Path, split_name: str):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gloss = row["Gloss"]
            rows.append((normalize_gloss(gloss), gloss, row["Video file"]))
    return rows


def main():

    sys.path.append(str(Path(__file__).parent.parent.resolve()))
    from training.dataset import ASLDataset

    wsasl_json = "data/WLASL_v0.3.json"
    processed_dir = "data/processed"
    aslc_dir = "data/ASL_Citizen"
    out_dir = "data/aslc_matches.csv"


    train_set = ASLDataset(processed_dir=processed_dir, wsasl_json=wsasl_json, split="train")
    wlasl_glosses = sorted(train_set.gloss2id.keys())

    aslc_root = Path(aslc_dir)
    aslc_rows = []
    for split_name, fname in [("train", "train.csv"), ("val", "val.csv"), ("test", "test.csv")]:
        path = aslc_root / "Splits" / fname
        rows = load_aslc_split(path, split_name)
        for norm_gloss, raw_gloss, video_file in rows:
            aslc_rows.append((norm_gloss, raw_gloss, video_file, split_name))

    aslc_by_norm = defaultdict(list)
    for norm_gloss, raw_gloss, video_file, split_name in aslc_rows:
        aslc_by_norm[norm_gloss].append((raw_gloss, video_file, split_name))

    matched_rows = []
    unmatched = []

    for wlasl_gloss in wlasl_glosses:
        norm = normalize_gloss(wlasl_gloss)
        candidates = aslc_by_norm.get(norm, [])
        if not candidates:
            unmatched.append(wlasl_gloss)
            continue
        for raw_gloss, video_file, split_name in candidates:
            matched_rows.append({
                "wlasl_gloss": wlasl_gloss,
                "aslc_split": split_name,
                "aslc_video_file": video_file,
                "aslc_raw_gloss": raw_gloss,
            })

    with open(out_dir, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wlasl_gloss", "aslc_split", "aslc_video_file", "aslc_raw_gloss"])
        writer.writeheader()
        writer.writerows(matched_rows)

    matched_glosses = {r["wlasl_gloss"] for r in matched_rows}
    per_gloss_counts = defaultdict(int)
    for r in matched_rows:
        per_gloss_counts[r["wlasl_gloss"]] += 1

    print(f"\n{len(matched_glosses)}/{len(wlasl_glosses)} WLASL-100 glosses in ASL Citizen")
    print(f"{len(matched_rows)} total matched videos written: {out_dir}")

    if unmatched:
        print(f"\n{len(unmatched)} glosses withou match")
        for g in unmatched:
            print(f"  {g}")

    if matched_glosses:
        counts = sorted(per_gloss_counts.values())
        print(f"\nPer-gloss counts w/ matched glosses: min={counts[0]}, median={counts[len(counts)//2]}, max={counts[-1]}")


if __name__ == "__main__":
    main()

