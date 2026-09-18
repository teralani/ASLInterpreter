
import json
from pathlib import Path


def build_gloss_vocab(processed_dir, wsasl_json_path, verbose=True):
    processed_dir = Path(processed_dir)

    with open(wsasl_json_path, "r") as f:
        wsasl = json.load(f)

    processed_stems = {
        f.stem for f in processed_dir.glob("*.npy") if not f.stem.endswith("_mask")
    }

    if verbose:
        print(f"processed_dir resolved to: {processed_dir.resolve()}")
        print(f"  {len(processed_stems)} processed sample stems found")
        print(f"wsasl_json resolved to: {Path(wsasl_json_path).resolve()}")
        print(f"  {len(wsasl)} gloss entries in json")

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

    if verbose:
        any_match = sum(1 for c in split_counts.values() if sum(c.values()) > 0)
        print(f"  {any_match}/{len(split_counts)} glosses have >=1 processed instance in ANY split")
        print(f"  {len(eligible)}/{len(split_counts)} glosses are eligible (>=1 in EVERY split)")
        if len(eligible) < 10:
            print(f"  eligible glosses: {eligible}")
        if any_match < len(split_counts) * 0.5:
            print(
                "most glosses have zero matching processed instances (wrong path, wrong json, or run from an unexpected working directory)."
            )

    top_glosses = set(eligible[:])

    gloss2id = {g: idx for idx, g in enumerate(sorted(top_glosses))}
    idx_to_gloss = {idx: g for g, idx in gloss2id.items()}
    return gloss2id, idx_to_gloss


def save_gloss_vocab(processed_dir, wsasl_json_path, out_path="gloss_vocab.json"):
    gloss2id, idx_to_gloss = build_gloss_vocab(processed_dir, wsasl_json_path)
    with open(out_path, "w") as f:
        json.dump({"gloss2id": gloss2id, "idx_to_gloss": idx_to_gloss}, f, indent=2)
    print(f"Saved {len(gloss2id)} classes to {out_path}")
    return gloss2id, idx_to_gloss


def load_gloss_vocab(path="gloss_vocab.json"):
    with open(path, "r") as f:
        data = json.load(f)
    idx_to_gloss = {int(k): v for k, v in data["idx_to_gloss"].items()}
    return data["gloss2id"], idx_to_gloss



if __name__ == "__main__":

    PROCESSED_DIR = "data/processed"
    JSON = "data/combined_dict.json"
    OUT = "live/gloss_vocab.json"

    save_gloss_vocab(PROCESSED_DIR, JSON, OUT)