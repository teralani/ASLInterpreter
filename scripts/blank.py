
import json
import csv
import os
import sys
from collections import defaultdict

try:
    import cv2
except ImportError:
    cv2 = None


DATA_ROOT = "data"

WLASL_JSON = os.path.join(DATA_ROOT, "WLASL_v0.3.json")
WLASL_VIDEO_DIR = os.path.join(DATA_ROOT, "videos")
ASL_CITIZEN_CSV = os.path.join(DATA_ROOT, "ASL_Citizen", "splits", "train.csv")
ASL_CITIZEN_VIDEO_DIR = os.path.join(DATA_ROOT, "ASL_Citizen", "videos")

CHECK_WLASL = True
CHECK_ASL_CITIZEN = False


def get_video_frame_count(video_path: str):
    if cv2 is None:
        return None
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n, fps


def check_wlasl(json_path: str, video_dir: str, max_examples: int = 5):
    print("\n=== WLASL ===")
    with open(json_path, "r") as f:
        data = json.load(f)

    total = 0
    has_margin = 0
    no_video_found = 0
    leading_margins = []
    trailing_margins = []
    examples = []

    for entry in data:
        gloss = entry.get("gloss", "?")
        for inst in entry.get("instances", []):
            total += 1
            video_id = inst.get("video_id")
            frame_start = inst.get("frame_start")
            frame_end = inst.get("frame_end")
            fps = inst.get("fps", None)

            if video_id is None or frame_start is None:
                continue

            video_path = None
            for ext in (".mp4", ".mkv", ".webm", ".avi"):
                candidate = os.path.join(video_dir, f"{video_id}{ext}")
                if os.path.exists(candidate):
                    video_path = candidate
                    break

            if video_path is None:
                no_video_found += 1
                continue

            result = get_video_frame_count(video_path)
            if result is None:
                no_video_found += 1
                continue
            total_frames, video_fps = result

            resolved_end = total_frames if frame_end in (None, -1) else frame_end
            leading = max(0, frame_start - 0)
            trailing = max(0, total_frames - resolved_end)

            if leading > 0 or trailing > 0:
                has_margin += 1
                if leading > 0:
                    leading_margins.append(leading)
                if trailing > 0:
                    trailing_margins.append(trailing)
                if len(examples) < max_examples:
                    examples.append({
                        "gloss": gloss,
                        "video_id": video_id,
                        "total_frames": total_frames,
                        "frame_start": frame_start,
                        "frame_end": frame_end,
                        "leading_margin_frames": leading,
                        "trailing_margin_frames": trailing,
                        "fps": video_fps or fps,
                    })

    _report("WLASL", total, has_margin, no_video_found, leading_margins, trailing_margins, examples)


def check_asl_citizen(csv_path: str, video_dir: str, max_examples: int = 5):
    print("\n=== ASL Citizen ===")
    total = 0
    has_start_end_cols = False
    rows = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_start_end_cols = any(
            col.lower() in ("frame_start", "frame_end", "start_frame", "end_frame")
            for col in fieldnames
        )
        for row in reader:
            total += 1
            rows.append(row)

    print(f"Columns found: {fieldnames}")
    if not has_start_end_cols:
        print(
            "No frame_start/frame_end-style columns found in ASL Citizen metadata.\n"
            "This strongly suggests clips are pre-trimmed to the sign itself with no\n"
            "stored margin -- i.e. Path A (margin mining) is likely NOT usable for\n"
            "ASL Citizen as released. Recommend relying on WLASL for margin frames,\n"
            "or checking ASL Citizen's project page/repo for an untrimmed variant."
        )
        durations = []
        for row in rows[: min(len(rows), 200)]:
            fname = row.get("Video file") or row.get("video") or row.get("filename")
            if not fname:
                continue
            video_path = os.path.join(video_dir, fname)
            result = get_video_frame_count(video_path)
            if result:
                n, fps = result
                if fps:
                    durations.append(n / fps)
        if durations:
            durations.sort()
            print(f"Sampled {len(durations)} clips.")
            print(f"  duration min/median/max (s): "
                  f"{durations[0]:.2f} / {durations[len(durations)//2]:.2f} / {durations[-1]:.2f}")
            print("  If median duration is well above typical single-sign length (~0.5-1.5s),")
            print("  there may still be margin *within* the clip worth checking manually.")
        else:
            print("  (Could not sample video durations -- check --video_dir path / filename column.)")
        return

    leading_margins, trailing_margins, examples = [], [], []
    has_margin = 0
    no_video_found = 0
    start_col = next(c for c in fieldnames if c.lower() in ("frame_start", "start_frame"))
    end_col = next(c for c in fieldnames if c.lower() in ("frame_end", "end_frame"))
    fname_col = next((c for c in fieldnames if "video" in c.lower() or "file" in c.lower()), None)

    for row in rows:
        fname = row.get(fname_col) if fname_col else None
        if not fname:
            continue
        video_path = os.path.join(video_dir, fname)
        result = get_video_frame_count(video_path)
        if result is None:
            no_video_found += 1
            continue
        total_frames, fps = result
        try:
            frame_start = int(row[start_col])
            frame_end = int(row[end_col])
        except (ValueError, TypeError):
            continue
        leading = max(0, frame_start)
        trailing = max(0, total_frames - frame_end)
        if leading > 0 or trailing > 0:
            has_margin += 1
            if leading > 0:
                leading_margins.append(leading)
            if trailing > 0:
                trailing_margins.append(trailing)
            if len(examples) < max_examples:
                examples.append({"file": fname, "leading": leading, "trailing": trailing,
                                  "total_frames": total_frames, "fps": fps})

    _report("ASL Citizen", total, has_margin, no_video_found, leading_margins, trailing_margins, examples)


def _report(name, total, has_margin, no_video_found, leading_margins, trailing_margins, examples):
    checked = total - no_video_found
    print(f"Total instances: {total}")
    print(f"Videos not found / unreadable: {no_video_found}")
    print(f"Instances actually checked: {checked}")
    if checked == 0:
        print("Could not verify any instances -- check --video_dir path.")
        return

    pct = 100.0 * has_margin / checked
    print(f"Instances with nonzero margin: {has_margin} ({pct:.1f}% of checked)")

    def summarize(name, vals):
        if not vals:
            print(f"  {name}: none found")
            return
        vals = sorted(vals)
        n = len(vals)
        print(f"  {name}: n={n}, min={vals[0]}, median={vals[n//2]}, max={vals[-1]}, "
              f"mean={sum(vals)/n:.1f} frames")

    summarize("Leading margin (frames)", leading_margins)
    summarize("Trailing margin (frames)", trailing_margins)

    if examples:
        print("\nExample instances with margin:")
        for ex in examples:
            print(f"  {ex}")

    if pct < 10:
        print(f"\n-> Verdict: margin frames are rare in {name}. Path A (mining blanks from")
        print("   margins) will yield a small/low-diversity blank class. Consider Path B")
        print("   (synthetic concatenation + CTC) or an external source of blank frames instead.")
    else:
        print(f"\n-> Verdict: {name} has meaningful margin frames. Path A looks viable --")
        print("   worth extracting these as a 'blank' class.")


def main():
    if cv2 is None:
        print("WARNING: opencv-python not installed. Install with `pip install opencv-python`.")
        sys.exit(1)

    print("Using paths from CONFIG at the top of this file:")
    print(f"  WLASL_JSON:             {WLASL_JSON}")
    print(f"  WLASL_VIDEO_DIR:        {WLASL_VIDEO_DIR}")
    print(f"  ASL_CITIZEN_CSV:        {ASL_CITIZEN_CSV}")
    print(f"  ASL_CITIZEN_VIDEO_DIR:  {ASL_CITIZEN_VIDEO_DIR}")

    if CHECK_WLASL:
        if not os.path.exists(WLASL_JSON):
            print(f"\nSkipping WLASL: {WLASL_JSON} not found. Edit WLASL_JSON in CONFIG.")
        elif not os.path.isdir(WLASL_VIDEO_DIR):
            print(f"\nSkipping WLASL: video dir {WLASL_VIDEO_DIR} not found. Edit WLASL_VIDEO_DIR in CONFIG.")
        else:
            check_wlasl(WLASL_JSON, WLASL_VIDEO_DIR)

    if CHECK_ASL_CITIZEN:
        if not os.path.exists(ASL_CITIZEN_CSV):
            print(f"\nSkipping ASL Citizen: {ASL_CITIZEN_CSV} not found. Edit ASL_CITIZEN_CSV in CONFIG.")
        elif not os.path.isdir(ASL_CITIZEN_VIDEO_DIR):
            print(f"\nSkipping ASL Citizen: video dir {ASL_CITIZEN_VIDEO_DIR} not found. "
                  f"Edit ASL_CITIZEN_VIDEO_DIR in CONFIG.")
        else:
            check_asl_citizen(ASL_CITIZEN_CSV, ASL_CITIZEN_VIDEO_DIR)


if __name__ == "__main__":
    main()