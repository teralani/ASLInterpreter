import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np

try:
    from extract import extract_video as run_pose_extraction
    _extract_import_error = None
except Exception as e:
        run_pose_extraction = None
        _extract_import_error = e

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def is_youtube_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return "youtube.com" in host or "youtu.be" in host


def download_direct(url: str, dest: Path, timeout: int = 30) -> tuple[bool, str]:
    try:
        with requests.get(
            url, stream=True, timeout=timeout, headers={"User-Agent": USER_AGENT}
        ) as r:
            if r.status_code != 200:
                return False, f"http_{r.status_code}"

            tmp_dest = dest.with_suffix(dest.suffix + ".part")
            total_bytes = 0
            with open(tmp_dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
                        total_bytes += len(chunk)

            if total_bytes < 10_000:
                tmp_dest.unlink(missing_ok=True)
                return False, f"too_small_{total_bytes}b"

            tmp_dest.rename(dest)
            return True, "ok"

    except requests.exceptions.Timeout:
        return False, "timeout"
    except requests.exceptions.RequestException as e:
        return False, f"request_error_{type(e).__name__}"


def download_youtube(url: str, dest: Path, timeout: int = 60) -> tuple[bool, str]:
    if shutil.which("yt-dlp") is None:
        return False, "yt-dlp_not_installed"

    tmp_template = str(dest.with_suffix("")) + ".%(ext)s"
    try:
        result = subprocess.run(
            [
                "yt-dlp", "-f", "mp4/best",
                "-o", tmp_template,
                "--no-playlist",
                url,
            ],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            reason = result.stderr.strip().splitlines()[-1] if result.stderr else "unknown"
            return False, f"ytdlp_error: {reason[:120]}"

        if not dest.exists():
            candidates = list(dest.parent.glob(dest.stem + ".*"))
            if candidates:
                candidates[0].rename(dest)
            else:
                return False, "ytdlp_no_output_file"

        return True, "ok"

    except subprocess.TimeoutExpired:
        return False, "ytdlp_timeout"


def extract_and_save(video_path: Path, video_id: str, instance: dict, processed_dir: Path) -> tuple[bool, str]:
    if run_pose_extraction is None:
        return False, f"extract_import_failed_{type(_extract_import_error).__name__}: {_extract_import_error}"

    try:
        arr, mask_arr = run_pose_extraction(str(video_path))
    except Exception as e:
        return False, f"extract_error_{type(e).__name__}: {e}"

    if arr.size == 0 or np.count_nonzero(arr) == 0:
        return False, "no_keypoints_detected"

    np.save(processed_dir / f"{video_id}.npy", arr)
    np.save(processed_dir / f"{video_id}_mask.npy", mask_arr)
    return True, "ok"


def process_one(video_id: str, instance: dict, video_dir: Path, processed_dir: Path) -> dict:
    row = {"video_id": video_id, "gloss": instance.get("_gloss", ""), "url": instance.get("url", "")}

    video_path = video_dir / f"{video_id}.mp4"
    npy_path = processed_dir / f"{video_id}.npy"

    if npy_path.exists():
        row["download_status"] = "already_processed"
        row["extract_status"] = "already_processed"
        return row

    url = instance.get("url", "")
    if not url:
        row["download_status"] = "no_url_in_wsasl_json"
        row["extract_status"] = "skipped"
        return row

    if not video_path.exists():
        if is_youtube_url(url):
            ok, status = download_youtube(url, video_path)
        else:
            ok, status = download_direct(url, video_path)
        row["download_status"] = status
        if not ok:
            row["extract_status"] = "skipped"
            return row
    else:
        row["download_status"] = "already_downloaded"

    ok, status = extract_and_save(video_path, video_id, instance, processed_dir)
    row["extract_status"] = status
    return row


def main():
    video_dir = Path("data/videos_2")
    processed_dir = Path("data/processed")
    video_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    with open("data/failed_videos.txt", "r") as f:
        failed_ids = [
            line.strip().removesuffix(".mp4")
            for line in f
            if line.strip()
        ]
    print(f"Loaded {len(failed_ids)} failed video_ids from {"data/failed_videos.txt"}")

    with open("data/WLASL_v0.3.json", "r") as f:
        wsasl = json.load(f)

    id_to_instance = {}
    for entry in wsasl:
        for inst in entry["instances"]:
            inst_copy = dict(inst)
            inst_copy["_gloss"] = entry["gloss"]
            id_to_instance[inst["video_id"]] = inst_copy

    missing_from_json = [vid for vid in failed_ids if vid not in id_to_instance]
    if missing_from_json:
        print(f"Warning: {len(missing_from_json)} video_ids from the failed list aren't in the WLASL json at all (skipping these)")

    targets = [(vid, id_to_instance[vid]) for vid in failed_ids if vid in id_to_instance]
    print(f"Attempting {len(targets)} downloads with {8} workers...")

    if run_pose_extraction is None:
        print(f"could not import extract_video from extract.py ({_extract_import_error}).")

    if shutil.which("yt-dlp") is None:
        print("yt-dlp is not installed")

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(process_one, vid, inst, video_dir, processed_dir): vid
            for vid, inst in targets
        }
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            if i % 25 == 0 or i == len(targets):
                print(f"  {i}/{len(targets)} processed ({time.time()-t0:.0f}s elapsed)")

    with open("data/retry_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "gloss", "url", "download_status", "extract_status"])
        writer.writeheader()
        writer.writerows(rows)

    downloaded_ok = sum(1 for r in rows if r["download_status"] in ("ok", "already_downloaded", "already_processed"))
    extracted_ok = sum(1 for r in rows if r["extract_status"] in ("ok", "already_processed"))

    print(f"\n {downloaded_ok}/{len(rows)} downloaded successfully, {extracted_ok}/{len(rows)} extracted successfully.")


if __name__ == "__main__":
    main()