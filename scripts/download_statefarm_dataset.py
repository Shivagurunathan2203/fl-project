#!/usr/bin/env python3
"""
Downloads and extracts the State Farm Distracted Driver benchmark dataset from Kaggle.
No password or author approval required.
Uses credentials from dingus.txt or ~/.kaggle/kaggle.json or environment variables.
"""

import os
import sys
import json
import base64
import urllib.request
import zipfile
import shutil
from pathlib import Path


STATEFARM_DATASET_SLUG = "mdriyadulhasan/distracted-driver-dataset"
DRIVER_CSV_SLUG = "rightway11/state-farm-distracted-driver-detection"


def get_kaggle_credentials(project_root: Path) -> tuple[str, str]:
    """Resolves Kaggle username and API key from environment variables or ~/.kaggle/kaggle.json."""
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if env_user and env_key:
        return env_user.strip(), env_key.strip()

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            with open(kaggle_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                u = data.get("username")
                k = data.get("key")
                if u and k:
                    return str(u).strip(), str(k).strip()
        except Exception as e:
            print(f"[Warning] Failed parsing ~/.kaggle/kaggle.json: {e}")

    raise RuntimeError(
        "Kaggle credentials not found! Set KAGGLE_USERNAME and KAGGLE_KEY environment variables or create ~/.kaggle/kaggle.json."
    )


def setup_kaggle_dir(username: str, key: str) -> None:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    target = kaggle_dir / "kaggle.json"
    if not target.exists():
        with open(target, "w", encoding="utf-8") as f:
            json.dump({"username": username, "key": key}, f, indent=2)
        try:
            os.chmod(target, 0o600)
        except Exception:
            pass
        print(f"[Info] Configured {target}")


def download_stream(username: str, key: str, url: str, out_file: Path, desc: str) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    auth = base64.b64encode(f"{username}:{key}".encode()).decode()
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}"})

    print(f"[Info] Connecting to Kaggle: {desc}...")
    with urllib.request.urlopen(req) as resp:
        total_bytes = int(resp.headers.get("Content-Length", 0))
        total_mb = total_bytes / (1024 * 1024) if total_bytes > 0 else 0
        if total_bytes > 0:
            print(f"[Info] Content-Length: {total_bytes} bytes (~{total_mb:.1f} MB)")

        downloaded = 0
        chunk_size = 1024 * 1024 * 4  # 4 MB chunks
        with open(out_file, "wb") as f_out:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
                downloaded += len(chunk)
                mb = downloaded / (1024 * 1024)
                if total_bytes > 0:
                    pct = (downloaded / total_bytes) * 100.0
                    sys.stdout.write(f"\rDownloading: {mb:.1f}/{total_mb:.1f} MB [{pct:.1f}%]")
                else:
                    sys.stdout.write(f"\rDownloading: {mb:.1f} MB")
                sys.stdout.flush()
    print("\n[Info] Download complete!")


def main():
    root = Path(__file__).resolve().parent.parent
    sf_dir = root / "data" / "statefarm"
    sf_dir.mkdir(parents=True, exist_ok=True)
    driver_csv = root / "data" / "driver_imgs_list.csv"
    sf_driver_csv = sf_dir / "driver_imgs_list.csv"
    zip_target = sf_dir / "distracted_driver_dataset.zip"

    # Optional cleanup of unusable encrypted AUC files to save disk quota
    auc_dir = root / "data" / "auc"
    if auc_dir.exists():
        auc_zips = list(auc_dir.glob("*.zip"))
        if auc_zips:
            print(f"[Info] Cleaning {len(auc_zips)} leftover encrypted AUC archive(s) to free disk quota...")
            for z in auc_zips:
                try:
                    z.unlink()
                except Exception:
                    pass

    username, key = get_kaggle_credentials(root)
    print(f"[Info] Authenticated as Kaggle user: {username}")
    setup_kaggle_dir(username, key)

    # 1. Download driver_imgs_list.csv metadata (800 KB)
    if not driver_csv.exists() and not sf_driver_csv.exists():
        csv_url = f"https://www.kaggle.com/api/v1/datasets/download/{DRIVER_CSV_SLUG}/driver_imgs_list.csv"
        try:
            download_stream(username, key, csv_url, driver_csv, "driver_imgs_list.csv metadata")
            shutil.copyfile(driver_csv, sf_driver_csv)
        except Exception as e:
            print(f"[Warning] Could not download driver_imgs_list.csv: {e}")
    elif driver_csv.exists() and not sf_driver_csv.exists():
        shutil.copyfile(driver_csv, sf_driver_csv)

    # 2. Check if images are already extracted
    existing_imgs = len(list(sf_dir.glob("**/*.jpg"))) + len(list(sf_dir.glob("**/*.png")))
    if existing_imgs > 50:
        print(f"[Info] State Farm dataset already extracted ({existing_imgs} images in {sf_dir}). Ready to use!")
        return

    # 3. Download State Farm 10-class dataset (906 MB)
    if not zip_target.exists():
        url = f"https://www.kaggle.com/api/v1/datasets/download/{STATEFARM_DATASET_SLUG}"
        download_stream(username, key, url, zip_target, f"State Farm Distracted Driver ({STATEFARM_DATASET_SLUG})")

    # 4. Extract
    print(f"[Info] Extracting {zip_target.name} to {sf_dir}...")
    with zipfile.ZipFile(zip_target, "r") as zf:
        zf.extractall(sf_dir)

    # 5. Delete zip to save quota (20 GB limit on server)
    if zip_target.exists():
        try:
            zip_target.unlink()
            print(f"[Info] Removed archive {zip_target.name} to free disk space.")
        except Exception:
            pass

    final_imgs = len(list(sf_dir.glob("**/*.jpg"))) + len(list(sf_dir.glob("**/*.png")))

    # Pre-cache MobileNetV2 pretrained weights on the login node for offline compute nodes
    try:
        import torchvision.models as models
        print("[Info] Pre-caching MobileNetV2 feature extractor weights...")
        try:
            models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        except Exception:
            models.mobilenet_v2(pretrained=True)
        print("[Success] MobileNetV2 weights pre-cached in ~/.cache/torch/hub/checkpoints/.")
    except Exception as e:
        print(f"[Warning] Could not pre-cache MobileNetV2 weights: {e}")

    print(
        f"\n{'='*70}\n"
        f"[Success] State Farm Distracted Driver dataset is ready in {sf_dir}!\n"
        f"          Total images: {final_imgs}\n"
        f"          Classes: 10 (c0: safe driving, c1..c9: various distractions)\n"
        f"          Metadata: {sf_driver_csv if sf_driver_csv.exists() else driver_csv}\n"
        f"{'='*70}\n"
    )


if __name__ == "__main__":
    main()
