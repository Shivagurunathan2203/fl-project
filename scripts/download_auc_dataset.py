#!/usr/bin/env python3
"""
Downloads and extracts distracted driver datasets for SecureDrive-FL.
Supports:
  1. AUC Distracted Driver Dataset (V1/V2):
     If password is provided (via --password, data/auc_password.txt, or AUC_PASSWORD env),
     extracts the password-protected archive (auc.distracted.driver.dataset_v2.zip).
  2. Unencrypted 10-class Distracted Driver Benchmark:
     Downloads the unencrypted 10-class dataset (mdriyadulhasan/distracted-driver-dataset, 906 MB)
     plus driver_imgs_list.csv (mapping all 26 drivers) without requiring any password.
"""

import os
import sys
import json
import base64
import urllib.request
import urllib.parse
import zipfile
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import Optional, Tuple


AUC_KAGGLE_SLUG = "tejakalepalle/auc-distracted-driver-dataset-v1"
STATEFARM_KAGGLE_SLUG = "mdriyadulhasan/distracted-driver-dataset"
DRIVER_CSV_SLUG = "rightway11/state-farm-distracted-driver-detection"


def get_kaggle_credentials(project_root: Path) -> Tuple[str, str]:
    """Resolves Kaggle username and API key from env, dingus.txt, or ~/.kaggle/kaggle.json."""
    # 1. Environment variables
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if env_user and env_key:
        return env_user.strip(), env_key.strip()

    # 2. dingus.txt
    dingus_path = project_root / "dingus.txt"
    if dingus_path.exists():
        try:
            with open(dingus_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                u = data.get("username")
                k = data.get("key")
                if u and k:
                    return str(u).strip(), str(k).strip()
        except Exception as e:
            print(f"[Warning] Failed parsing dingus.txt: {e}")

    # 3. ~/.kaggle/kaggle.json
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
        "Kaggle credentials not found! Please ensure dingus.txt is in the project root, "
        "or set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
    )


def setup_kaggle_dir(username: str, key: str) -> None:
    """Configures ~/.kaggle/kaggle.json for Kaggle CLI integration."""
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


def get_auc_password(project_root: Path, cli_pwd: Optional[str]) -> Optional[str]:
    """Retrieves AUC extraction password if available."""
    if cli_pwd:
        return cli_pwd.strip()
    env_pwd = os.environ.get("AUC_PASSWORD")
    if env_pwd:
        return env_pwd.strip()
    pwd_file = project_root / "data" / "auc_password.txt"
    if pwd_file.exists():
        try:
            return pwd_file.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return None


def download_stream(username: str, key: str, url: str, out_file: Path, desc: str) -> None:
    """Downloads a file with streaming progress bar."""
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


def extract_zip(zip_path: Path, dest_dir: Path, password: Optional[str] = None) -> bool:
    """Extracts a zip archive, supporting passwords via zipfile and system unzip/7z fallback."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    pwd_bytes = password.encode("utf-8") if password else None

    # Method 1: Python zipfile
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir, pwd=pwd_bytes)
        return True
    except RuntimeError as e:
        if "encrypted" in str(e).lower() or "password" in str(e).lower() or "bad password" in str(e).lower():
            if not password:
                raise RuntimeError(
                    f"Archive '{zip_path.name}' is password encrypted. Password required."
                )
            print(f"[Warning] Python zipfile password failed ({e}), trying system unzip/7z...")
        else:
            raise

    # Method 2: System unzip tool (handles AES-256 and legacy encryption)
    if password:
        try:
            res = subprocess.run(
                ["unzip", "-P", password, "-o", str(zip_path), "-d", str(dest_dir)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if res.returncode == 0:
                return True
        except Exception:
            pass

        # Method 3: 7z tool
        try:
            res = subprocess.run(
                ["7z", "x", f"-p{password}", f"-o{dest_dir}", "-y", str(zip_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if res.returncode == 0:
                return True
        except Exception:
            pass

    return False


def handle_auc_dataset(root: Path, username: str, key: str, password: Optional[str]) -> bool:
    """Attempts downloading and extracting AUC dataset."""
    auc_dir = root / "data" / "auc"
    auc_dir.mkdir(parents=True, exist_ok=True)
    zip_target = auc_dir / "auc_dataset.zip"

    # Check for already downloaded nested zip or main zip
    nested_zips = list(auc_dir.glob("*.zip"))
    if not nested_zips and not zip_target.exists():
        url = f"https://www.kaggle.com/api/v1/datasets/download/{AUC_KAGGLE_SLUG}"
        download_stream(username, key, url, zip_target, f"AUC Distracted Driver ({AUC_KAGGLE_SLUG})")
        extract_zip(zip_target, auc_dir)
        if zip_target.exists():
            try:
                zip_target.unlink()
            except Exception:
                pass

    # Process nested zips (e.g. auc.distracted.driver.dataset_v2.zip)
    nested_zips = list(auc_dir.glob("*.zip"))
    for f in nested_zips:
        print(f"[Info] Processing nested archive '{f.name}'...")
        if not password:
            print(
                f"\n{'='*70}\n"
                f"[Notice] '{f.name}' is password-protected by the AUC dataset authors\n"
                f"(Hesham Eraqi & Yehya Abouelnaga) under an academic license.\n\n"
                f"If you have received this password, run:\n"
                f"    python scripts/download_auc_dataset.py --password <YOUR_PASSWORD>\n"
                f"or save the password in data/auc_password.txt.\n"
                f"{'='*70}\n"
            )
            return False

        print(f"[Info] Attempting extraction with provided password...")
        try:
            success = extract_zip(f, auc_dir, password=password)
            if success:
                print(f"[Success] Successfully decrypted and extracted {f.name}!")
                try:
                    f.unlink()
                except Exception:
                    pass
                return True
            else:
                print(f"[Error] Extraction of {f.name} failed with the provided password.")
                return False
        except Exception as e:
            print(f"[Error] Failed to extract {f.name}: {e}")
            return False

    # Check if images exist
    img_count = len(list(auc_dir.glob("**/*.jpg"))) + len(list(auc_dir.glob("**/*.png")))
    return img_count > 50


def handle_unencrypted_fallback(root: Path, username: str, key: str) -> None:
    """
    Downloads and extracts the unencrypted 10-class Distracted Driver benchmark dataset (906 MB)
    and driver_imgs_list.csv (800 KB) into data/statefarm/.
    """
    sf_dir = root / "data" / "statefarm"
    sf_dir.mkdir(parents=True, exist_ok=True)
    zip_target = sf_dir / "distracted_driver_dataset.zip"
    driver_csv = root / "data" / "driver_imgs_list.csv"
    sf_driver_csv = sf_dir / "driver_imgs_list.csv"

    print(
        f"\n{'='*70}\n"
        f"[Info] Downloading unencrypted 10-class Distracted Driver benchmark dataset...\n"
        f"       Dataset: {STATEFARM_KAGGLE_SLUG} (906 MB, no password required)\n"
        f"       Metadata: driver_imgs_list.csv (26 drivers, c0-c9 ground truth)\n"
        f"{'='*70}"
    )

    # 1. Download driver_imgs_list.csv
    if not driver_csv.exists() and not sf_driver_csv.exists():
        csv_url = f"https://www.kaggle.com/api/v1/datasets/download/{DRIVER_CSV_SLUG}/driver_imgs_list.csv"
        try:
            download_stream(username, key, csv_url, driver_csv, "driver_imgs_list.csv metadata")
            shutil.copyfile(driver_csv, sf_driver_csv)
        except Exception as e:
            print(f"[Warning] Could not download driver_imgs_list.csv: {e}")

    # 2. Check if already extracted
    existing_imgs = len(list(sf_dir.glob("**/*.jpg"))) + len(list(sf_dir.glob("**/*.png")))
    if existing_imgs > 50:
        print(f"[Info] Dataset already extracted in {sf_dir} ({existing_imgs} images). Ready to use!")
        return

    # 3. Download dataset zip
    if not zip_target.exists():
        url = f"https://www.kaggle.com/api/v1/datasets/download/{STATEFARM_KAGGLE_SLUG}"
        download_stream(username, key, url, zip_target, f"10-class Distracted Driver ({STATEFARM_KAGGLE_SLUG})")

    # 4. Extract
    print(f"[Info] Extracting {zip_target.name} to {sf_dir}...")
    extract_zip(zip_target, sf_dir)

    # 5. Remove zip archive to preserve disk quota
    if zip_target.exists():
        try:
            zip_target.unlink()
            print(f"[Info] Removed temporary archive {zip_target.name} to save disk space.")
        except Exception:
            pass

    # Verify
    final_imgs = len(list(sf_dir.glob("**/*.jpg"))) + len(list(sf_dir.glob("**/*.png")))
    print(
        f"\n{'='*70}\n"
        f"[Success] Real Distracted Driver dataset is ready in {sf_dir}!\n"
        f"          Total images: {final_imgs}\n"
        f"          Classes: 10 (c0: safe driving, c1..c9: various distractions)\n"
        f"          Metadata: {driver_csv if driver_csv.exists() else sf_driver_csv}\n"
        f"{'='*70}\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Download & extract driver datasets for SecureDrive-FL.")
    parser.add_argument("--password", type=str, default=None, help="Password for encrypted AUC archive")
    parser.add_argument("--fallback", action="store_true", help="Download unencrypted 10-class benchmark dataset directly")
    parser.add_argument("--statefarm", action="store_true", help="Alias for --fallback")
    parser.add_argument("--clean", action="store_true", help="Delete existing zip archives before starting")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"

    if args.clean:
        for z in data_dir.glob("**/*.zip"):
            try:
                z.unlink()
                print(f"[Info] Cleaned up {z}")
            except Exception:
                pass

    username, key = get_kaggle_credentials(root)
    print(f"[Info] Authenticated as Kaggle user: {username}")
    setup_kaggle_dir(username, key)

    password = get_auc_password(root, args.password)

    if args.fallback or args.statefarm:
        handle_unencrypted_fallback(root, username, key)
        return

    # Check if real data already exists and ready
    from securedrive_fl.data import check_real_data_exists
    exists, _ = check_real_data_exists(str(data_dir))
    if exists:
        img_count = len(list(data_dir.glob("**/*.jpg"))) + len(list(data_dir.glob("**/*.png")))
        print(f"[Info] Real driver dataset already found ({img_count} images in {data_dir}). Ready to use!")
        return

    # Try AUC first
    auc_ok = handle_auc_dataset(root, username, key, password)
    if auc_ok:
        print("[Success] AUC Distracted Driver dataset ready in data/auc!")
        return

    # If AUC cannot be extracted (due to password), automatically fallback or prompt
    print("\n[Action] Falling back to unencrypted 10-class benchmark dataset...")
    handle_unencrypted_fallback(root, username, key)


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


if __name__ == "__main__":
    main()
