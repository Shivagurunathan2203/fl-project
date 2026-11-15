#!/usr/bin/env python3
"""
Calibrates TenSEAL CKKS cryptographic operation timings and ciphertext sizes.
Writes results to results_ext/calibration.json.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import tenseal as ts
    from securedrive_fl.he import TenSealCKKS
    HAS_TENSEAL = True
except ImportError:
    HAS_TENSEAL = False


def calibrate(n_repeats: int = 200, out_path: str = "results_ext/calibration.json") -> dict:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if Path(out_path).exists():
        try:
            with open(out_path, "r") as f:
                calib = json.load(f)
            print(f"[Info] Calibration already exists at {out_path}. Skipping.")
            return calib
        except Exception:
            pass

    if not HAS_TENSEAL:
        sys.exit("ERROR: TenSEAL is not installed. Calibration requires TenSEAL to measure cryptographic timings.")

    print(f"[Info] Running TenSEAL CKKS calibration ({n_repeats} chunks)...")
    backend = TenSealCKKS()
    chunk_size = 4096

    enc_times = []
    dec_times = []
    add_times = []
    sizes = []

    rng = np.random.default_rng(42)

    for i in range(n_repeats):
        v1 = rng.normal(0, 1.0, size=chunk_size)
        v2 = rng.normal(0, 1.0, size=chunk_size)

        # 1. Encryption
        t0 = time.perf_counter()
        c1 = backend.encrypt(v1)
        t_enc = time.perf_counter() - t0
        enc_times.append(t_enc)

        c2 = backend.encrypt(v2)

        # Size of serialized ciphertext
        serialized = c1[0][0].serialize()
        sizes.append(len(serialized))

        # 2. Addition
        t0 = time.perf_counter()
        c_sum = backend.add(c1, c2)
        t_add = time.perf_counter() - t0
        add_times.append(t_add)

        # 3. Decryption
        t0 = time.perf_counter()
        _ = backend.decrypt(c_sum)
        t_dec = time.perf_counter() - t0
        dec_times.append(t_dec)

    calib = {
        "t_enc": float(np.median(enc_times)),
        "t_dec": float(np.median(dec_times)),
        "t_add": float(np.median(add_times)),
        "bytes_per_chunk": int(np.median(sizes)),
        "plain_bytes_per_coord": 8,
        "calibration_source": "measured",
        "n_repeats": n_repeats,
        "poly_modulus_degree": 8192,
        "enc_iqr": float(np.percentile(enc_times, 75) - np.percentile(enc_times, 25)),
        "dec_iqr": float(np.percentile(dec_times, 75) - np.percentile(dec_times, 25)),
    }

    with open(out_path, "w") as f:
        json.dump(calib, f, indent=2)

    print(f"[Success] Calibrated timings saved to {out_path}:")
    print(f"  t_enc: {calib['t_enc']*1000:.2f} ms | t_dec: {calib['t_dec']*1000:.2f} ms | t_add: {calib['t_add']*1000:.2f} ms")
    print(f"  Ciphertext size: {calib['bytes_per_chunk']} bytes/chunk")
    return calib


if __name__ == "__main__":
    calibrate()
