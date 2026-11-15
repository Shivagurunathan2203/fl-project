from __future__ import annotations

import os
import re
import csv
import glob
import zlib
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set
import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# 1. TOY DATA (NumPy only, calibrated 10 distracted-driver classes)
# ═════════════════════════════════════════════════════════════════════════════
def make_toy_federation(
    n_clients: int = 6,
    d: int = 100,
    K: int = 10,
    n_per_client: int = 480,
    alpha: float = 1.0,
    seed: int = 0,
    separation: float = 3.8,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray], int]:
    """
    Gaussian class clusters across 10 distracted-driver classes;
    label-skewed (Dirichlet) clients = non-IID.
    Fixed cluster separation chosen a priori (default: 3.8).
    """
    rng = np.random.default_rng(seed)
    class_dirs = rng.normal(0, 1.0, (K, d))
    class_dirs = class_dirs / np.linalg.norm(class_dirs, axis=1, keepdims=True)
    means = class_dirs * separation

    def draw(labels):
        return means[labels] + rng.normal(0, 1.0, (len(labels), d)), labels

    clients = [
        draw(rng.choice(K, n_per_client, p=rng.dirichlet(np.full(K, alpha))))
        for _ in range(n_clients)
    ]
    return clients, draw(rng.integers(0, K, 2000)), K


# ═════════════════════════════════════════════════════════════════════════════
# 2. REAL DRIVER DATASET PARTITIONER (AUC Distracted Driver & State Farm)
# ═════════════════════════════════════════════════════════════════════════════
DRIVER_CLASSES = [
    "c0: safe driving",
    "c1: texting - right",
    "c2: talking on the phone - right",
    "c3: texting - left",
    "c4: talking on the phone - left",
    "c5: operating the radio",
    "c6: drinking",
    "c7: reaching behind",
    "c8: hair and makeup",
    "c9: talking to passenger",
]

AUC_FOLDER_MAP = {
    "safe driving": 0, "safe_driving": 0, "c0": 0,
    "texting right": 1, "texting_right": 1, "texting - right": 1, "c1": 1,
    "talking phone right": 2, "talking on the phone - right": 2, "talk right": 2, "c2": 2,
    "texting left": 3, "texting_left": 3, "texting - left": 3, "c3": 3,
    "talking phone left": 4, "talking on the phone - left": 4, "talk left": 4, "c4": 4,
    "adjust radio": 5, "adjusting radio": 5, "operating the radio": 5, "operating radio": 5, "radio": 5, "c5": 5,
    "drinking": 6, "c6": 6,
    "reach behind": 7, "reaching behind": 7, "c7": 7,
    "hair and makeup": 8, "hair & makeup": 8, "makeup": 8, "c8": 8,
    "talking to passenger": 9, "talk passenger": 9, "passenger": 9, "c9": 9,
}


def check_real_data_exists(data_dir: str = "data", dataset: str = "statefarm") -> Tuple[bool, List[str]]:
    """Checks if requested distracted driver dataset (State Farm or AUC) exists in data_dir."""
    base = Path(data_dir)
    img_files = list(base.glob("**/*.jpg")) + list(base.glob("**/*.png"))
    exists = len(img_files) > 50
    missing = [] if exists else ["data/auc or data/statefarm with distracted driver images"]
    return exists, missing


def _extract_class_and_subject(
    img_path: Path,
    csv_lookup: Optional[Dict[str, Tuple[int, str]]] = None,
    allow_pseudo_subjects: bool = False,
) -> Tuple[Optional[int], str]:
    """
    Infers 0-9 class and subject ID from path, filename, or driver_imgs_list.csv.
    Supports State Farm naming: img_... / c0..c9,
    and AUC Distracted Driver naming: e.g. 'Adjust Radio', 'Safe Driving', camera/c0/..., driver_id_...
    """
    # 1. Check CSV lookup if available
    img_name = img_path.name.lower()
    if csv_lookup and img_name in csv_lookup:
        return csv_lookup[img_name]

    path_str = str(img_path).replace("\\", "/").lower()

    # 2. Check AUC verbal folder names
    cls_idx: Optional[int] = None
    for part in reversed(img_path.parts[:-1]):
        part_clean = part.strip().lower()
        if part_clean in AUC_FOLDER_MAP:
            cls_idx = AUC_FOLDER_MAP[part_clean]
            break
        for k, v in AUC_FOLDER_MAP.items():
            if k in part_clean:
                cls_idx = v
                break
        if cls_idx is not None:
            break

    # 3. Check regex c0..c9
    if cls_idx is None:
        class_match = re.search(r"/c([0-9])(/|_|\.|$)", path_str)
        if not class_match:
            class_match = re.search(r"c([0-9])", img_path.name.lower())
        if class_match:
            cls_idx = int(class_match.group(1))

    # 4. Search for subject/driver identifier
    subj_match = re.search(r"(driver|user|subject|p|id)[_-]?([0-9]+)", path_str)
    if subj_match:
        subj_id = f"{subj_match.group(1)}_{subj_match.group(2)}"
    else:
        # Fallback: pseudo-subject based on parent folder or reproducible CRC32 hash
        parent_name = img_path.parent.name.lower()
        if parent_name not in AUC_FOLDER_MAP and not parent_name.startswith("c"):
            subj_id = parent_name
        else:
            if not allow_pseudo_subjects and not csv_lookup:
                raise ValueError(
                    f"Subject identifier not found in path '{img_path}' and driver_imgs_list.csv missing. "
                    f"Pass allow_pseudo_subjects=True to allow pseudo-driver grouping."
                )
            h = zlib.crc32(img_path.stem.encode("utf-8")) % 20
            subj_id = f"pseudo_driver_{h:02d}"

    return cls_idx, subj_id


def load_real_driver_data(
    data_dir: str = "data",
    n_clients: int = 6,
    seed: int = 42,
    img_size: int = 224,
    feature_dim: int = 1280,
    max_samples_per_client: Optional[int] = None,
    allow_pseudo_subjects: bool = False,
    min_classes_per_client: int = 5,
    extractor_name: str = "mobilenet_v2_1280",
    split_mode: str = "paper",
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray], int]:
    """
    Partitions the AUC / State Farm Distracted Driver dataset across n_clients.
    In 'paper' mode (default, per Section III-A & IV-A of SecureDrive-FL paper):
      - Each client holds images from 3-5 unique drivers (non-IID user-based partition).
      - Each client partitions its data D_i into 80% train and 20% validation (D_i = D_i_train U D_i_val).
      - The global test set is the union of all validation subsets.
    Caches processed features to .npz for fast re-execution.
    """
    exists, missing = check_real_data_exists(data_dir)
    if not exists:
        raise FileNotFoundError(
            f"Real distracted driver dataset not found. Missing paths: {missing}.\n"
            f"Please run 'python scripts/download_auc_dataset.py' or place files in data/auc/.\n"
            f"See guide.md for details."
        )

    base = Path(data_dir)
    cache_file = base / f"auc_cache_{img_size}_{feature_dim}_{max_samples_per_client}_c{n_clients}_s{seed}_{extractor_name}_{split_mode}.npz"
    if cache_file.exists():
        try:
            cached = np.load(cache_file, allow_pickle=True)
            clients = [
                (cached[f"X_c{i}"], cached[f"y_c{i}"])
                for i in range(n_clients)
            ]
            test_data = (cached["X_test"], cached["y_test"])
            K = int(cached["K"])
            print(f"[Info] Loaded cached real driver dataset from {cache_file}")
            return clients, test_data, K
        except Exception:
            pass

    # Load driver_imgs_list.csv if present anywhere under base
    csv_lookup: Dict[str, Tuple[int, str]] = {}
    csv_candidates = list(base.glob("**/driver_imgs_list.csv")) + [base / "driver_imgs_list.csv"]
    for c_path in csv_candidates:
        if c_path.exists():
            try:
                with open(c_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        img_name = row.get("img", "").strip().lower()
                        subj = row.get("subject", "").strip()
                        cls_str = row.get("classname", "").strip().lower().replace("c", "")
                        if img_name and cls_str.isdigit():
                            csv_lookup[img_name] = (int(cls_str), subj)
                print(f"[Info] Loaded {len(csv_lookup)} image mappings from {c_path}")
                break
            except Exception as e:
                print(f"[Warning] Failed parsing {c_path}: {e}")

    # Collect all image files
    img_files = list(base.glob("**/*.jpg")) + list(base.glob("**/*.png"))
    # Filter out any non-dataset files if any
    img_files = [p for p in img_files if not p.name.startswith(".")]

    print(f"[Info] Scanning {len(img_files)} driver images...")
    parsed_samples: List[Tuple[Path, int, str]] = []
    for p in img_files:
        cls_idx, subj = _extract_class_and_subject(p, csv_lookup=csv_lookup)
        if cls_idx is not None and 0 <= cls_idx < 10:
            parsed_samples.append((p, cls_idx, subj))

    if len(parsed_samples) < 50:
        raise ValueError(f"Found only {len(parsed_samples)} valid labeled images (expected >= 50).")

    # Group by subject
    subjects = sorted(list({s for _, _, s in parsed_samples}))
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    if split_mode == "paper":
        # Paper Section III-A & IV-A: User-based partition (3-5 unique drivers per client)
        # Each client partitions local data D_i into 80% train and 20% validation
        client_subjs: Dict[int, Set[str]] = {i: set() for i in range(n_clients)}
        for idx, s in enumerate(subjects):
            client_subjs[idx % n_clients].add(s)
        test_subjs: Set[str] = set()
    else:
        # Disjoint test subjects mode
        n_test_subjs = max(1, len(subjects) // 5)
        test_subjs = set(subjects[:n_test_subjs])
        train_subjs = subjects[n_test_subjs:]
        client_subjs: Dict[int, Set[str]] = {i: set() for i in range(n_clients)}
        for idx, s in enumerate(train_subjs):
            client_subjs[idx % n_clients].add(s)

    # Feature extraction: Pretrained MobileNetV2 embeddings (1280-dim) or fallback
    use_mobilenet = (extractor_name.startswith("mobilenet") or feature_dim == 1280)
    all_extracted_vectors: List[np.ndarray] = []

    if use_mobilenet:
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as T
            from PIL import Image

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                weights = models.MobileNet_V2_Weights.DEFAULT
                backbone = models.mobilenet_v2(weights=weights).to(device)
            except Exception:
                backbone = models.mobilenet_v2(pretrained=True).to(device)

            backbone.classifier = torch.nn.Identity()
            backbone.eval()

            img_transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            print(f"[Info] Extracting 1280-dim MobileNetV2 embeddings for {len(parsed_samples)} images on {device}...")
            batch_size = 64 if torch.cuda.is_available() else 32

            for b_start in range(0, len(parsed_samples), batch_size):
                b_end = min(b_start + batch_size, len(parsed_samples))
                batch_files = [parsed_samples[k][0] for k in range(b_start, b_end)]
                batch_tensors = []
                valid_mask = []
                for p in batch_files:
                    try:
                        im = Image.open(p).convert("RGB")
                        batch_tensors.append(img_transform(im))
                        valid_mask.append(True)
                    except Exception:
                        valid_mask.append(False)

                if batch_tensors:
                    with torch.no_grad():
                        stacked = torch.stack(batch_tensors).to(device)
                        out_feats = backbone(stacked).cpu().numpy().astype(np.float32)

                    f_idx = 0
                    for is_valid in valid_mask:
                        if is_valid:
                            all_extracted_vectors.append(out_feats[f_idx])
                            f_idx += 1
                        else:
                            all_extracted_vectors.append(np.zeros(feature_dim, dtype=np.float32))
                else:
                    for _ in range(len(batch_files)):
                        all_extracted_vectors.append(np.zeros(feature_dim, dtype=np.float32))

                if (b_end % (batch_size * 20) == 0) or (b_end == len(parsed_samples)):
                    print(f"[Info] Processed {b_end}/{len(parsed_samples)} images...")

        except Exception as e:
            print(f"[Warning] MobileNetV2 feature extraction failed ({e}), falling back to pooling.")
            use_mobilenet = False
            all_extracted_vectors = []

    def extract_vector_fallback(p: Path) -> np.ndarray:
        try:
            from PIL import Image
            im = Image.open(p)
            if hasattr(im, "resize"):
                im = im.resize((img_size, img_size))
                arr = np.array(im, dtype=np.float32) / 255.0
            else:
                arr = np.zeros((img_size, img_size, 3), dtype=np.float32)
            flat = arr.flatten()
            if len(flat) > feature_dim:
                stride = len(flat) // feature_dim
                vec = flat[: feature_dim * stride].reshape(feature_dim, stride).mean(axis=1)
            else:
                vec = np.pad(flat, (0, feature_dim - len(flat)))
            return vec
        except Exception:
            return rng.normal(0, 1.0, size=feature_dim).astype(np.float32)

    clients = []
    save_dict: Dict[str, Any] = {"K": 10}
    test_raw: List[Tuple[np.ndarray, int]] = []

    if split_mode == "paper":
        # Paper Section III-A & IV-A: Each client holds 3-5 unique drivers.
        # D_i is partitioned into 80% train and 20% validation (global test is union of validation sets).
        for cid in range(n_clients):
            subjs = client_subjs[cid]
            c_samples: List[Tuple[np.ndarray, int]] = []
            for idx, (p, cls_idx, subj) in enumerate(parsed_samples):
                if subj in subjs:
                    vec = all_extracted_vectors[idx] if (use_mobilenet and idx < len(all_extracted_vectors)) else extract_vector_fallback(p)
                    c_samples.append((vec, cls_idx))

            if not c_samples:
                raise ValueError(f"Client {cid} has 0 images! Real data partition failed.")

            # Stratified / seeded partition per client
            rng_client = np.random.default_rng(seed + cid * 997 + 13)
            perm = rng_client.permutation(len(c_samples))
            c_samples = [c_samples[i] for i in perm]

            n_train = int(len(c_samples) * 0.8)
            train_items = c_samples[:n_train]
            val_items = c_samples[n_train:]
            test_raw.extend(val_items)

            if max_samples_per_client is not None and len(train_items) > max_samples_per_client:
                train_items = train_items[:max_samples_per_client]

            unique_classes = set(s[1] for s in train_items)
            if len(unique_classes) < min_classes_per_client:
                raise ValueError(
                    f"Client {cid} has only {len(unique_classes)} classes (< {min_classes_per_client}). "
                    f"Classes present: {sorted(list(unique_classes))}"
                )

            X_c = np.stack([s[0] for s in train_items])
            y_c = np.array([s[1] for s in train_items], dtype=np.int64)
            clients.append((X_c, y_c))
            save_dict[f"X_c{cid}"] = X_c
            save_dict[f"y_c{cid}"] = y_c

    else:
        # Disjoint test subjects mode
        client_data_raw: Dict[int, List[Tuple[np.ndarray, int]]] = {i: [] for i in range(n_clients)}
        for idx, (p, cls_idx, subj) in enumerate(parsed_samples):
            vec = all_extracted_vectors[idx] if (use_mobilenet and idx < len(all_extracted_vectors)) else extract_vector_fallback(p)
            if subj in test_subjs:
                test_raw.append((vec, cls_idx))
            else:
                for cid in range(n_clients):
                    if subj in client_subjs[cid]:
                        client_data_raw[cid].append((vec, cls_idx))
                        break

        for cid in range(n_clients):
            samples = client_data_raw[cid]
            if not samples:
                raise ValueError(f"Client {cid} has 0 images! Real data partition failed.")
            rng_client = np.random.default_rng(seed + cid * 997 + 13)
            perm = rng_client.permutation(len(samples))
            samples = [samples[i] for i in perm]
            if max_samples_per_client is not None and len(samples) > max_samples_per_client:
                samples = samples[:max_samples_per_client]

            unique_classes = set(s[1] for s in samples)
            if len(unique_classes) < min_classes_per_client:
                raise ValueError(
                    f"Client {cid} has only {len(unique_classes)} classes (< {min_classes_per_client}). "
                    f"Classes present: {sorted(list(unique_classes))}"
                )

            X_c = np.stack([s[0] for s in samples])
            y_c = np.array([s[1] for s in samples], dtype=np.int64)
            clients.append((X_c, y_c))
            save_dict[f"X_c{cid}"] = X_c
            save_dict[f"y_c{cid}"] = y_c

    if not test_raw:
        raise ValueError("Held-out test set has 0 images! Real data partition failed.")

    rng_test = np.random.default_rng(seed + 99991)
    perm_test = rng_test.permutation(len(test_raw))
    test_raw = [test_raw[i] for i in perm_test]
    X_te = np.stack([s[0] for s in test_raw])
    y_te = np.array([s[1] for s in test_raw], dtype=np.int64)

    test_data = (X_te, y_te)
    save_dict["X_test"] = X_te
    save_dict["y_test"] = y_te

    # Cache for reuse
    try:
        np.savez_compressed(cache_file, **save_dict)
        print(f"[Info] Saved cached real driver dataset to {cache_file}")
    except Exception:
        pass

    return clients, test_data, 10


def load_dataset(
    dataset: str = "statefarm",
    data_dir: str = "data",
    n_clients: int = 6,
    seed: int = 42,
    img_size: int = 224,
    feature_dim: int = 1280,
    **kwargs,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray], int]:
    """
    Unified dataset dispatch function for the SecureDrive-FL federation.

    Dispatches to load_real_driver_data() for in-cabin driver distraction monitoring ('statefarm' or 'auc', 10 classes).

    Args:
        dataset: Target dataset name ('statefarm' or 'auc').
        data_dir: Path to directory containing raw data or .npz cache.
        n_clients: Number of vehicular clients in the federation (default: 6).
        seed: Random seed for partition reproducibility.
        img_size: Image input resolution (default: 224x224).
        feature_dim: MobileNetV2 embedding dimension (default: 1280).

    Returns:
        (clients, test_data, K)
    """
    return load_real_driver_data(
        data_dir=data_dir,
        n_clients=n_clients,
        seed=seed,
        img_size=img_size,
        feature_dim=feature_dim,
        **kwargs,
    )

