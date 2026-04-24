"""
PV-specialist preprocessing (HER-88).

Produces PV-only .npy arrays with PV-specific norm stats, separate from the
v2 multi-equipment pipeline in `preprocess.py`.

Splits the existing PV normals (`pv_om_inspection` + `pv_thermal_inspection`)
deterministically (seed=42) into ~90% train / ~10% held-out test so the
specialist can be evaluated on data it never saw at training time:

  data/CA_Preprocessed/pv/
    train_normal.npy    (~8,020, 240, 320) float32
    test_normal.npy     (~890,   240, 320) float32
    test_fault.npy      (populated once HER-87 delivers the fault dataset)
    norm_stats.npy      {"mean": ..., "std": ...} specific to PV

When `data/CA_Training_Data/test/fault/pv/` exists and is non-empty, this
script also emits test_fault.npy. Until then, rerun after the fault dataset
lands.
"""

import os
import random
import numpy as np
from PIL import Image
from pathlib import Path

# --- Config ---
ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "CA_Training_Data"
OUTPUT_DIR = ROOT / "data" / "CA_Preprocessed" / "pv"
TARGET_SIZE = (320, 240)   # width, height
HOLDOUT_FRACTION = 0.10
RANDOM_SEED = 42

PV_TRAIN_FOLDERS = [
    "pv_om_inspection",
    "pv_thermal_inspection",
]

# Name of the fault subfolder under test/fault/. Waiting on HER-87.
PV_TEST_FAULT_FOLDER = "pv"


def load_image_raw(filepath):
    """Load a single image as grayscale float32 scaled to 0-255.

    Mirrors preprocess.py so the PV specialist sees identically formatted
    inputs (any divergence here would invalidate apples-to-apples comparison
    against the v2 baseline).
    """
    img = Image.open(filepath)
    arr = np.array(img, dtype=np.float32)

    if arr.ndim == 3:
        arr = arr.mean(axis=2)

    if arr.max() > 255:
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        else:
            arr = np.zeros_like(arr)

    img_resized = Image.fromarray(arr, mode="F").resize(TARGET_SIZE, Image.LANCZOS)
    return np.array(img_resized, dtype=np.float32)


def collect_files(folders, base_subpath):
    """Gather (folder_name, full_path) tuples under INPUT_DIR / base_subpath / folder."""
    out = []
    for folder in folders:
        folder_path = INPUT_DIR / base_subpath / folder
        if not folder_path.exists():
            print(f"  MISSING: {folder_path}")
            continue
        for fname in sorted(os.listdir(folder_path)):
            fpath = folder_path / fname
            if fpath.is_file():
                out.append((folder, fpath))
    return out


def load_arrays(paths, label):
    """Load all arrays from (folder, path) tuples with progress."""
    arrs = []
    skipped = 0
    total = len(paths)
    for i, (_folder, fpath) in enumerate(paths):
        try:
            arrs.append(load_image_raw(fpath))
        except Exception as e:
            print(f"  SKIP {fpath.name}: {e}")
            skipped += 1
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"    {label}: loaded {i + 1}/{total}")
    return arrs, skipped


def main():
    print("=" * 60)
    print("PV-SPECIALIST PREPROCESSING (HER-88)")
    print("=" * 60)
    print(f"  Holdout fraction: {HOLDOUT_FRACTION:.0%}")
    print(f"  Random seed:      {RANDOM_SEED}")
    print(f"  Output dir:       {OUTPUT_DIR}")
    print()

    # ---- Phase 1: collect PV normals and split deterministically ----
    print("Phase 1: collecting PV normal file paths...")
    pv_files = collect_files(PV_TRAIN_FOLDERS, base_subpath="train/normal")
    n = len(pv_files)
    print(f"  Total PV normal images: {n}")
    for folder in PV_TRAIN_FOLDERS:
        count = sum(1 for (f, _) in pv_files if f == folder)
        print(f"    {folder}: {count}")

    if n == 0:
        print("ERROR: no PV files found. Check data/CA_Training_Data/train/normal/pv_* paths.")
        return

    rng = random.Random(RANDOM_SEED)
    indices = list(range(n))
    rng.shuffle(indices)
    split = int(n * (1 - HOLDOUT_FRACTION))
    train_idx = sorted(indices[:split])
    test_idx = sorted(indices[split:])
    print(f"  Split: {len(train_idx)} train / {len(test_idx)} holdout  (seed={RANDOM_SEED})")

    train_paths = [pv_files[i] for i in train_idx]
    holdout_paths = [pv_files[i] for i in test_idx]

    # ---- Phase 2: load training arrays, compute PV-only norm stats ----
    print("\nPhase 2: loading training images...")
    train_arrs, train_skipped = load_arrays(train_paths, "train")
    if not train_arrs:
        print("ERROR: no training arrays loaded")
        return
    train_arr = np.stack(train_arrs, axis=0)
    print(f"  Stacked: {train_arr.shape}  (skipped {train_skipped})")

    global_mean = float(train_arr.mean())
    global_std = float(train_arr.std())
    print(f"\n  PV-only mean: {global_mean:.4f}")
    print(f"  PV-only std:  {global_std:.4f}")
    print(f"  (v2 global was mean=171.73, std=45.97 across all equipment)")

    # ---- Phase 3: z-score normalize and save train ----
    print("\nPhase 3: normalizing and saving train_normal...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_normed = (train_arr - global_mean) / global_std
    np.save(OUTPUT_DIR / "train_normal.npy", train_normed.astype(np.float32))
    print(f"  Saved: {OUTPUT_DIR / 'train_normal.npy'}  shape {train_normed.shape}")
    del train_arrs, train_arr, train_normed

    # ---- Phase 4: holdout test normals ----
    print("\nPhase 4: loading + normalizing holdout test normals...")
    holdout_arrs, holdout_skipped = load_arrays(holdout_paths, "holdout")
    holdout_arr = np.stack(holdout_arrs, axis=0)
    holdout_normed = (holdout_arr - global_mean) / global_std
    np.save(OUTPUT_DIR / "test_normal.npy", holdout_normed.astype(np.float32))
    print(f"  Saved: {OUTPUT_DIR / 'test_normal.npy'}  shape {holdout_normed.shape}  (skipped {holdout_skipped})")
    del holdout_arrs, holdout_arr, holdout_normed

    # ---- Phase 5: PV fault (if sourced via HER-87) ----
    fault_dir = INPUT_DIR / "test" / "fault" / PV_TEST_FAULT_FOLDER
    if fault_dir.exists() and any(fault_dir.iterdir()):
        print(f"\nPhase 5: loading PV fault test set from {fault_dir}...")
        fault_files = collect_files([PV_TEST_FAULT_FOLDER], base_subpath="test/fault")
        fault_arrs, fault_skipped = load_arrays(fault_files, "fault")
        if fault_arrs:
            fault_arr = np.stack(fault_arrs, axis=0)
            fault_normed = (fault_arr - global_mean) / global_std
            np.save(OUTPUT_DIR / "test_fault.npy", fault_normed.astype(np.float32))
            print(f"  Saved: {OUTPUT_DIR / 'test_fault.npy'}  shape {fault_normed.shape}  (skipped {fault_skipped})")
        else:
            print("  No fault arrays loaded")
    else:
        print("\nPhase 5: skipped (no PV fault data yet).")
        print(f"  Expected path: {fault_dir}")
        print("  Once HER-87 delivers the fault dataset and it's copied in,")
        print("  rerun this script to produce test_fault.npy.")

    # ---- Phase 6: save PV-specific norm stats ----
    stats = {"mean": global_mean, "std": global_std}
    stats_path = OUTPUT_DIR / "norm_stats.npy"
    np.save(stats_path, stats)
    print(f"\nSaved PV norm stats: {stats_path}")
    print(f"  mean={global_mean:.4f}, std={global_std:.4f}")

    # ---- Persist the file-index split for reproducibility ----
    split_manifest = OUTPUT_DIR / "split_manifest.txt"
    with split_manifest.open("w", encoding="utf-8") as f:
        f.write(f"# PV specialist split manifest\n")
        f.write(f"# seed={RANDOM_SEED}, holdout_fraction={HOLDOUT_FRACTION}\n")
        f.write(f"# {len(train_paths)} train / {len(holdout_paths)} holdout\n\n")
        f.write("[TRAIN]\n")
        for _label, p in train_paths:
            f.write(f"{p.relative_to(ROOT).as_posix()}\n")
        f.write("\n[HOLDOUT]\n")
        for _label, p in holdout_paths:
            f.write(f"{p.relative_to(ROOT).as_posix()}\n")
    print(f"Saved split manifest: {split_manifest}")

    print("\nDone.")


if __name__ == "__main__":
    main()
