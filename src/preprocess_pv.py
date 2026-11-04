"""
PV-specialist preprocessing (HER-88).

Produces PV-only .npy arrays with PV-specific norm stats, separate from the
v2 multi-equipment pipeline in `preprocess.py`.

Splits the existing PV normals (`pv_om_inspection` + `pv_thermal_inspection`)
deterministically (seed=42) into ~90% train / ~10% test. The split is random
per frame, not per video or site, so neighbouring frames of the same video can
land on both sides; the "test" set is not independent of training:

  data/CA_Preprocessed/pv/
    train_normal.npy    (~8,020, 240, 320) float32
    test_normal.npy     (~890,   240, 320) float32
    test_fault.npy      (populated once HER-87 delivers the fault dataset)
    norm_stats.npy      {"mean": ..., "std": ...} specific to PV

When `data/CA_Training_Data/test/fault/pv/` exists and is non-empty, this
script also emits test_fault.npy. Until then, rerun after the fault dataset
lands.

main() takes the output folder, the healthy folders, the representation and
the fit (hlab.transform), so the variants (preprocess_pv_om.py,
preprocess_pv_matched.py) pass arguments instead of patching module globals.
This script's own run keeps "scale_by_max" + "stretch", the transform the
stored pv/ arrays and the PV specialist were made with.
"""

import os
import random
import numpy as np

from hlab.datasets import PV_FAULT_FOLDER, PV_FOLDERS, write_manifest
from hlab.paths import PREPROCESSED, RAW, ROOT
from hlab.transform import load_gray, normalize, to_model_input

# --- Config ---
INPUT_DIR = RAW
OUTPUT_DIR = PREPROCESSED / "pv"
HOLDOUT_FRACTION = 0.10
RANDOM_SEED = 42

# Which raw folders feed the PV pools is declared once, in hlab.datasets.
PV_TRAIN_FOLDERS = list(PV_FOLDERS["pv"])
PV_TEST_FAULT_FOLDER = PV_FAULT_FOLDER


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


def load_arrays(paths, label, representation="scale_by_max", fit="stretch"):
    """Load all (folder, path) tuples as 0-255 model-input arrays, with progress.

    hlab.transform with the given representation and fit, the same for healthy
    and fault files. The defaults are preprocess.py's ("scale_by_max",
    "stretch"), which carry the 16-bit / 8-bit format gap (HER-107).
    """
    arrs = []
    skipped = 0
    total = len(paths)
    for i, (_folder, fpath) in enumerate(paths):
        try:
            arrs.append(to_model_input(load_gray(fpath), representation, fit))
        except Exception as e:
            print(f"  SKIP {fpath.name}: {e}")
            skipped += 1
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"    {label}: loaded {i + 1}/{total}")
    return arrs, skipped


def main(output_dir=OUTPUT_DIR, folders=PV_TRAIN_FOLDERS, representation="scale_by_max", fit="stretch"):
    folders = list(folders)
    print("=" * 60)
    print("PV-SPECIALIST PREPROCESSING (HER-88)")
    print("=" * 60)
    print(f"  Healthy folders:  {folders}")
    print(f"  Transform:        representation={representation}, fit={fit}")
    print(f"  Holdout fraction: {HOLDOUT_FRACTION:.0%}")
    print(f"  Random seed:      {RANDOM_SEED}")
    print(f"  Output dir:       {output_dir}")
    print()

    # ---- Phase 1: collect PV normals and split deterministically ----
    print("Phase 1: collecting PV normal file paths...")
    pv_files = collect_files(folders, base_subpath="train/normal")
    n = len(pv_files)
    print(f"  Total PV normal images: {n}")
    for folder in folders:
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
    train_arrs, train_skipped = load_arrays(train_paths, "train", representation, fit)
    if not train_arrs:
        print("ERROR: no training arrays loaded")
        return
    train_arr = np.stack(train_arrs, axis=0)
    print(f"  Stacked: {train_arr.shape}  (skipped {train_skipped})")

    global_mean = float(train_arr.mean())
    global_std = float(train_arr.std())
    stats = {"mean": global_mean, "std": global_std}
    print(f"\n  PV-only mean: {global_mean:.4f}")
    print(f"  PV-only std:  {global_std:.4f}")
    print(f"  (v2 global was mean=171.73, std=45.97 across all equipment)")

    # ---- Phase 3: z-score normalize and save train ----
    print("\nPhase 3: normalizing and saving train_normal...")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_normed = normalize(train_arr, stats)
    np.save(output_dir / "train_normal.npy", train_normed.astype(np.float32))
    print(f"  Saved: {output_dir / 'train_normal.npy'}  shape {train_normed.shape}")
    del train_arrs, train_arr, train_normed

    # ---- Phase 4: holdout test normals ----
    print("\nPhase 4: loading + normalizing holdout test normals...")
    holdout_arrs, holdout_skipped = load_arrays(holdout_paths, "holdout", representation, fit)
    holdout_arr = np.stack(holdout_arrs, axis=0)
    holdout_normed = normalize(holdout_arr, stats)
    np.save(output_dir / "test_normal.npy", holdout_normed.astype(np.float32))
    print(f"  Saved: {output_dir / 'test_normal.npy'}  shape {holdout_normed.shape}  (skipped {holdout_skipped})")
    del holdout_arrs, holdout_arr, holdout_normed

    # ---- Phase 5: PV fault (if sourced via HER-87) ----
    fault_dir = INPUT_DIR / "test" / "fault" / PV_TEST_FAULT_FOLDER
    if fault_dir.exists() and any(fault_dir.iterdir()):
        print(f"\nPhase 5: loading PV fault test set from {fault_dir}...")
        fault_files = collect_files([PV_TEST_FAULT_FOLDER], base_subpath="test/fault")
        fault_arrs, fault_skipped = load_arrays(fault_files, "fault", representation, fit)
        if fault_arrs:
            fault_arr = np.stack(fault_arrs, axis=0)
            fault_normed = normalize(fault_arr, stats)
            np.save(output_dir / "test_fault.npy", fault_normed.astype(np.float32))
            print(f"  Saved: {output_dir / 'test_fault.npy'}  shape {fault_normed.shape}  (skipped {fault_skipped})")
        else:
            print("  No fault arrays loaded")
    else:
        print("\nPhase 5: skipped (no PV fault data yet).")
        print(f"  Expected path: {fault_dir}")
        print("  Once HER-87 delivers the fault dataset and it's copied in,")
        print("  rerun this script to produce test_fault.npy.")

    # ---- Phase 6: save PV-specific norm stats ----
    stats_path = output_dir / "norm_stats.npy"
    np.save(stats_path, stats)
    print(f"\nSaved PV norm stats: {stats_path}")
    print(f"  mean={global_mean:.4f}, std={global_std:.4f}")

    # ---- Persist the file-index split for reproducibility (hlab.datasets reads it) ----
    split_manifest = output_dir / "split_manifest.txt"
    sections = {
        "TRAIN": [p.relative_to(ROOT).as_posix() for _label, p in train_paths],
        "HOLDOUT": [p.relative_to(ROOT).as_posix() for _label, p in holdout_paths],
    }
    if fault_dir.exists() and any(fault_dir.iterdir()):
        sections["FAULT"] = [p.relative_to(ROOT).as_posix() for _label, p in fault_files]
    write_manifest(split_manifest, sections, header=[
        "PV specialist split manifest",
        f"seed={RANDOM_SEED}, holdout_fraction={HOLDOUT_FRACTION}",
        f"transform: representation={representation}, fit={fit}",
        f"{len(train_paths)} train / {len(holdout_paths)} holdout",
    ])
    print(f"Saved split manifest: {split_manifest}")

    print("\nDone.")


if __name__ == "__main__":
    main()
