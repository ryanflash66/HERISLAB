"""
Preprocessing pipeline for Convolutional Autoencoder.

Loads images from CA_Training_Data/, standardizes them, and saves
as .npy arrays ready for training.

Approach:
  - Single-channel grayscale
  - 320x240 pixels (native motor image resolution)
  - All images brought to a common 0-255 numeric range first. This is NOT
    camera-agnostic; the two branches differ (see hlab.transform.to_model_input,
    representation "scale_by_max"):
      * 16-bit radiometric TIFFs: per-image min-max scaled to 0-255
      * 8-bit images: passed through unchanged
  - Then Z-score normalization using train-set global stats: (pixel - mean) / std
  - Within a 16-bit image the relative intensity pattern is kept; absolute
    temperature is discarded, and 8-bit and 16-bit sources end up with
    different brightness distributions (HER-107)
  - Drops solar_modules (24x40, too small to upscale meaningfully)

Untested on other cameras: running the same two steps (scale to 0-255, then
Z-score with saved mean/std) on a new camera's output is the plan, not a
verified property.
"""

import os
import numpy as np

from hlab.datasets import V2_FOLDERS, raw_files, write_manifest
from hlab.paths import PREPROCESSED, RAW, ROOT
from hlab.transform import load_gray, normalize, to_model_input

# --- Config ---
INPUT_DIR = RAW
OUTPUT_DIR = PREPROCESSED

# Which raw folders feed which split is declared once, in hlab.datasets
# (solar_modules is excluded there: 24x40 images are too small).
TRAIN_FOLDERS = list(V2_FOLDERS["train"])
TEST_NORMAL_FOLDERS = list(V2_FOLDERS["test_normal"])
TEST_FAULT_FOLDERS = list(V2_FOLDERS["test_fault"])


def load_folder(subdir, folder):
    """Load RAW/subdir/folder as 0-255 model-input arrays (not yet z-scored).

    Returns (images, skipped, frame_ids); frame_ids are repo-relative and
    aligned with images, so the split manifest matches the arrays.
    """
    images, frame_ids = [], []
    skipped = 0

    if not (INPUT_DIR / subdir / folder).exists():
        print(f"  MISSING: {INPUT_DIR / subdir / folder}")
        return images, skipped, frame_ids

    for frame_id in raw_files(subdir, folder):
        try:
            images.append(to_model_input(load_gray(ROOT / frame_id)))
            frame_ids.append(frame_id)
        except Exception as e:
            print(f"  SKIP {os.path.basename(frame_id)}: {e}")
            skipped += 1

    return images, skipped, frame_ids


def main():
    print("=" * 60)
    print("PHASE 1: Load all training images (raw) to compute stats")
    print("=" * 60)

    # Load all train/normal images raw
    train_images = []
    manifest = {"TRAIN": [], "TEST_NORMAL": [], "TEST_FAULT": []}
    for folder in TRAIN_FOLDERS:
        print(f"Loading train/normal/{folder}...")
        imgs, skipped, ids = load_folder("train/normal", folder)
        print(f"  {len(imgs)} loaded, {skipped} skipped")
        train_images.extend(imgs)
        manifest["TRAIN"].extend(ids)

    if not train_images:
        print("ERROR: No training images found.")
        return

    # Stack into single array and compute global stats
    train_arr = np.stack(train_images, axis=0)  # (N, 240, 320)
    global_mean = train_arr.mean()
    global_std = train_arr.std()
    stats = {"mean": float(global_mean), "std": float(global_std)}

    print(f"\nTraining set: {train_arr.shape[0]} images")
    print(f"Global mean: {global_mean:.4f}")
    print(f"Global std:  {global_std:.4f}")

    print("\n" + "=" * 60)
    print("PHASE 2: Z-score normalize and save all splits")
    print("=" * 60)

    # Normalize and save training data
    train_normalized = normalize(train_arr, stats)
    print(f"\nTrain normalized -- mean: {train_normalized.mean():.4f}, std: {train_normalized.std():.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(OUTPUT_DIR / "train_normal.npy", train_normalized.astype(np.float32))
    print(f"Saved: {OUTPUT_DIR / 'train_normal.npy'} -- shape {train_normalized.shape}")

    # Free memory
    del train_arr, train_images, train_normalized

    # Process test/normal
    print(f"\nLoading test/normal...")
    test_normal_images = []
    for folder in TEST_NORMAL_FOLDERS:
        print(f"  Loading test/normal/{folder}...")
        imgs, skipped, ids = load_folder("test/normal", folder)
        print(f"    {len(imgs)} loaded, {skipped} skipped")
        test_normal_images.extend(imgs)
        manifest["TEST_NORMAL"].extend(ids)

    if test_normal_images:
        test_normal_arr = np.stack(test_normal_images, axis=0)
        test_normal_normalized = normalize(test_normal_arr, stats)
        print(f"Test/normal normalized -- mean: {test_normal_normalized.mean():.4f}, std: {test_normal_normalized.std():.4f}")
        np.save(OUTPUT_DIR / "test_normal.npy", test_normal_normalized.astype(np.float32))
        print(f"Saved: {OUTPUT_DIR / 'test_normal.npy'} -- shape {test_normal_normalized.shape}")
        del test_normal_arr, test_normal_images, test_normal_normalized

    # Process test/fault
    print(f"\nLoading test/fault...")
    test_fault_images = []
    for folder in TEST_FAULT_FOLDERS:
        print(f"  Loading test/fault/{folder}...")
        imgs, skipped, ids = load_folder("test/fault", folder)
        print(f"    {len(imgs)} loaded, {skipped} skipped")
        test_fault_images.extend(imgs)
        manifest["TEST_FAULT"].extend(ids)

    if test_fault_images:
        test_fault_arr = np.stack(test_fault_images, axis=0)
        test_fault_normalized = normalize(test_fault_arr, stats)
        print(f"Test/fault normalized -- mean: {test_fault_normalized.mean():.4f}, std: {test_fault_normalized.std():.4f}")
        np.save(OUTPUT_DIR / "test_fault.npy", test_fault_normalized.astype(np.float32))
        print(f"Saved: {OUTPUT_DIR / 'test_fault.npy'} -- shape {test_fault_normalized.shape}")
        del test_fault_arr, test_fault_images, test_fault_normalized

    # Save normalization stats for inference later
    stats_path = OUTPUT_DIR / "norm_stats.npy"
    np.save(stats_path, stats)
    print(f"\nSaved normalization stats: {stats_path}")
    print(f"  mean={global_mean:.4f}, std={global_std:.4f}")

    # Which raw file sits at which array row (hlab.datasets reads this)
    manifest_path = OUTPUT_DIR / "split_manifest.txt"
    write_manifest(manifest_path, manifest, header=[
        "v2 multi-equipment split manifest (preprocess.py)",
        f"{len(manifest['TRAIN'])} train / {len(manifest['TEST_NORMAL'])} test normal / "
        f"{len(manifest['TEST_FAULT'])} test fault",
    ])
    print(f"Saved split manifest: {manifest_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
