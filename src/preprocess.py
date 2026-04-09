"""
Preprocessing pipeline for Convolutional Autoencoder (Track A).

Loads images from CA_Training_Data/, standardizes them, and saves
as .npy arrays ready for training.

Approach:
  - Single-channel grayscale
  - 320x240 pixels (native motor image resolution)
  - All images scaled to 0-255 range first (camera-agnostic):
      * 16-bit radiometric TIFFs: per-image min-max scaled to 0-255
      * 8-bit images: already in 0-255
  - Then Z-score normalization using train-set global stats: (pixel - mean) / std
  - Preserves relative intensity patterns across all source types
  - Drops solar_modules (24x40, too small to upscale meaningfully)

In production, the same two-step process runs on any camera input:
  1. Scale raw pixels to 0-255
  2. Z-score with saved mean/std
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path

# --- Config ---
ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "CA_Training_Data"
OUTPUT_DIR = ROOT / "data" / "CA_Preprocessed"
TARGET_SIZE = (320, 240)  # width, height

# Exclude solar_modules: 24x40 images are too small
TRAIN_FOLDERS = [
    "electric_motor",
    "induction_motor",
    "pv_om_inspection",
    "pv_thermal_inspection",
]

TEST_NORMAL_FOLDERS = [
    "electric_motor",
    "induction_motor",
]

TEST_FAULT_FOLDERS = [
    "electric_motor",
    "induction_motor",
]


def load_image_raw(filepath):
    """Load a single image as grayscale float32 scaled to 0-255 range.

    This makes the pipeline camera-agnostic:
      - 16-bit radiometric TIFFs (values like 6000-14000) get min-max
        scaled to 0-255, preserving the thermal contrast within the image
      - 8-bit PNGs/BMPs/JPGs are already in 0-255
    """
    img = Image.open(filepath)
    arr = np.array(img, dtype=np.float32)

    # Convert RGB/RGBA to grayscale by averaging channels
    if arr.ndim == 3:
        arr = arr.mean(axis=2)

    # Scale to common 0-255 range based on source bit depth
    if arr.max() > 255:
        # 16-bit radiometric: min-max scale to 0-255
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        else:
            arr = np.zeros_like(arr)
    # 8-bit images are already in 0-255, no conversion needed

    # Resize to target using LANCZOS (high-quality resampling)
    img_resized = Image.fromarray(arr, mode="F")
    img_resized = img_resized.resize(TARGET_SIZE, Image.LANCZOS)
    arr_resized = np.array(img_resized, dtype=np.float32)

    return arr_resized


def load_folder(input_folder):
    """Load all images from a folder as raw arrays."""
    images = []
    skipped = 0

    if not input_folder.exists():
        print(f"  MISSING: {input_folder}")
        return images, skipped

    files = sorted(os.listdir(input_folder))
    for fname in files:
        fpath = os.path.join(input_folder, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            img = load_image_raw(fpath)
            images.append(img)
        except Exception as e:
            print(f"  SKIP {fname}: {e}")
            skipped += 1

    return images, skipped


def main():
    print("=" * 60)
    print("PHASE 1: Load all training images (raw) to compute stats")
    print("=" * 60)

    # Load all train/normal images raw
    train_images = []
    for folder in TRAIN_FOLDERS:
        input_folder = INPUT_DIR / "train" / "normal" / folder
        print(f"Loading train/normal/{folder}...")
        imgs, skipped = load_folder(input_folder)
        print(f"  {len(imgs)} loaded, {skipped} skipped")
        train_images.extend(imgs)

    if not train_images:
        print("ERROR: No training images found.")
        return

    # Stack into single array and compute global stats
    train_arr = np.stack(train_images, axis=0)  # (N, 240, 320)
    global_mean = train_arr.mean()
    global_std = train_arr.std()

    print(f"\nTraining set: {train_arr.shape[0]} images")
    print(f"Global mean: {global_mean:.4f}")
    print(f"Global std:  {global_std:.4f}")

    print("\n" + "=" * 60)
    print("PHASE 2: Z-score normalize and save all splits")
    print("=" * 60)

    # Normalize and save training data
    train_normalized = (train_arr - global_mean) / global_std
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
        input_folder = INPUT_DIR / "test" / "normal" / folder
        print(f"  Loading test/normal/{folder}...")
        imgs, skipped = load_folder(input_folder)
        print(f"    {len(imgs)} loaded, {skipped} skipped")
        test_normal_images.extend(imgs)

    if test_normal_images:
        test_normal_arr = np.stack(test_normal_images, axis=0)
        test_normal_normalized = (test_normal_arr - global_mean) / global_std
        print(f"Test/normal normalized -- mean: {test_normal_normalized.mean():.4f}, std: {test_normal_normalized.std():.4f}")
        np.save(OUTPUT_DIR / "test_normal.npy", test_normal_normalized.astype(np.float32))
        print(f"Saved: {OUTPUT_DIR / 'test_normal.npy'} -- shape {test_normal_normalized.shape}")
        del test_normal_arr, test_normal_images, test_normal_normalized

    # Process test/fault
    print(f"\nLoading test/fault...")
    test_fault_images = []
    for folder in TEST_FAULT_FOLDERS:
        input_folder = INPUT_DIR / "test" / "fault" / folder
        print(f"  Loading test/fault/{folder}...")
        imgs, skipped = load_folder(input_folder)
        print(f"    {len(imgs)} loaded, {skipped} skipped")
        test_fault_images.extend(imgs)

    if test_fault_images:
        test_fault_arr = np.stack(test_fault_images, axis=0)
        test_fault_normalized = (test_fault_arr - global_mean) / global_std
        print(f"Test/fault normalized -- mean: {test_fault_normalized.mean():.4f}, std: {test_fault_normalized.std():.4f}")
        np.save(OUTPUT_DIR / "test_fault.npy", test_fault_normalized.astype(np.float32))
        print(f"Saved: {OUTPUT_DIR / 'test_fault.npy'} -- shape {test_fault_normalized.shape}")
        del test_fault_arr, test_fault_images, test_fault_normalized

    # Save normalization stats for inference later
    stats = {"mean": float(global_mean), "std": float(global_std)}
    stats_path = OUTPUT_DIR / "norm_stats.npy"
    np.save(stats_path, stats)
    print(f"\nSaved normalization stats: {stats_path}")
    print(f"  mean={global_mean:.4f}, std={global_std:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
