"""
Generate reconstruction error heatmaps for visual inspection (HER-25).

For each input image, computes:
  - Original (denormalized for display)
  - Model reconstruction (denormalized)
  - Per-pixel squared error
  - Heatmap overlay (error map applied as red transparency over original)

Outputs PNG grids at results/heatmaps/:
  - gt_motors_grid.png       -- all 40 expert-annotated GT motor images
  - fault_samples_grid.png   -- 12 representative fault images (motor + transformer)
  - normal_samples_grid.png  -- 6 representative normal images (sanity check)

Quick-wins scope: visual side-by-side panels only. No mask-based IoU
comparison this round (mask format needs investigation).
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from PIL import Image

from train_autoencoder import ThermalAutoencoder

# --- Config ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "CA_Preprocessed"
TRAINING_DATA_DIR = ROOT / "data" / "CA_Training_Data"
GT_DIR = ROOT / "data" / "ground_truth" / "induction_motor_40_gt" / "Subset_40_Thermal_GT"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "results" / "heatmaps"
TARGET_SIZE = (320, 240)  # width, height — matches preprocess.py

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image_for_inference(path, mean, std):
    """Load a raw image, scale to 0-255, resize, then z-score normalize.

    Mirrors preprocess.py:load_image_raw + z-score step so heatmaps
    reflect what the model actually saw at training time.
    """
    img = Image.open(path)
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
    arr_resized = np.array(img_resized, dtype=np.float32)
    raw_0_255 = arr_resized.copy()
    normed = (arr_resized - mean) / std
    return raw_0_255, normed


def reconstruct(model, normed_batch):
    """Run model forward pass, return per-pixel reconstruction + error.

    normed_batch: (N, 240, 320) numpy
    Returns: (recon_normed, per_pixel_err) both (N, 240, 320)
    """
    x = torch.from_numpy(normed_batch[:, np.newaxis, :, :]).float().to(DEVICE)
    with torch.no_grad():
        recon = model(x)
    recon_np = recon.squeeze(1).cpu().numpy()
    err = (normed_batch - recon_np) ** 2
    return recon_np, err


def denorm(arr_normed, mean, std):
    """Reverse z-score back to 0-255 range for display."""
    return arr_normed * std + mean


def make_panel_row(ax_row, original, recon_denorm, error_map, title_prefix):
    """Render a 4-column row: original | reconstruction | error | overlay."""
    # Column 0: Original (grayscale)
    ax_row[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    ax_row[0].set_title(f"{title_prefix}", fontsize=8)
    ax_row[0].axis("off")

    # Column 1: Reconstruction
    ax_row[1].imshow(recon_denorm, cmap="gray", vmin=0, vmax=255)
    ax_row[1].axis("off")

    # Column 2: Error map (hot colormap)
    ax_row[2].imshow(error_map, cmap="hot")
    ax_row[2].axis("off")

    # Column 3: Overlay -- original in grayscale + red error tint
    ax_row[3].imshow(original, cmap="gray", vmin=0, vmax=255)
    # Normalize error for alpha (avoid divide-by-zero)
    err_max = error_map.max() if error_map.max() > 0 else 1.0
    alpha_map = np.clip(error_map / err_max, 0, 1) * 0.7
    red_overlay = np.zeros((*error_map.shape, 4))
    red_overlay[..., 0] = 1.0  # red channel
    red_overlay[..., 3] = alpha_map  # alpha
    ax_row[3].imshow(red_overlay)
    ax_row[3].axis("off")


def make_grid(image_paths, model, mean, std, out_path, title, ncols_panel=4):
    """Build and save an N-row × 4-col grid of (orig | recon | err | overlay)."""
    n = len(image_paths)
    if n == 0:
        print(f"  SKIP {out_path.name}: no input images")
        return

    # Load + run inference in a single batch
    raws, normeds = [], []
    for p in image_paths:
        raw, normed = load_image_for_inference(p, mean, std)
        raws.append(raw)
        normeds.append(normed)
    raws = np.stack(raws)
    normeds = np.stack(normeds)
    recons, errs = reconstruct(model, normeds)
    recons_denorm = denorm(recons, mean, std)

    # Plot
    fig, axes = plt.subplots(n, ncols_panel, figsize=(ncols_panel * 3, n * 2.2))
    if n == 1:
        axes = axes[np.newaxis, :]

    # Column headers (only above first row)
    headers = ["Original", "Reconstruction", "Error map", "Overlay"]
    for c, h in enumerate(headers):
        axes[0, c].set_title(h, fontsize=10, fontweight="bold")

    for i, path in enumerate(image_paths):
        # First-row title gets overwritten by header above; use ylabel instead
        label = f"{path.parent.name}/{path.name}"
        if i == 0:
            # Combine header + label
            axes[i, 0].set_title(f"Original\n{label}", fontsize=8)
        else:
            axes[i, 0].set_title(label, fontsize=8)

        axes[i, 0].imshow(raws[i], cmap="gray", vmin=0, vmax=255)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(recons_denorm[i], cmap="gray", vmin=0, vmax=255)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(errs[i], cmap="hot")
        axes[i, 2].axis("off")

        # Overlay
        axes[i, 3].imshow(raws[i], cmap="gray", vmin=0, vmax=255)
        em = errs[i]
        em_max = em.max() if em.max() > 0 else 1.0
        alpha = np.clip(em / em_max, 0, 1) * 0.7
        red = np.zeros((*em.shape, 4))
        red[..., 0] = 1.0
        red[..., 3] = alpha
        axes[i, 3].imshow(red)
        axes[i, 3].axis("off")

    # Suptitle
    mean_err = errs.mean()
    fig.suptitle(f"{title}\n(n={n}, mean per-pixel MSE = {mean_err:.4f})", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}  ({n} images, mean err {mean_err:.4f})")


def collect_gt_motors():
    """40 expert-annotated motor images (.bmp only; skip .tiff masks for this round)."""
    return sorted(GT_DIR.glob("*.bmp"))


def collect_fault_samples(n_motor=6, n_transformer=6):
    """Sample fault images across motor + transformer test sets."""
    motor_dir = TRAINING_DATA_DIR / "test" / "fault" / "induction_motor"
    trans_dir = TRAINING_DATA_DIR / "test" / "fault" / "transformer"

    motor_files = sorted(motor_dir.glob("*.bmp"))
    trans_files = sorted(trans_dir.glob("*.bmp"))

    # Take evenly-spaced samples
    def evenly_spaced(files, n):
        if len(files) <= n:
            return files
        step = len(files) / n
        return [files[int(i * step)] for i in range(n)]

    return evenly_spaced(motor_files, n_motor) + evenly_spaced(trans_files, n_transformer)


def collect_normal_samples(n=6):
    """Sample normal images for sanity check (low-error baseline)."""
    motor_normal = TRAINING_DATA_DIR / "test" / "normal" / "induction_motor"
    trans_normal = TRAINING_DATA_DIR / "test" / "normal" / "transformer"
    elec_normal = TRAINING_DATA_DIR / "test" / "normal" / "electric_motor"

    samples = []
    for d in [motor_normal, trans_normal, elec_normal]:
        files = sorted(d.iterdir())
        if files:
            samples.extend(files[:2])
    return samples[:n]


def main():
    print(f"Device: {DEVICE}")

    # Load norm stats
    stats = np.load(DATA_DIR / "norm_stats.npy", allow_pickle=True).item()
    mean, std = stats["mean"], stats["std"]
    print(f"Norm stats: mean={mean:.4f}, std={std:.4f}")

    # Load model
    model = ThermalAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "autoencoder_best.pth", weights_only=True))
    model.eval()
    print("Loaded autoencoder_best.pth\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. GT motors (40) ---
    gt_files = collect_gt_motors()
    print(f"GT motors: {len(gt_files)} images")
    make_grid(
        gt_files, model, mean, std,
        OUT_DIR / "gt_motors_grid.png",
        "Expert-annotated motor GTs -- reconstruction & error",
    )

    # --- 2. Fault samples (12) ---
    fault_files = collect_fault_samples(n_motor=6, n_transformer=6)
    print(f"\nFault samples: {len(fault_files)} images")
    make_grid(
        fault_files, model, mean, std,
        OUT_DIR / "fault_samples_grid.png",
        "Fault samples (motor + transformer)",
    )

    # --- 3. Normal samples (6) ---
    normal_files = collect_normal_samples(n=6)
    print(f"\nNormal samples: {len(normal_files)} images")
    make_grid(
        normal_files, model, mean, std,
        OUT_DIR / "normal_samples_grid.png",
        "Normal samples (sanity check -- error should be uniformly low)",
    )

    print(f"\nDone. Heatmaps in {OUT_DIR}/")


if __name__ == "__main__":
    main()
