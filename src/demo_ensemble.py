"""
Demo: ensemble inference pipeline stub (HER-81 preview).

For a single image, runs:
  1. ML path   — v2 autoencoder → per-image reconstruction error → ML verdict
  2. Rule path — surface-temperature proxy from the 8-bit image → threshold
                 comparison against per-equipment manual limits → rule verdict
  3. Ensemble  — combined decision with explanation

Produces a 5-panel matplotlib visualization:
  Original | Reconstruction | Error map | Overlay | Decision panel (text)

Usage:
  venv/Scripts/python.exe src/demo_ensemble.py
      --> runs on a curated 4-image sample (2 normal, 2 fault)

  venv/Scripts/python.exe src/demo_ensemble.py --image <path> --equipment transformer|pv

NOTE: this is a STUB, not HER-81's production implementation. The "temperature
extraction" step assumes a linear map from 8-bit pixel values to a plausible
camera operating range (20-120°C) — real radiometric-to-°C calibration is
HER-81 scope.
"""

import argparse
import sys
from pathlib import Path

# Force utf-8 stdout so Unicode chars (ΔT, °C, etc.) print cleanly on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from PIL import Image

from train_autoencoder import ThermalAutoencoder

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "CA_Preprocessed"
TRAINING_DATA_DIR = ROOT / "data" / "CA_Training_Data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR / "demo"
TARGET_SIZE = (320, 240)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default model config (v2 multi-equipment). --model pv overrides these in main().
MODEL_WEIGHTS = MODEL_DIR / "autoencoder_best.pth"
NORM_STATS_PATH = DATA_DIR / "norm_stats.npy"
# Default ML threshold (v2's F1-optimized cutoff from results/eval_metrics.npy)
ML_THRESHOLD = 0.000339

# Stub temperature calibration: assume the camera's 8-bit output maps linearly
# from its operational min to max. These are plausible inspection defaults, not
# per-image calibrations. Real radiometric extraction is HER-81.
CAMERA_TEMP_MIN = 20.0
CAMERA_TEMP_MAX = 120.0

# Tank-surface -> top-oil calibration offset (IEEE C57.91 literature, conservative)
TRANSFORMER_SURFACE_OFFSET_C = 10.0

# Ambient temperature default (Greenville summer estimate)
AMBIENT_C = 35.0

# Per-equipment thresholds — v1 config (HER-79 research + DAR-141)
THRESHOLDS = {
    "transformer": {
        "type": "top_oil",
        "ceiling_c": AMBIENT_C + 85.0,  # 120°C for Greenville summer
        "description": "Top-oil ceiling: ambient + 85°C (HI-105 §6)",
        "needs_surface_offset": True,
    },
    "pv": {
        # For 8-bit PV thermal imagery we can't read absolute cell °C, so the
        # rule layer uses the hot-spot ΔT proxy (p95 - mean, scaled to ~°C).
        # Tiers come from DAR-141's "hot-spot ΔT above mean" research.
        "type": "hotspot_delta",
        "tiers": [
            ("normal", 0, 15, "green"),
            ("warning", 15, 40, "gold"),
            ("alarm", 40, 75, "orange"),
            ("critical", 75, 999, "red"),
        ],
        "description": "PV hot-spot ΔT tiers (DAR-141 research)",
        "needs_surface_offset": False,
    },
}

# Curated demo samples — transformer (v2 model) arc.
# Filenames verified against the actual CA_Training_Data test folders.
DEFAULT_SAMPLES_V2 = [
    # 4-panel arc: NORMAL -> ML-only flag (imbalance) -> ML-only flag (mild fault) -> both flag (severe)
    (TRAINING_DATA_DIR / "test" / "normal" / "electric_motor" / "no_fault_105.png", "transformer", "baseline motor normal"),
    (TRAINING_DATA_DIR / "test" / "normal" / "transformer"    / "p1010.bmp",        "transformer", "transformer normal (flagged)"),
    (TRAINING_DATA_DIR / "test" / "fault"  / "transformer"    / "p2_80_p2003.bmp",  "transformer", "fault (mild, p2)"),
    (TRAINING_DATA_DIR / "test" / "fault"  / "transformer"    / "p9_600_p9086.bmp", "transformer", "fault (severe, p9)"),
]

# Curated demo samples — PV specialist arc.
# Selected by error ranking against the PV specialist (epoch 5, val_loss 0.017):
#   [1] holdout normal, lowest error  ............. confident NORMAL
#   [2] holdout normal, just under threshold  ..... borderline NORMAL
#   [3] PVMD hotspot, error just above threshold .. subtle FAULT (ML only)
#   [4] PVMD crack, highest error in dataset  ..... severe FAULT (both layers)
DEFAULT_SAMPLES_PV = [
    (TRAINING_DATA_DIR / "train" / "normal" / "pv_om_inspection"      / "sr_frame_002784.tiff", "pv", "normal (confident)"),
    (TRAINING_DATA_DIR / "train" / "normal" / "pv_thermal_inspection" / "THM_00078_01017.tif",  "pv", "normal (borderline)"),
    (TRAINING_DATA_DIR / "test"  / "fault"  / "pv" / "H237.jpeg", "pv", "fault (mild hotspot)"),
    (TRAINING_DATA_DIR / "test"  / "fault"  / "pv" / "C144.jpeg", "pv", "fault (severe crack)"),
]

DEFAULT_SAMPLES_BY_MODEL = {
    "v2": DEFAULT_SAMPLES_V2,
    "pv": DEFAULT_SAMPLES_PV,
}

# Back-compat alias; rebound in main() based on --model.
DEFAULT_SAMPLES = DEFAULT_SAMPLES_V2


# --- Preprocessing (mirrors preprocess.py / generate_heatmaps.py) ---

def load_image_for_inference(path, mean, std):
    """Scale to 0-255, resize, then z-score normalize.

    Returns: (raw_0_255 as (240,320), normed as (240,320))
    """
    img = Image.open(path)
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if arr.max() > 255:
        arr_min, arr_max = arr.min(), arr.max()
        arr = (arr - arr_min) / (arr_max - arr_min) * 255.0 if arr_max > arr_min else np.zeros_like(arr)
    img_resized = Image.fromarray(arr, mode="F").resize(TARGET_SIZE, Image.LANCZOS)
    arr_resized = np.array(img_resized, dtype=np.float32)
    raw = arr_resized.copy()
    normed = (arr_resized - mean) / std
    return raw, normed


# --- ML path ---

def run_ml_inference(model, normed):
    """Returns (recon_normed, per_pixel_err, mean_err)."""
    x = torch.from_numpy(normed[np.newaxis, np.newaxis, :, :]).float().to(DEVICE)
    with torch.no_grad():
        recon = model(x)
    recon_np = recon.squeeze().cpu().numpy()
    err = (normed - recon_np) ** 2
    mean_err = float(err.mean())
    return recon_np, err, mean_err


def ml_verdict(mean_err):
    if mean_err > ML_THRESHOLD:
        return "FAULT", f"MSE {mean_err:.6f} > threshold {ML_THRESHOLD:.6f}"
    return "NORMAL", f"MSE {mean_err:.6f} ≤ threshold {ML_THRESHOLD:.6f}"


# --- Rule path ---

def estimate_rule_signal(raw_0_255, equipment):
    """Compute the primary rule-layer signal for the given equipment type.

    Returns (signal_c, display_bits) where `signal_c` is the °C-ish value the
    rule tiers compare against, and `display_bits` is a dict of extra fields
    the decision panel can show (max pixel, p95, mean, etc.).

    Real radiometric calibration (per-pixel °C from a manufacturer-specific
    camera calibration) is HER-81 territory. The mappings here are stubs
    sized to produce usable demo signals from 8-bit imagery.
    """
    max_pixel = float(raw_0_255.max())

    if equipment == "transformer":
        # Transformer: linear map pixel -> absolute surface °C; max pixel
        # represents the hottest visible region on the tank.
        surface_c = CAMERA_TEMP_MIN + (max_pixel / 255.0) * (CAMERA_TEMP_MAX - CAMERA_TEMP_MIN)
        return surface_c, {
            "label": "Surface proxy",
            "value_str": f"{surface_c:.1f}°C",
            "extra": f"Max pixel: {max_pixel:.0f} / 255",
        }

    if equipment == "pv":
        # PV panels run hot and uniformly, so max-pixel alone is a bad signal.
        # Instead use p95 - mean as a hot-spot ΔT proxy scaled into the
        # DAR-141 hot-spot ΔT tier range (0-100-ish "°C").
        p95 = float(np.percentile(raw_0_255, 95))
        mean_pixel = float(raw_0_255.mean())
        delta_c = (p95 - mean_pixel) * (100.0 / 255.0)
        return delta_c, {
            "label": "Hot-spot ΔT proxy",
            "value_str": f"{delta_c:.1f}°C (p95 − mean scaled)",
            "extra": f"p95 pixel: {p95:.0f}, mean pixel: {mean_pixel:.0f}",
        }

    return 0.0, {"label": "Unknown", "value_str": "n/a", "extra": ""}


def rule_verdict(signal_c, equipment):
    """Apply per-equipment rule-based thresholds. Returns (severity, explanation)."""
    cfg = THRESHOLDS[equipment]
    if equipment == "transformer":
        calibrated_c = signal_c + TRANSFORMER_SURFACE_OFFSET_C if cfg["needs_surface_offset"] else signal_c
        ceiling = cfg["ceiling_c"]
        if calibrated_c > ceiling:
            return "CRITICAL", (
                f"Estimated top-oil = surface {signal_c:.1f}°C + 10°C offset = {calibrated_c:.1f}°C "
                f"exceeds ceiling {ceiling:.0f}°C"
            )
        return "PASS", (
            f"Estimated top-oil = surface {signal_c:.1f}°C + 10°C offset = {calibrated_c:.1f}°C "
            f"within ceiling {ceiling:.0f}°C"
        )
    elif equipment == "pv":
        for name, lo, hi, _color in cfg["tiers"]:
            if lo <= signal_c < hi:
                return name.upper(), (
                    f"Hot-spot ΔT proxy {signal_c:.1f}°C falls in '{name}' tier [{lo}-{hi}°C)"
                )
        return "OUT_OF_RANGE", f"Hot-spot ΔT proxy {signal_c:.1f}°C outside all tiers"
    return "UNKNOWN", "No rule for this equipment type"


# --- Ensemble ---

def ensemble_decision(ml, rule):
    """Combine ML + rule verdicts.

    - If either flags FAULT/CRITICAL/ALARM -> combined FLAGGED
    - If rule says WARNING -> combined MONITOR
    - Otherwise NORMAL
    """
    ml_flagged = (ml == "FAULT")
    rule_flagged = rule in ("CRITICAL", "ALARM", "FAULT")
    rule_monitor = rule in ("WARNING",)

    if ml_flagged and rule_flagged:
        return "FLAGGED (both layers)", "red"
    if ml_flagged:
        return "FLAGGED (ML only)", "red"
    if rule_flagged:
        return "FLAGGED (rules only)", "red"
    if rule_monitor:
        return "MONITOR (warning tier)", "gold"
    return "NORMAL", "green"


# --- Visualization ---

def render_panel(path, equipment, raw, recon_denorm, err, mean_err,
                 ml, ml_reason, signal_display, rule, rule_reason,
                 final, final_color, out_path, title_suffix=""):
    fig = plt.figure(figsize=(16, 4.2))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.6], wspace=0.18)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(raw, cmap="gray", vmin=0, vmax=255)
    ax0.set_title("Original", fontsize=10, fontweight="bold")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(recon_denorm, cmap="gray", vmin=0, vmax=255)
    ax1.set_title("Reconstruction", fontsize=10, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(err, cmap="hot")
    ax2.set_title("Per-pixel error", fontsize=10, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 3])
    ax3.imshow(raw, cmap="gray", vmin=0, vmax=255)
    em = err
    em_max = em.max() if em.max() > 0 else 1.0
    alpha = np.clip(em / em_max, 0, 1) * 0.7
    red = np.zeros((*em.shape, 4))
    red[..., 0] = 1.0
    red[..., 3] = alpha
    ax3.imshow(red)
    ax3.set_title("Overlay", fontsize=10, fontweight="bold")
    ax3.axis("off")

    # Decision panel
    ax4 = fig.add_subplot(gs[0, 4])
    ax4.axis("off")

    ml_color = "red" if ml == "FAULT" else "green"
    rule_color_map = {
        "PASS": "green", "NORMAL": "green",
        "WARNING": "gold", "ALARM": "orange",
        "CRITICAL": "red", "FAULT": "red",
        "OUT_OF_RANGE": "grey", "UNKNOWN": "grey",
    }
    rule_color = rule_color_map.get(rule, "grey")

    lines = [
        ("Equipment:", "black", "bold"),
        (f"  {equipment}", "black", "normal"),
        ("", "black", "normal"),
        ("─── ML Layer ───", "#1f4e79", "bold"),
        (f"  Verdict: {ml}", ml_color, "bold"),
        (f"  Mean MSE: {mean_err:.6f}", "black", "normal"),
        (f"  Threshold: {ML_THRESHOLD:.6f}", "grey", "normal"),
        ("", "black", "normal"),
        ("─── Rule-Based Layer ───", "#e67e22", "bold"),
        (f"  Verdict: {rule}", rule_color, "bold"),
        (f"  {signal_display['extra']}", "grey", "normal"),
        (f"  {signal_display['label']}: {signal_display['value_str']}", "black", "normal"),
        (f"  {rule_reason}", "grey", "normal"),
        ("", "black", "normal"),
        ("─── Ensemble Decision ───", "#1f4e79", "bold"),
        (f"  {final}", final_color, "bold"),
    ]

    y = 0.97
    for text, color, weight in lines:
        # Line-wrap long lines
        wrap_width = 46
        if len(text) > wrap_width and not text.startswith("─"):
            words = text.split(" ")
            chunks, current = [], ""
            for w in words:
                if len(current) + len(w) + 1 <= wrap_width:
                    current = (current + " " + w).strip()
                else:
                    chunks.append(current)
                    current = w
            if current:
                chunks.append(current)
            for chunk in chunks:
                ax4.text(0.0, y, chunk, transform=ax4.transAxes,
                         fontsize=9, color=color, fontweight=weight,
                         family="monospace")
                y -= 0.055
        else:
            ax4.text(0.0, y, text, transform=ax4.transAxes,
                     fontsize=9, color=color, fontweight=weight,
                     family="monospace")
            y -= 0.055

    # Stub warning footnote
    ax4.text(0.0, 0.02, "NOTE: surface→temp is a stub mapping for the demo (HER-81 = real calibration)",
             transform=ax4.transAxes, fontsize=7, color="grey", style="italic")

    title = f"{path.parent.name}/{path.name}"
    if title_suffix:
        title = f"{title} — {title_suffix}"
    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.99)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --- Main pipeline ---

def run_one(path, equipment, expected, model, mean, std, idx):
    raw, normed = load_image_for_inference(path, mean, std)
    recon_normed, err, mean_err = run_ml_inference(model, normed)
    recon_denorm = recon_normed * std + mean

    ml, ml_reason = ml_verdict(mean_err)
    signal_c, signal_display = estimate_rule_signal(raw, equipment)
    rule, rule_reason = rule_verdict(signal_c, equipment)
    final, final_color = ensemble_decision(ml, rule)

    # Console output
    print(f"\n[{idx}] {path.parent.name}/{path.name}  (expected: {expected}, equipment: {equipment})")
    print(f"    ML   : {ml}   (MSE {mean_err:.6f} vs threshold {ML_THRESHOLD:.6f})")
    print(f"    RULE : {rule}   — {rule_reason}")
    print(f"    FINAL: {final}")

    out_path = OUT_DIR / f"demo_{idx:02d}_{equipment}_{expected}_{path.stem}.png"
    render_panel(
        path, equipment, raw, recon_denorm, err, mean_err,
        ml, ml_reason, signal_display, rule, rule_reason,
        final, final_color, out_path,
        title_suffix=f"expected {expected}",
    )
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None,
                        help="Single image path. If omitted, runs the curated sample set.")
    parser.add_argument("--equipment", type=str, default="transformer",
                        choices=["transformer", "pv"],
                        help="Equipment type for rule-based layer.")
    parser.add_argument("--model", type=str, default="v2",
                        choices=["v2", "pv"],
                        help="Which autoencoder + norm stats to load. 'v2' = multi-equipment baseline, 'pv' = PV specialist.")
    parser.add_argument("--expected", type=str, default="unknown",
                        help="Expected label for the input image (cosmetic).")
    args = parser.parse_args()

    # Resolve model-specific paths, threshold, and curated sample set
    global ML_THRESHOLD, OUT_DIR, DEFAULT_SAMPLES
    DEFAULT_SAMPLES = DEFAULT_SAMPLES_BY_MODEL[args.model]
    if args.model == "pv":
        pv_dir = DATA_DIR / "pv"
        norm_stats_path = pv_dir / "norm_stats.npy"
        weights_path = MODEL_DIR / "autoencoder_pv_best.pth"
        pv_metrics_path = RESULTS_DIR / "pv_eval_metrics.npy"
        if pv_metrics_path.exists():
            pv_metrics = np.load(pv_metrics_path, allow_pickle=True).item()
            ML_THRESHOLD = float(pv_metrics["threshold"])
            print(f"Loaded PV threshold {ML_THRESHOLD:.6f} from eval metrics")
        else:
            # Fallback until evaluate_autoencoder_pv.py runs
            ML_THRESHOLD = 0.001
            print(f"PV eval metrics not found; using fallback threshold {ML_THRESHOLD:.6f}")
            print(f"  (Run src/evaluate_autoencoder_pv.py after fault data lands to calibrate.)")
        OUT_DIR = RESULTS_DIR / "demo" / "pv"
    else:
        norm_stats_path = NORM_STATS_PATH
        weights_path = MODEL_WEIGHTS

    # Norm stats + model
    stats = np.load(norm_stats_path, allow_pickle=True).item()
    mean, std = stats["mean"], stats["std"]
    print(f"Device: {DEVICE}")
    print(f"Model:  {args.model} ({weights_path.name})")
    print(f"Norm stats: mean={mean:.4f}, std={std:.4f}")

    model = ThermalAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    print(f"Loaded {weights_path.name}\n")

    if args.image:
        path = Path(args.image)
        if not path.exists():
            print(f"ERROR: image not found: {path}")
            return
        run_one(path, args.equipment, args.expected, model, mean, std, idx=1)
    else:
        print("Running curated sample set (4 images)...")
        for i, (path, equipment, expected) in enumerate(DEFAULT_SAMPLES, 1):
            if not path.exists():
                print(f"  SKIP [{i}] {path}: not found")
                continue
            run_one(path, equipment, expected, model, mean, std, idx=i)

    print(f"\nVisualizations saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
