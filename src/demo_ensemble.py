"""
Demo: ensemble inference pipeline stub (HER-81 preview).

For a single image, runs:
  1. ML path   — v2 autoencoder → per-image reconstruction error → ML verdict
  2. Rule path — contrast proxy from the 8-bit image (placeholder map, not
                 real temperature) → threshold comparison against
                 per-equipment manual limits → rule verdict
  3. Ensemble  — combined decision with explanation

Produces a 5-panel matplotlib visualization:
  Original | Reconstruction | Error map | Overlay | Decision panel (text)

Usage:
  venv/Scripts/python.exe src/demo_ensemble.py
      --> runs on a curated 4-image sample (2 normal, 2 fault)

  venv/Scripts/python.exe src/demo_ensemble.py --image <path> --equipment transformer|pv

  venv/Scripts/python.exe src/demo_ensemble.py --thresholds config/scada_greenville.json
      --> same pipeline, utility-specific setpoints instead of the IEC defaults

Rule-layer thresholds live in JSON, not in this file. The defaults ship at
config/thresholds_iec.json; --thresholds points at any file with the same
shape. The rule layer itself (schema, rule types, severities, ensemble) is
src/hlab/rules.py; this file only renders and drives it.

NOTE: this is a STUB, not HER-81's production implementation. The "temperature
extraction" step is a placeholder: a linear map of pixel 0 = 20°C, 255 = 120°C.
Images are min-max stretched per image first, so the numbers are not real
temperatures and the ΔT proxy measures contrast, not degrees. Real
radiometric-to-°C calibration is HER-81 scope.
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
from hlab.transform import denormalize, load_gray, normalize, to_model_input
from hlab.rules import DEFAULT_CONFIG_PATH, RuleConfig, combine, evaluate

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "CA_Preprocessed"
TRAINING_DATA_DIR = ROOT / "data" / "CA_Training_Data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR / "demo"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")


# Default model config (v2 multi-equipment). --model pv overrides these in main().
MODEL_WEIGHTS = MODEL_DIR / "autoencoder_best.pth"
NORM_STATS_PATH = DATA_DIR / "norm_stats.npy"
# Default ML threshold (v2's F1-optimized cutoff from results/eval_metrics.npy)
ML_THRESHOLD = 0.000339

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


# --- Display helpers ---

def load_original_for_display(path):
    """Load the image in its original form for human-viewable display.

    Returns (array, mode):
      - ("rgb", (H, W, 3) uint8) if the file was stored with color (e.g. JPEG
        with a thermal palette baked in).
      - ("mono", (H, W) float) if single-channel (raw radiometric data).
        Caller should render with a thermal colormap.
    """
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        # RGB or RGBA — drop alpha if present, return as uint8 RGB
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr, "rgb"
    return arr.astype(np.float32), "mono"


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


# --- Visualization ---

def render_panel(path, equipment, raw, recon_denorm, err, mean_err,
                 ml, ml_reason, verdict, decision, out_path, title_suffix=""):
    """Six-panel figure. `verdict` is a hlab.rules.RuleVerdict, `decision` a Decision."""
    fig = plt.figure(figsize=(18, 4.2))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 1.8], wspace=0.18)

    # Panel 0: thermal view (original color if RGB, else thermal colormap)
    ax_thermal = fig.add_subplot(gs[0, 0])
    orig_arr, orig_mode = load_original_for_display(path)
    if orig_mode == "rgb":
        ax_thermal.imshow(orig_arr)
        title = "Thermal (camera view)"
    else:
        # Apply a thermal colormap so the human viewer can see heat structure
        mn, mx = float(orig_arr.min()), float(orig_arr.max())
        if mx > mn:
            normed = (orig_arr - mn) / (mx - mn)
        else:
            normed = np.zeros_like(orig_arr)
        ax_thermal.imshow(normed, cmap="inferno")
        title = "Thermal (color-mapped)"
    ax_thermal.set_title(title, fontsize=10, fontweight="bold")
    ax_thermal.axis("off")

    # Panel 1: model input (grayscale, after preprocessing)
    ax0 = fig.add_subplot(gs[0, 1])
    ax0.imshow(raw, cmap="gray", vmin=0, vmax=255)
    ax0.set_title("Model input", fontsize=10, fontweight="bold")
    ax0.axis("off")

    # Panel 2: reconstruction
    ax1 = fig.add_subplot(gs[0, 2])
    ax1.imshow(recon_denorm, cmap="gray", vmin=0, vmax=255)
    ax1.set_title("Reconstruction", fontsize=10, fontweight="bold")
    ax1.axis("off")

    # Panel 3: per-pixel error
    ax2 = fig.add_subplot(gs[0, 3])
    ax2.imshow(err, cmap="hot")
    ax2.set_title("Per-pixel error", fontsize=10, fontweight="bold")
    ax2.axis("off")

    # Panel 4: overlay
    ax3 = fig.add_subplot(gs[0, 4])
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

    # Panel 5: Decision text
    ax4 = fig.add_subplot(gs[0, 5])
    ax4.axis("off")

    ml_color = "red" if ml == "FAULT" else "green"
    signal_display = verdict.display

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
        (f"  Verdict: {verdict.severity}", verdict.color, "bold"),
        (f"  {signal_display['extra']}", "grey", "normal"),
        (f"  {signal_display['label']}: {signal_display['value_str']}", "black", "normal"),
        (f"  {verdict.explanation}", "grey", "normal"),
        ("", "black", "normal"),
        ("─── Ensemble Decision ───", "#1f4e79", "bold"),
        (f"  {decision.label}", decision.color, "bold"),
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

def run_one(path, equipment, expected, model, stats, idx, config):
    # Same transform as preprocessing (hlab.transform), so the model sees what it trained on.
    raw = to_model_input(load_gray(path))
    normed = normalize(raw, stats)
    recon_normed, err, mean_err = run_ml_inference(model, normed)
    recon_denorm = denormalize(recon_normed, stats)

    ml, ml_reason = ml_verdict(mean_err)
    verdict = evaluate(config, raw, equipment)
    decision = combine(ml == "FAULT", verdict)

    # Console output
    print(f"\n[{idx}] {path.parent.name}/{path.name}  (expected: {expected}, equipment: {equipment})")
    print(f"    ML   : {ml}   (MSE {mean_err:.6f} vs threshold {ML_THRESHOLD:.6f})")
    print(f"    RULE : {verdict.severity}   — {verdict.explanation}")
    print(f"    FINAL: {decision.label}")

    out_path = OUT_DIR / f"demo_{idx:02d}_{equipment}_{expected}_{path.stem}.png"
    render_panel(
        path, equipment, raw, recon_denorm, err, mean_err,
        ml, ml_reason, verdict, decision, out_path,
        title_suffix=f"expected {expected}",
    )
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None,
                        help="Single image path. If omitted, runs the curated sample set.")
    parser.add_argument("--equipment", type=str, default="transformer",
                        help="Equipment type for rule-based layer. Must be a key under "
                             "'equipment' in the threshold config (default config defines "
                             "transformer, pv).")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Path to a rule-layer threshold JSON config. Defaults to "
                             f"{DEFAULT_CONFIG_PATH.name} (IEC / manufacturer values). "
                             "Point this at a utility's SCADA setpoints to swap them in.")
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

    # Rule-layer thresholds. Load + validate before touching the model so a bad
    # config fails in a second instead of after weights are on the device.
    try:
        config = RuleConfig.load(args.thresholds)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Thresholds: {config.name}  ({config.source})")

    # Every equipment type this run will touch must exist in the config.
    if args.image:
        needed = {args.equipment}
    else:
        needed = {equipment for _path, equipment, _expected in DEFAULT_SAMPLES}
    try:
        for equipment in sorted(needed):
            config.equipment(equipment)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    # Norm stats + model
    stats = np.load(norm_stats_path, allow_pickle=True).item()
    print(f"Device: {DEVICE}")
    print(f"Model:  {args.model} ({weights_path.name})")
    print(f"Norm stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    model = ThermalAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    print(f"Loaded {weights_path.name}\n")

    if args.image:
        path = Path(args.image)
        if not path.exists():
            print(f"ERROR: image not found: {path}")
            return 1
        run_one(path, args.equipment, args.expected, model, stats, idx=1, config=config)
    else:
        print("Running curated sample set (4 images)...")
        for i, (path, equipment, expected) in enumerate(DEFAULT_SAMPLES, 1):
            if not path.exists():
                print(f"  SKIP [{i}] {path}: not found")
                continue
            run_one(path, equipment, expected, model, stats, idx=i, config=config)

    print(f"\nVisualizations saved to: {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
