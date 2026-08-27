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

  venv/Scripts/python.exe src/demo_ensemble.py --thresholds config/scada_greenville.json
      --> same pipeline, utility-specific setpoints instead of the IEC defaults

Rule-layer thresholds live in JSON, not in this file. The defaults ship at
config/thresholds_iec.json; --thresholds points at any file with the same
shape. See load_threshold_config() for the schema and RULE_TYPES for the
supported rule types.

NOTE: this is a STUB, not HER-81's production implementation. The "temperature
extraction" step assumes a linear map from 8-bit pixel values to a plausible
camera operating range (20-120°C) — real radiometric-to-°C calibration is
HER-81 scope.
"""

import argparse
import json
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
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")


# Default model config (v2 multi-equipment). --model pv overrides these in main().
MODEL_WEIGHTS = MODEL_DIR / "autoencoder_best.pth"
NORM_STATS_PATH = DATA_DIR / "norm_stats.npy"
# Default ML threshold (v2's F1-optimized cutoff from results/eval_metrics.npy)
ML_THRESHOLD = 0.000339

# Rule-layer thresholds (camera calibration, ambient, per-equipment setpoints)
# live here. Swap in utility SCADA setpoints with --thresholds <path>.
DEFAULT_THRESHOLD_CONFIG = ROOT / "config" / "thresholds_iec.json"

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


# --- Rule path: severity vocabulary ---

# Every severity a rule type can emit is declared here once. `action` drives
# the ensemble decision, `color` drives the decision panel. A severity that
# isn't in this table is an error, not a silent pass — see severity_entry().
SEVERITY_REGISTRY = {
    "PASS":     {"action": "pass",    "color": "green"},
    "NORMAL":   {"action": "pass",    "color": "green"},
    "WARNING":  {"action": "monitor", "color": "gold"},
    "ALARM":    {"action": "flag",    "color": "orange"},
    "CRITICAL": {"action": "flag",    "color": "red"},
    "FAULT":    {"action": "flag",    "color": "red"},
    # Emitted when a signal falls outside every configured tier. Reachable in
    # practice: a hot spot covering <5% of pixels puts p95 below the mean, so
    # the ΔT proxy goes negative and misses the 0-999 tier range. 'pass' keeps
    # the pre-config behavior, but it IS a fail-open — a config whose tiers
    # cover the full signal range never produces this, and switching it to
    # 'monitor' surfaces the gap instead of swallowing it.
    "OUT_OF_RANGE": {"action": "pass", "color": "grey"},
}


def severity_entry(severity):
    """Look up a severity. Raises ValueError on an unknown label.

    Fail-loud on purpose: an unrecognized severity used to fall through the
    ensemble's membership tests and read as "not flagged", so a config using
    a different vocabulary (ALERT vs ALARM) would silently pass faults.
    """
    entry = SEVERITY_REGISTRY.get(severity)
    if entry is None:
        raise ValueError(
            f"unknown severity {severity!r}; known severities: "
            f"{', '.join(sorted(SEVERITY_REGISTRY))}. Add it to SEVERITY_REGISTRY "
            f"before referencing it from a threshold config."
        )
    return entry


def severity_action(severity):
    """'pass' | 'monitor' | 'flag' for a severity label."""
    return severity_entry(severity)["action"]


def severity_color(severity):
    """Display color for a severity label."""
    return severity_entry(severity)["color"]


# --- Rule path: config validation helpers ---

def _require(condition, message):
    if not condition:
        raise ValueError(f"threshold config: {message}")


def _as_number(value, field):
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"'{field}' must be a number, got {value!r}",
    )
    return float(value)


# --- Rule path: signal extraction (one per rule type) ---

def _signal_top_oil(raw_0_255, eq_cfg, config, equipment):
    """Linear map pixel -> absolute surface °C.

    Max pixel represents the hottest visible region on the tank.
    """
    camera = config["camera"]
    pixel_max = camera["pixel_max"]
    temp_min, temp_max = camera["temp_min_c"], camera["temp_max_c"]
    max_pixel = float(raw_0_255.max())
    surface_c = temp_min + (max_pixel / pixel_max) * (temp_max - temp_min)
    return surface_c, {
        "label": "Surface proxy",
        "value_str": f"{surface_c:.1f}°C",
        "extra": f"Max pixel: {max_pixel:.0f} / {pixel_max:.0f}",
    }


def _signal_hotspot_delta(raw_0_255, eq_cfg, config, equipment):
    """Hot-spot ΔT proxy: (percentile - mean), scaled into the tier range.

    PV panels run hot and uniformly, so max-pixel alone is a bad signal.
    """
    pixel_max = config["camera"]["pixel_max"]
    percentile = eq_cfg["percentile"]
    p_hi = float(np.percentile(raw_0_255, percentile))
    mean_pixel = float(raw_0_255.mean())
    delta_c = (p_hi - mean_pixel) * (eq_cfg["delta_span_c"] / pixel_max)
    return delta_c, {
        "label": "Hot-spot ΔT proxy",
        "value_str": f"{delta_c:.1f}°C (p{percentile:g} − mean scaled)",
        "extra": f"p{percentile:g} pixel: {p_hi:.0f}, mean pixel: {mean_pixel:.0f}",
    }


# --- Rule path: verdict evaluation (one per rule type) ---

def _resolve_ceiling_c(eq_cfg, config):
    """Absolute ceiling, either given directly or derived from ambient.

    SCADA setpoints are absolute (`ceiling_c`); the IEC default is expressed
    relative to ambient (`ceiling_over_ambient_c`). Validation guarantees
    exactly one is present.
    """
    if "ceiling_c" in eq_cfg:
        return float(eq_cfg["ceiling_c"])
    return float(config["ambient_c"]) + float(eq_cfg["ceiling_over_ambient_c"])


def _verdict_top_oil(signal_c, eq_cfg, config, equipment):
    offset_c = eq_cfg["surface_offset_c"]
    calibrated_c = signal_c + offset_c
    ceiling_c = _resolve_ceiling_c(eq_cfg, config)
    detail = (
        f"Estimated top-oil = surface {signal_c:.1f}°C + {offset_c:.0f}°C offset "
        f"= {calibrated_c:.1f}°C "
    )
    if calibrated_c > ceiling_c:
        return "CRITICAL", detail + f"exceeds ceiling {ceiling_c:.0f}°C"
    return "PASS", detail + f"within ceiling {ceiling_c:.0f}°C"


def _verdict_hotspot_delta(signal_c, eq_cfg, config, equipment):
    for tier in eq_cfg["tiers"]:
        lo, hi = tier["min_c"], tier["max_c"]
        if lo <= signal_c < hi:
            severity = tier["severity"]
            return severity.upper(), (
                f"Hot-spot ΔT proxy {signal_c:.1f}°C falls in "
                f"'{severity}' tier [{lo}-{hi}°C)"
            )
    # No tier matched. See SEVERITY_REGISTRY["OUT_OF_RANGE"] for why this is a
    # verdict rather than an error.
    return "OUT_OF_RANGE", f"Hot-spot ΔT proxy {signal_c:.1f}°C outside all tiers"


# --- Rule path: per-type validation ---

def _validate_top_oil(name, eq_cfg):
    _as_number(eq_cfg.get("surface_offset_c"), f"equipment.{name}.surface_offset_c")
    has_absolute = "ceiling_c" in eq_cfg
    has_relative = "ceiling_over_ambient_c" in eq_cfg
    _require(
        has_absolute or has_relative,
        f"equipment.{name} needs either 'ceiling_c' (absolute, e.g. a SCADA setpoint) "
        f"or 'ceiling_over_ambient_c' (derived from ambient_c)",
    )
    _require(
        not (has_absolute and has_relative),
        f"equipment.{name} sets both 'ceiling_c' and 'ceiling_over_ambient_c'; pick one",
    )
    field = "ceiling_c" if has_absolute else "ceiling_over_ambient_c"
    _as_number(eq_cfg[field], f"equipment.{name}.{field}")


def _validate_hotspot_delta(name, eq_cfg):
    percentile = _as_number(eq_cfg.get("percentile"), f"equipment.{name}.percentile")
    _require(
        0 <= percentile <= 100,
        f"equipment.{name}.percentile must be within 0-100, got {percentile}",
    )
    _as_number(eq_cfg.get("delta_span_c"), f"equipment.{name}.delta_span_c")

    tiers = eq_cfg.get("tiers")
    _require(
        isinstance(tiers, list) and tiers,
        f"equipment.{name}.tiers must be a non-empty list",
    )

    spans = []
    for i, tier in enumerate(tiers):
        where = f"equipment.{name}.tiers[{i}]"
        _require(isinstance(tier, dict), f"{where} must be an object")
        severity = tier.get("severity")
        _require(isinstance(severity, str), f"{where}.severity must be a string")
        try:
            severity_entry(severity.upper())
        except ValueError as exc:
            raise ValueError(f"threshold config: {where}.severity: {exc}") from exc
        lo = _as_number(tier.get("min_c"), f"{where}.min_c")
        hi = _as_number(tier.get("max_c"), f"{where}.max_c")
        _require(lo < hi, f"{where}: min_c ({lo}) must be less than max_c ({hi})")
        spans.append((lo, hi, i))

    spans.sort()
    for (lo_a, hi_a, i_a), (lo_b, _hi_b, i_b) in zip(spans, spans[1:]):
        _require(
            hi_a <= lo_b,
            f"equipment.{name}.tiers[{i_a}] [{lo_a}-{hi_a}) overlaps "
            f"tiers[{i_b}] [{lo_b}-...)",
        )


# Rule-type registry — this is what `type` in the config dispatches on.
# Adding a rule type means adding one entry here, not editing the pipeline.
RULE_TYPES = {
    "top_oil": {
        "signal": _signal_top_oil,
        "verdict": _verdict_top_oil,
        "validate": _validate_top_oil,
    },
    "hotspot_delta": {
        "signal": _signal_hotspot_delta,
        "verdict": _verdict_hotspot_delta,
        "validate": _validate_hotspot_delta,
    },
}


# --- Rule path: config loading ---

def validate_threshold_config(config):
    """Check a parsed config end to end. Raises ValueError on the first problem.

    Everything a rule type needs is checked here so failures surface at load
    time with a file path attached, rather than mid-run on image 3 of 4.
    """
    _require(isinstance(config, dict), "top level must be an object")
    _as_number(config.get("ambient_c"), "ambient_c")

    camera = config.get("camera")
    _require(isinstance(camera, dict), "'camera' must be an object")
    for field in ("temp_min_c", "temp_max_c", "pixel_max"):
        _as_number(camera.get(field), f"camera.{field}")
    _require(
        camera["temp_max_c"] > camera["temp_min_c"],
        "camera.temp_max_c must be greater than camera.temp_min_c",
    )
    _require(camera["pixel_max"] > 0, "camera.pixel_max must be positive")

    equipment_map = config.get("equipment")
    _require(
        isinstance(equipment_map, dict) and equipment_map,
        "'equipment' must be a non-empty object",
    )

    for name, eq_cfg in equipment_map.items():
        _require(isinstance(eq_cfg, dict), f"equipment.{name} must be an object")
        rule_type = eq_cfg.get("type")
        _require(
            rule_type in RULE_TYPES,
            f"equipment.{name}.type {rule_type!r} is not a known rule type; "
            f"known types: {', '.join(sorted(RULE_TYPES))}",
        )
        RULE_TYPES[rule_type]["validate"](name, eq_cfg)

    return config


def load_threshold_config(path=None):
    """Load and validate a rule-layer threshold config.

    Schema (see config/thresholds_iec.json):
      ambient_c            number  -- site ambient, used by ceiling_over_ambient_c
      camera               object  -- temp_min_c, temp_max_c, pixel_max
      equipment            object  -- keyed by equipment name; each entry has a
                                      `type` from RULE_TYPES plus that type's fields

    `path=None` loads the shipped IEC defaults.
    """
    path = Path(path) if path else DEFAULT_THRESHOLD_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"threshold config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        try:
            config = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"threshold config {path} is not valid JSON: {exc}") from exc
    # Stashed for error messages so a bad value points back at its file.
    config["_source"] = str(path)
    return validate_threshold_config(config)


def equipment_config(config, equipment):
    """Per-equipment block, or a ValueError naming what the config does define."""
    equipment_map = config["equipment"]
    if equipment not in equipment_map:
        raise ValueError(
            f"equipment {equipment!r} is not defined in {config['_source']}; "
            f"available: {', '.join(sorted(equipment_map))}"
        )
    return equipment_map[equipment]


# --- Rule path: public entry points ---

def estimate_rule_signal(raw_0_255, equipment, config):
    """Compute the primary rule-layer signal for the given equipment type.

    Returns (signal_c, display_bits) where `signal_c` is the °C-ish value the
    rule tiers compare against, and `display_bits` is a dict of extra fields
    the decision panel can show (max pixel, percentile, mean, etc.).

    Dispatch is on the equipment's `type` field, so a new equipment entry that
    reuses an existing rule type needs no code change here.

    Real radiometric calibration (per-pixel °C from a manufacturer-specific
    camera calibration) is HER-81 territory. The mappings here are stubs
    sized to produce usable demo signals from 8-bit imagery.
    """
    eq_cfg = equipment_config(config, equipment)
    extractor = RULE_TYPES[eq_cfg["type"]]["signal"]
    return extractor(raw_0_255, eq_cfg, config, equipment)


def rule_verdict(signal_c, equipment, config):
    """Apply per-equipment rule-based thresholds. Returns (severity, explanation)."""
    eq_cfg = equipment_config(config, equipment)
    evaluator = RULE_TYPES[eq_cfg["type"]]["verdict"]
    return evaluator(signal_c, eq_cfg, config, equipment)


# --- Ensemble ---

def ensemble_decision(ml, rule):
    """Combine ML + rule verdicts.

    - If either flags (ML FAULT, or a rule severity whose action is 'flag')
      -> combined FLAGGED
    - If the rule severity's action is 'monitor' -> combined MONITOR
    - Otherwise NORMAL

    Raises ValueError on an unknown rule severity rather than treating it as
    'not flagged'.
    """
    ml_flagged = (ml == "FAULT")
    action = severity_action(rule)

    if ml_flagged and action == "flag":
        return "FLAGGED (both layers)", "red"
    if ml_flagged:
        return "FLAGGED (ML only)", "red"
    if action == "flag":
        return "FLAGGED (rules only)", "red"
    if action == "monitor":
        return "MONITOR (warning tier)", "gold"
    return "NORMAL", "green"


# --- Visualization ---

def render_panel(path, equipment, raw, recon_denorm, err, mean_err,
                 ml, ml_reason, signal_display, rule, rule_reason,
                 final, final_color, out_path, title_suffix=""):
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
    rule_color = severity_color(rule)

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

def run_one(path, equipment, expected, model, mean, std, idx, config):
    raw, normed = load_image_for_inference(path, mean, std)
    recon_normed, err, mean_err = run_ml_inference(model, normed)
    recon_denorm = recon_normed * std + mean

    ml, ml_reason = ml_verdict(mean_err)
    signal_c, signal_display = estimate_rule_signal(raw, equipment, config)
    rule, rule_reason = rule_verdict(signal_c, equipment, config)
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
                        help="Equipment type for rule-based layer. Must be a key under "
                             "'equipment' in the threshold config (default config defines "
                             "transformer, pv).")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Path to a rule-layer threshold JSON config. Defaults to "
                             f"{DEFAULT_THRESHOLD_CONFIG.name} (IEC / manufacturer values). "
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
        config = load_threshold_config(args.thresholds)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Thresholds: {config.get('name', 'unnamed')}  ({config['_source']})")

    # Every equipment type this run will touch must exist in the config.
    if args.image:
        needed = {args.equipment}
    else:
        needed = {equipment for _path, equipment, _expected in DEFAULT_SAMPLES}
    try:
        for equipment in sorted(needed):
            equipment_config(config, equipment)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

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
            return 1
        run_one(path, args.equipment, args.expected, model, mean, std, idx=1, config=config)
    else:
        print("Running curated sample set (4 images)...")
        for i, (path, equipment, expected) in enumerate(DEFAULT_SAMPLES, 1):
            if not path.exists():
                print(f"  SKIP [{i}] {path}: not found")
                continue
            run_one(path, equipment, expected, model, mean, std, idx=i, config=config)

    print(f"\nVisualizations saved to: {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
