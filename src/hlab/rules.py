"""
Rule layer: threshold config, per-equipment signal and verdict, ensemble (HER-124).

Moved out of src/demo_ensemble.py as one block. Imports numpy and json only, so
the tests run without torch or matplotlib.

Public interface:
  RuleConfig.load(path=None)                validated config; default config/thresholds_iec.json
  RuleConfig.from_dict(d, source="<dict>")  same validation, same ValueError messages
  config.source / .name / .ambient_c / .camera / .equipment(name)
  evaluate(config, image_0_255, equipment) -> RuleVerdict
      severity, action ('pass'|'monitor'|'flag'), color, signal_c, explanation,
      display (label, value_str, extra)
  combine(ml_flagged: bool, verdict) -> Decision (label, color, action)
  SEVERITY_REGISTRY, severity_action, severity_color, RULE_TYPES

Adding a rule type is one RULE_TYPES entry with three callables:
  signal(raw_0_255, eq_cfg, config, equipment) -> (signal_c, display dict)
  verdict(signal_c, eq_cfg, config, equipment) -> (severity, explanation)
  validate(name, eq_cfg)                       -> None; raises ValueError("threshold config: ...")
Every severity a verdict returns must be in SEVERITY_REGISTRY.

Internal seams kept for tests: estimate_rule_signal, rule_verdict, decide, the
per-type _signal_* / _verdict_* / _validate_*, _resolve_ceiling_c,
validate_threshold_config.

NOTE: the "temperature" signals are stubs. Pixel 0 = temp_min_c, 255 = temp_max_c
after a per-image min-max stretch, so they measure contrast, not degrees. Real
radiometric calibration is HER-81 scope.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Until hlab.paths exists this module knows where the shipped defaults live.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "thresholds_iec.json"


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
    """Placeholder linear map pixel -> "surface °C" (0 = temp_min_c, 255 = temp_max_c).

    Not a real temperature: the image was min-max stretched per image, so the
    brightest pixel maps to ~temp_max_c whatever the scene. Max pixel is the
    hottest visible region on the tank.
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
    """Hot-spot contrast proxy: (percentile - mean), scaled into the tier range.

    Measures pixel contrast, not degrees (placeholder scaling; see module NOTE).
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


class RuleConfig:
    """A validated threshold config.

    Schema (see config/thresholds_iec.json):
      ambient_c            number  -- site ambient, used by ceiling_over_ambient_c
      camera               object  -- temp_min_c, temp_max_c, pixel_max
      equipment            object  -- keyed by equipment name; each entry has a
                                      `type` from RULE_TYPES plus that type's fields

    Indexable like the dict it wraps (config["camera"], config["ambient_c"]), so
    the per-type callables read it unchanged. `source` names the file (or
    "<dict>") for error messages.
    """

    def __init__(self, data, source):
        self._data = validate_threshold_config(data)
        self.source = str(source)

    @classmethod
    def from_dict(cls, data, source="<dict>"):
        return cls(data, source)

    @classmethod
    def load(cls, path=None):
        """Read and validate a JSON config; `path=None` loads the shipped IEC defaults."""
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(f"threshold config not found: {path}")
        with path.open(encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError as exc:
                raise ValueError(f"threshold config {path} is not valid JSON: {exc}") from exc
        return cls(data, path)

    @property
    def name(self):
        return self._data.get("name", "unnamed")

    @property
    def ambient_c(self):
        return float(self._data["ambient_c"])

    @property
    def camera(self):
        return self._data["camera"]

    def equipment(self, name):
        """Per-equipment block, or a ValueError naming what the config does define."""
        equipment_map = self._data["equipment"]
        if name not in equipment_map:
            raise ValueError(
                f"equipment {name!r} is not defined in {self.source}; "
                f"available: {', '.join(sorted(equipment_map))}"
            )
        return equipment_map[name]

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data


# --- Rule path: public entry points ---

def estimate_rule_signal(raw_0_255, equipment, config):
    """Compute the primary rule-layer signal for the given equipment type.

    Returns (signal_c, display_bits) where `signal_c` is the placeholder "°C" value the
    rule tiers compare against, and `display_bits` is a dict of extra fields
    the decision panel can show (max pixel, percentile, mean, etc.).

    Dispatch is on the equipment's `type` field, so a new equipment entry that
    reuses an existing rule type needs no code change here.

    Real radiometric calibration (per-pixel °C from a manufacturer-specific
    camera calibration) is HER-81 territory. The mappings here are stubs
    sized to produce usable demo signals from 8-bit imagery.
    """
    eq_cfg = config.equipment(equipment)
    extractor = RULE_TYPES[eq_cfg["type"]]["signal"]
    return extractor(raw_0_255, eq_cfg, config, equipment)


def rule_verdict(signal_c, equipment, config):
    """Apply per-equipment rule-based thresholds. Returns (severity, explanation)."""
    eq_cfg = config.equipment(equipment)
    evaluator = RULE_TYPES[eq_cfg["type"]]["verdict"]
    return evaluator(signal_c, eq_cfg, config, equipment)


# --- Ensemble ---

@dataclass(frozen=True)
class RuleVerdict:
    """What the rule layer says about one image: severity plus how it got there."""
    severity: str
    action: str          # 'pass' | 'monitor' | 'flag', from SEVERITY_REGISTRY
    color: str           # display colour, from SEVERITY_REGISTRY
    signal_c: float
    explanation: str
    display: dict        # label, value_str, extra for the decision panel


@dataclass(frozen=True)
class Decision:
    """ML and rule layers combined."""
    label: str           # "FLAGGED (both layers)" | "FLAGGED (ML only)" | "FLAGGED (rules only)" | "MONITOR (warning tier)" | "NORMAL"
    color: str
    action: str          # 'flag' | 'monitor' | 'pass'


def decide(ml_flagged, severity):
    """Combine the ML flag with a rule severity.

    - If either flags (ML, or a rule severity whose action is 'flag')
      -> combined FLAGGED
    - If the rule severity's action is 'monitor' -> combined MONITOR
    - Otherwise NORMAL

    Raises ValueError on an unknown rule severity rather than treating it as
    'not flagged'.
    """
    ml_flagged = bool(ml_flagged)
    action = severity_action(severity)

    if ml_flagged and action == "flag":
        return Decision("FLAGGED (both layers)", "red", "flag")
    if ml_flagged:
        return Decision("FLAGGED (ML only)", "red", "flag")
    if action == "flag":
        return Decision("FLAGGED (rules only)", "red", "flag")
    if action == "monitor":
        return Decision("MONITOR (warning tier)", "gold", "monitor")
    return Decision("NORMAL", "green", "pass")


def evaluate(config, image_0_255, equipment):
    """Signal, then verdict, for one 0-255 image and an equipment name in `config`."""
    signal_c, display = estimate_rule_signal(image_0_255, equipment, config)
    severity, explanation = rule_verdict(signal_c, equipment, config)
    return RuleVerdict(severity, severity_action(severity), severity_color(severity),
                       float(signal_c), explanation, display)


def combine(ml_flagged, verdict):
    """ML flag (bool) + RuleVerdict -> Decision."""
    return decide(ml_flagged, verdict.severity)
