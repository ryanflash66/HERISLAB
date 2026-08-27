"""Tests for the rule layer and its JSON threshold config.

The bulk of these are golden-master tests: `legacy_*` below are verbatim copies
of the hardcoded rule logic as it stood before thresholds were externalized.
Every assertion compares the config-driven implementation against them, so the
shipped IEC defaults must reproduce the old behavior exactly -- same severity,
same explanation string, character for character.

Run:
  venv/Scripts/python.exe -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # no display in CI / headless runs

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import demo_ensemble as de  # noqa: E402


# --- Legacy implementation (pre-refactor, hardcoded) ---

LEGACY_CAMERA_TEMP_MIN = 20.0
LEGACY_CAMERA_TEMP_MAX = 120.0
LEGACY_TRANSFORMER_SURFACE_OFFSET_C = 10.0
LEGACY_AMBIENT_C = 35.0
LEGACY_THRESHOLDS = {
    "transformer": {
        "type": "top_oil",
        "ceiling_c": LEGACY_AMBIENT_C + 85.0,
        "needs_surface_offset": True,
    },
    "pv": {
        "type": "hotspot_delta",
        "tiers": [
            ("normal", 0, 15, "green"),
            ("warning", 15, 40, "gold"),
            ("alarm", 40, 75, "orange"),
            ("critical", 75, 999, "red"),
        ],
        "needs_surface_offset": False,
    },
}


def legacy_estimate_rule_signal(raw_0_255, equipment):
    max_pixel = float(raw_0_255.max())
    if equipment == "transformer":
        surface_c = LEGACY_CAMERA_TEMP_MIN + (max_pixel / 255.0) * (
            LEGACY_CAMERA_TEMP_MAX - LEGACY_CAMERA_TEMP_MIN
        )
        return surface_c, {
            "label": "Surface proxy",
            "value_str": f"{surface_c:.1f}°C",
            "extra": f"Max pixel: {max_pixel:.0f} / 255",
        }
    if equipment == "pv":
        p95 = float(np.percentile(raw_0_255, 95))
        mean_pixel = float(raw_0_255.mean())
        delta_c = (p95 - mean_pixel) * (100.0 / 255.0)
        return delta_c, {
            "label": "Hot-spot ΔT proxy",
            "value_str": f"{delta_c:.1f}°C (p95 − mean scaled)",
            "extra": f"p95 pixel: {p95:.0f}, mean pixel: {mean_pixel:.0f}",
        }
    return 0.0, {"label": "Unknown", "value_str": "n/a", "extra": ""}


def legacy_rule_verdict(signal_c, equipment):
    cfg = LEGACY_THRESHOLDS[equipment]
    if equipment == "transformer":
        calibrated_c = (
            signal_c + LEGACY_TRANSFORMER_SURFACE_OFFSET_C
            if cfg["needs_surface_offset"]
            else signal_c
        )
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


def legacy_ensemble_decision(ml, rule):
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


# --- Synthetic images ---

def image_with_max(max_pixel, fill=10.0):
    """Uniform image with one hot pixel -- drives the top_oil signal."""
    arr = np.full((240, 320), fill, dtype=np.float32)
    arr[120, 160] = max_pixel
    return arr


def image_with_hotspot(fill, hot_value, hot_fraction):
    """Image with a controlled hot region -- drives the hotspot_delta signal."""
    arr = np.full((240, 320), float(fill), dtype=np.float32)
    n_hot = int(arr.size * hot_fraction)
    flat = arr.reshape(-1)
    flat[:n_hot] = float(hot_value)
    return flat.reshape(arr.shape)


TRANSFORMER_IMAGES = [image_with_max(m) for m in (0, 64, 128, 200, 240, 250, 255)]

PV_IMAGES = [
    np.full((240, 320), 100.0, dtype=np.float32),          # flat -> normal
    image_with_hotspot(80, 140, 0.02),                     # <5% hot -> negative ΔT
    image_with_hotspot(60, 200, 0.04),                     # <5% hot -> negative ΔT
    image_with_hotspot(50, 150, 0.10),                     # moderate -> warning
    np.linspace(0, 255, 240 * 320, dtype=np.float32).reshape(240, 320),  # alarm
    image_with_hotspot(20, 255, 0.06),                     # severe -> critical
    image_with_hotspot(0, 255, 0.10),                      # extreme -> critical
]


def write_config(directory, config, name="thresholds.json"):
    path = Path(directory) / name
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def default_config_dict():
    """Fresh parse of the shipped defaults, safe to mutate."""
    return json.loads(de.DEFAULT_THRESHOLD_CONFIG.read_text(encoding="utf-8"))


class TestShippedDefaults(unittest.TestCase):
    """The shipped JSON must carry the exact constants that were hardcoded."""

    @classmethod
    def setUpClass(cls):
        cls.config = de.load_threshold_config()

    def test_default_config_file_exists_and_loads(self):
        self.assertTrue(de.DEFAULT_THRESHOLD_CONFIG.exists())
        self.assertEqual(self.config["_source"], str(de.DEFAULT_THRESHOLD_CONFIG))
        self.assertIn("name", self.config)

    def test_global_constants_match_legacy(self):
        self.assertEqual(self.config["ambient_c"], LEGACY_AMBIENT_C)
        camera = self.config["camera"]
        self.assertEqual(camera["temp_min_c"], LEGACY_CAMERA_TEMP_MIN)
        self.assertEqual(camera["temp_max_c"], LEGACY_CAMERA_TEMP_MAX)
        self.assertEqual(camera["pixel_max"], 255.0)

    def test_transformer_block_matches_legacy(self):
        eq_cfg = de.equipment_config(self.config, "transformer")
        self.assertEqual(eq_cfg["type"], "top_oil")
        self.assertEqual(eq_cfg["surface_offset_c"], LEGACY_TRANSFORMER_SURFACE_OFFSET_C)
        # Legacy baked ceiling_c = AMBIENT_C + 85.0 at import time.
        self.assertEqual(
            de._resolve_ceiling_c(eq_cfg, self.config),
            LEGACY_THRESHOLDS["transformer"]["ceiling_c"],
        )

    def test_pv_block_matches_legacy(self):
        eq_cfg = de.equipment_config(self.config, "pv")
        self.assertEqual(eq_cfg["type"], "hotspot_delta")
        self.assertEqual(eq_cfg["percentile"], 95)
        self.assertEqual(eq_cfg["delta_span_c"], 100.0)
        tiers = [(t["severity"], t["min_c"], t["max_c"]) for t in eq_cfg["tiers"]]
        legacy_tiers = [(n, lo, hi) for n, lo, hi, _c in LEGACY_THRESHOLDS["pv"]["tiers"]]
        self.assertEqual(tiers, legacy_tiers)

    def test_every_tier_severity_is_registered(self):
        for eq_cfg in self.config["equipment"].values():
            for tier in eq_cfg.get("tiers", []):
                self.assertIn(tier["severity"].upper(), de.SEVERITY_REGISTRY)


class TestBehaviorMatchesLegacy(unittest.TestCase):
    """Same inputs -> same signal, same severity, same explanation text."""

    @classmethod
    def setUpClass(cls):
        cls.config = de.load_threshold_config()

    def test_transformer_signal_matches_legacy(self):
        for arr in TRANSFORMER_IMAGES:
            with self.subTest(max_pixel=float(arr.max())):
                got_c, got_display = de.estimate_rule_signal(arr, "transformer", self.config)
                want_c, want_display = legacy_estimate_rule_signal(arr, "transformer")
                self.assertAlmostEqual(got_c, want_c, places=10)
                self.assertEqual(got_display, want_display)

    def test_transformer_verdict_matches_legacy(self):
        seen = set()
        for arr in TRANSFORMER_IMAGES:
            signal_c, _ = legacy_estimate_rule_signal(arr, "transformer")
            with self.subTest(signal_c=signal_c):
                got = de.rule_verdict(signal_c, "transformer", self.config)
                want = legacy_rule_verdict(signal_c, "transformer")
                self.assertEqual(got, want)
                seen.add(got[0])
        # Guard against a vacuous pass: both branches must have been exercised.
        self.assertEqual(seen, {"PASS", "CRITICAL"})

    def test_pv_signal_matches_legacy(self):
        for i, arr in enumerate(PV_IMAGES):
            with self.subTest(image=i):
                got_c, got_display = de.estimate_rule_signal(arr, "pv", self.config)
                want_c, want_display = legacy_estimate_rule_signal(arr, "pv")
                self.assertAlmostEqual(got_c, want_c, places=10)
                self.assertEqual(got_display, want_display)

    def test_pv_verdict_matches_legacy(self):
        seen = set()
        for i, arr in enumerate(PV_IMAGES):
            signal_c, _ = legacy_estimate_rule_signal(arr, "pv")
            with self.subTest(image=i, signal_c=signal_c):
                got = de.rule_verdict(signal_c, "pv", self.config)
                want = legacy_rule_verdict(signal_c, "pv")
                self.assertEqual(got, want)
                seen.add(got[0])
        self.assertGreaterEqual(len(seen), 4, f"only exercised {seen}")
        self.assertIn("OUT_OF_RANGE", seen)
        self.assertIn("CRITICAL", seen)

    def test_small_hotspot_yields_negative_delta_and_out_of_range(self):
        """A hot spot under 5% of pixels puts p95 below the mean.

        This is why an uncovered signal is a verdict, not an exception: raising
        here would crash the demo on exactly the subtle-fault images it exists
        to catch.
        """
        arr = image_with_hotspot(80, 140, 0.02)
        signal_c, _ = de.estimate_rule_signal(arr, "pv", self.config)
        self.assertLess(signal_c, 0.0)
        self.assertEqual(
            de.rule_verdict(signal_c, "pv", self.config),
            legacy_rule_verdict(signal_c, "pv"),
        )
        self.assertEqual(de.rule_verdict(signal_c, "pv", self.config)[0], "OUT_OF_RANGE")

    def test_ensemble_matches_legacy_truth_table(self):
        for ml in ("FAULT", "NORMAL"):
            for rule in sorted(de.SEVERITY_REGISTRY):
                with self.subTest(ml=ml, rule=rule):
                    self.assertEqual(
                        de.ensemble_decision(ml, rule),
                        legacy_ensemble_decision(ml, rule),
                    )

    def test_severity_colors_match_legacy_color_map(self):
        legacy_color_map = {
            "PASS": "green", "NORMAL": "green",
            "WARNING": "gold", "ALARM": "orange",
            "CRITICAL": "red", "FAULT": "red",
        }
        for severity, color in legacy_color_map.items():
            with self.subTest(severity=severity):
                self.assertEqual(de.severity_color(severity), color)

    def test_offset_text_reads_from_config_not_a_literal(self):
        """The '+ 10°C offset' string must track surface_offset_c."""
        config = default_config_dict()
        config["equipment"]["transformer"]["surface_offset_c"] = 25.0
        with tempfile.TemporaryDirectory() as tmp:
            loaded = de.load_threshold_config(write_config(tmp, config))
            _severity, explanation = de.rule_verdict(80.0, "transformer", loaded)
        self.assertIn("+ 25°C offset", explanation)
        self.assertNotIn("+ 10°C offset", explanation)
        self.assertIn("= 105.0°C", explanation)


class TestFailLoud(unittest.TestCase):
    """Unknown vocabulary and uncovered signals must raise, not pass silently."""

    @classmethod
    def setUpClass(cls):
        cls.config = de.load_threshold_config()

    def test_unknown_severity_raises(self):
        with self.assertRaises(ValueError) as ctx:
            de.ensemble_decision("NORMAL", "ALERT")
        self.assertIn("unknown severity", str(ctx.exception))
        # Legacy silently treated this as not-flagged -- the bug being fixed.
        self.assertEqual(legacy_ensemble_decision("NORMAL", "ALERT")[0], "NORMAL")

    def test_unknown_severity_raises_even_when_ml_flags(self):
        with self.assertRaises(ValueError):
            de.ensemble_decision("FAULT", "ALERT")

    def test_severity_color_raises_on_unknown(self):
        with self.assertRaises(ValueError):
            de.severity_color("SCORCHING")

    def test_unknown_equipment_raises_and_lists_available(self):
        for call in (
            lambda: de.rule_verdict(50.0, "breaker", self.config),
            lambda: de.estimate_rule_signal(TRANSFORMER_IMAGES[0], "breaker", self.config),
        ):
            with self.subTest(call=call):
                with self.assertRaises(ValueError) as ctx:
                    call()
                message = str(ctx.exception)
                self.assertIn("breaker", message)
                self.assertIn("pv", message)
                self.assertIn("transformer", message)

    def test_signal_outside_all_tiers_is_out_of_range_not_a_crash(self):
        """Uncovered signal -> OUT_OF_RANGE, matching pre-config behavior.

        OUT_OF_RANGE is a registered severity with action 'pass', so this is a
        deliberate fail-open declared in one place rather than a fall-through.
        """
        config = default_config_dict()
        config["equipment"]["pv"]["tiers"] = [
            {"severity": "normal", "min_c": 0, "max_c": 15},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loaded = de.load_threshold_config(write_config(tmp, config))
            severity, explanation = de.rule_verdict(50.0, "pv", loaded)
        self.assertEqual(severity, "OUT_OF_RANGE")
        # Legacy's out-of-range wording, verbatim. (Not compared against
        # legacy_rule_verdict here: under the full legacy tiers 50.0 is 'alarm'
        # -- it is only uncovered by this deliberately truncated config.)
        self.assertEqual(explanation, "Hot-spot ΔT proxy 50.0°C outside all tiers")
        self.assertEqual(de.severity_action(severity), "pass")
        self.assertEqual(de.ensemble_decision("NORMAL", severity)[0], "NORMAL")

    def test_missing_config_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            de.load_threshold_config(ROOT / "config" / "does_not_exist.json")

    def test_malformed_json_raises_valueerror(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                de.load_threshold_config(path)
        self.assertIn("not valid JSON", str(ctx.exception))


class TestConfigValidation(unittest.TestCase):
    """Bad configs fail at load time with a message naming the field."""

    def load_mutated(self, mutate):
        config = default_config_dict()
        mutate(config)
        with tempfile.TemporaryDirectory() as tmp:
            return de.load_threshold_config(write_config(tmp, config))

    def assert_rejected(self, mutate, *expected_fragments):
        with self.assertRaises(ValueError) as ctx:
            self.load_mutated(mutate)
        message = str(ctx.exception)
        for fragment in expected_fragments:
            self.assertIn(fragment, message)

    def test_rejects_unknown_rule_type(self):
        def mutate(config):
            config["equipment"]["pv"]["type"] = "magic_vision"
        self.assert_rejected(mutate, "magic_vision", "known types")

    def test_rejects_both_ceiling_forms(self):
        def mutate(config):
            config["equipment"]["transformer"]["ceiling_c"] = 110.0
        self.assert_rejected(mutate, "pick one")

    def test_rejects_missing_ceiling(self):
        def mutate(config):
            del config["equipment"]["transformer"]["ceiling_over_ambient_c"]
        self.assert_rejected(mutate, "ceiling_c", "ceiling_over_ambient_c")

    def test_rejects_unknown_tier_severity(self):
        def mutate(config):
            config["equipment"]["pv"]["tiers"][1]["severity"] = "elevated"
        self.assert_rejected(mutate, "tiers[1].severity", "unknown severity")

    def test_rejects_overlapping_tiers(self):
        def mutate(config):
            config["equipment"]["pv"]["tiers"][0]["max_c"] = 30
        self.assert_rejected(mutate, "overlaps")

    def test_rejects_inverted_tier(self):
        def mutate(config):
            config["equipment"]["pv"]["tiers"][0]["max_c"] = -5
        self.assert_rejected(mutate, "must be less than max_c")

    def test_rejects_non_numeric_ambient(self):
        def mutate(config):
            config["ambient_c"] = "hot"
        self.assert_rejected(mutate, "ambient_c", "must be a number")

    def test_rejects_missing_camera_field(self):
        def mutate(config):
            del config["camera"]["pixel_max"]
        self.assert_rejected(mutate, "camera.pixel_max")

    def test_rejects_inverted_camera_range(self):
        def mutate(config):
            config["camera"]["temp_min_c"] = 200.0
        self.assert_rejected(mutate, "temp_max_c must be greater")

    def test_rejects_out_of_range_percentile(self):
        def mutate(config):
            config["equipment"]["pv"]["percentile"] = 150
        self.assert_rejected(mutate, "percentile must be within 0-100")

    def test_rejects_empty_equipment_map(self):
        def mutate(config):
            config["equipment"] = {}
        self.assert_rejected(mutate, "non-empty")


class TestScadaSwap(unittest.TestCase):
    """The point of the refactor: swap setpoints without touching code."""

    def test_absolute_ceiling_overrides_ambient_derivation(self):
        """SCADA setpoints are absolute; ceiling_c must win over ambient math."""
        config = default_config_dict()
        transformer = config["equipment"]["transformer"]
        del transformer["ceiling_over_ambient_c"]
        transformer["ceiling_c"] = 90.0
        with tempfile.TemporaryDirectory() as tmp:
            loaded = de.load_threshold_config(write_config(tmp, config))
            # surface 85 + 10 offset = 95: under the IEC 120 ceiling, over a 90 setpoint.
            severity, explanation = de.rule_verdict(85.0, "transformer", loaded)
        self.assertEqual(severity, "CRITICAL")
        self.assertIn("ceiling 90°C", explanation)
        self.assertEqual(legacy_rule_verdict(85.0, "transformer")[0], "PASS")

    def test_ambient_change_moves_the_derived_ceiling(self):
        config = default_config_dict()
        config["ambient_c"] = 10.0  # winter site: ceiling becomes 95
        with tempfile.TemporaryDirectory() as tmp:
            loaded = de.load_threshold_config(write_config(tmp, config))
            severity, explanation = de.rule_verdict(90.0, "transformer", loaded)
        self.assertEqual(severity, "CRITICAL")
        self.assertIn("ceiling 95°C", explanation)

    def test_tighter_scada_tiers_change_pv_severity(self):
        config = default_config_dict()
        config["equipment"]["pv"]["tiers"] = [
            {"severity": "normal", "min_c": 0, "max_c": 5},
            {"severity": "warning", "min_c": 5, "max_c": 10},
            {"severity": "critical", "min_c": 10, "max_c": 999},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loaded = de.load_threshold_config(write_config(tmp, config))
            severity, _ = de.rule_verdict(12.0, "pv", loaded)
        self.assertEqual(severity, "CRITICAL")
        # Same signal under IEC defaults is merely 'normal' (tier 0-15).
        self.assertEqual(legacy_rule_verdict(12.0, "pv")[0], "NORMAL")

    def test_new_equipment_type_needs_no_code_change(self):
        """A new asset reusing an existing rule type is pure config."""
        config = default_config_dict()
        config["equipment"]["breaker"] = {
            "type": "top_oil",
            "surface_offset_c": 0.0,
            "ceiling_c": 75.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            loaded = de.load_threshold_config(write_config(tmp, config))
            signal_c, display = de.estimate_rule_signal(
                image_with_max(255), "breaker", loaded
            )
            severity, explanation = de.rule_verdict(signal_c, "breaker", loaded)
        self.assertEqual(signal_c, 120.0)
        self.assertEqual(display["label"], "Surface proxy")
        self.assertEqual(severity, "CRITICAL")
        self.assertIn("ceiling 75°C", explanation)

    def test_percentile_and_span_are_configurable(self):
        config = default_config_dict()
        config["equipment"]["pv"]["percentile"] = 99
        config["equipment"]["pv"]["delta_span_c"] = 255.0  # 1:1 pixel -> degC
        with tempfile.TemporaryDirectory() as tmp:
            loaded = de.load_threshold_config(write_config(tmp, config))
            arr = image_with_hotspot(50, 200, 0.05)
            signal_c, display = de.estimate_rule_signal(arr, "pv", loaded)
        expected = float(np.percentile(arr, 99)) - float(arr.mean())
        self.assertAlmostEqual(signal_c, expected, places=10)
        self.assertIn("p99", display["extra"])


if __name__ == "__main__":
    unittest.main()
