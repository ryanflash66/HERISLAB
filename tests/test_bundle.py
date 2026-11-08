"""Tests for hlab.bundle (HER-126).

Resolution table and threshold reading need no data; scoring uses a tiny
fixed-weight model (torch required); the FakeBundle drives the demo's
decision path without weights; the golden test reproduces the stored PV
errors from the real bundle and skips when weights or data are absent.
"""
import os

import numpy as np
import pytest

from hlab import bundle as B
from hlab.paths import MODELS, PREPROCESSED, RESULTS

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------- resolution
def test_resolution_table():
    p = B.resolve("pv")
    assert p == {"weights": MODELS / "autoencoder_pv_best.pth",
                 "stats": PREPROCESSED / "pv" / "norm_stats.npy",
                 "metrics": RESULTS / "pv_eval_metrics.npy"}
    assert B.resolve("v2")["weights"] == MODELS / "autoencoder_best.pth"
    assert B.resolve("v2")["stats"] == PREPROCESSED / "norm_stats.npy"
    assert B.resolve("pv_om")["metrics"] == RESULTS / "pv_om_eval_metrics.npy"
    with pytest.raises(ValueError, match="unknown bundle"):
        B.resolve("mars")


def test_threshold_none_when_metrics_absent(tmp_path):
    assert B.read_threshold(tmp_path / "missing.npy") is None
    np.save(tmp_path / "m.npy", {"auroc": 0.9, "threshold": 0.25})
    assert B.read_threshold(tmp_path / "m.npy") == 0.25
    b = B.Bundle("t", tmp_path / "w.pth", {"mean": 0.0, "std": 1.0}, None, device="cpu",
                 metrics_path=tmp_path / "missing.npy")
    with pytest.raises(RuntimeError, match="missing.npy"):
        b.require_threshold()


# ---------------------------------------------------------------- scoring with a tiny fixed model
class _Halve(torch.nn.Module):
    def forward(self, x):
        return x / 2


def _bundle(model=None):
    return B.Bundle("tiny", None, {"mean": 10.0, "std": 2.0}, 0.5, device="cpu", model=model or _Halve())


def test_score_normalized_is_mean_squared_error_per_image():
    x = np.random.default_rng(0).normal(size=(5, 240, 320)).astype(np.float32)
    got = _bundle().score_normalized(x, batch_size=2)
    want = ((x - x / 2) ** 2).reshape(5, -1).mean(axis=1)
    assert got.shape == (5,) and np.allclose(got, want, atol=1e-6)


def test_score_normalizes_0_255_input_and_reconstruct_denormalizes():
    img = np.full((1, 240, 320), 14.0, np.float32)           # z = (14 - 10) / 2 = 2
    b = _bundle()
    assert np.allclose(b.score(img), [1.0])                     # (2 - 1)^2
    recon, err = b.reconstruct(img)
    assert recon.shape == (1, 240, 320) and np.allclose(recon, 12.0)   # z 1 -> 1 * 2 + 10
    assert np.allclose(err, 1.0)
    assert b.score_normalized(img[0]).shape == (1,)            # a single 2-D image is accepted


def test_model_loads_once_and_lazily(tmp_path):
    from hlab.model import ThermalAutoencoder
    weights = tmp_path / "w.pth"
    torch.save(ThermalAutoencoder().state_dict(), weights)
    np.save(tmp_path / "s.npy", {"mean": 0.0, "std": 1.0})
    b = B.Bundle.from_paths(weights, tmp_path / "s.npy", device="cpu")
    assert b.threshold is None and b._model is None
    m1 = b.model
    assert m1 is b.model and not m1.training
    assert b.score_normalized(np.zeros((1, 240, 320), np.float32)).shape == (1,)


# ---------------------------------------------------------------- FakeBundle drives the demo decision path
def test_fake_bundle_drives_demo_decision():
    os.environ.setdefault("MPLBACKEND", "Agg")
    demo = pytest.importorskip("demo_ensemble")
    from hlab.rules import RuleConfig
    config = RuleConfig.load()
    raw = np.full((240, 320), 100.0, np.float32)               # flat PV frame: rule tier 'normal'
    hot = B.FakeBundle(threshold=0.5, score_fn=lambda x: np.full(len(x), 1.0))
    recon, err, mean_err, ml, reason, verdict, decision = demo.decide_image(hot, raw, "pv", config)
    assert mean_err == pytest.approx(1.0) and ml == "FAULT" and "> threshold 0.500000" in reason
    assert verdict.severity == "NORMAL" and decision.label == "FLAGGED (ML only)"
    assert recon.shape == err.shape == (240, 320)
    cold = B.FakeBundle(threshold=0.5, score_fn=lambda x: np.full(len(x), 0.1))
    *_, ml, _reason, _verdict, decision = demo.decide_image(cold, raw, "pv", config)
    assert ml == "NORMAL" and decision.label == "NORMAL"
    with pytest.raises(RuntimeError):
        demo.decide_image(B.FakeBundle(None, lambda x: np.zeros(len(x))), raw, "pv", config)


# ---------------------------------------------------------------- golden
def test_golden_pv_bundle_reproduces_saved_errors():
    from hlab.datasets import Dataset
    p = B.resolve("pv")
    if not (p["weights"].exists() and p["stats"].exists() and (RESULTS / "pv_normal_errors.npy").exists()):
        pytest.skip("PV bundle or saved errors not present")
    ds = Dataset("pv")
    if not ds.has_split("holdout"):
        pytest.skip("PV holdout array not present")
    b = B.Bundle.load("pv")
    assert b.threshold == pytest.approx(0.076174, abs=1e-6)      # chosen on the validation half (HER-97)
    got = b.score_normalized(ds.split("holdout").x[:16])
    want = np.load(RESULTS / "pv_normal_errors.npy")[:16]
    assert np.allclose(got, want, atol=1e-6)
