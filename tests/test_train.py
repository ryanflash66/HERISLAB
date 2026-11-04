"""Tests for hlab.train (HER-128).

The loop runs on a 64-frame in-memory dataset on CPU for two epochs; the
holdout rules are checked against the formulas the deleted scripts used.
"""
import json
import random
from collections import defaultdict

import numpy as np
import pytest

from hlab import datasets as D
from hlab import train as T
from hlab.paths import CONFIG

torch = pytest.importorskip("torch")

H, W = 240, 320


# ---------------------------------------------------------------- holdout rules
def _legacy_random_split(n, seed, val_split=0.1):
    """train_autoencoder.py / train_autoencoder_pv.py, verbatim."""
    n_val = int(n * val_split)
    indices = np.random.default_rng(seed).permutation(n)
    return indices[n_val:], indices[:n_val]


def _legacy_flight_holdout(meta, seed=42, frac=0.10):
    """train_autoencoder_india.py:flight_holdout, verbatim."""
    by_flight = defaultdict(list)
    for i, m in enumerate(meta):
        by_flight[m["flight"]].append(i)
    flights = sorted(by_flight)
    random.Random(seed).shuffle(flights)
    target, held, chosen = frac * len(meta), [], []
    for f in flights:
        if len(held) >= target:
            break
        held += by_flight[f]; chosen.append(f)
    held_set = set(held)
    train_idx = [i for i in range(len(meta)) if i not in held_set]
    return np.array(train_idx), np.array(sorted(held)), chosen


def _split(n, meta=None):
    return D.Split("train", np.zeros((n, 2, 2), np.float32), np.zeros(n, int), [""] * n,
                   np.array([""] * n), meta or [{}] * n)


@pytest.mark.parametrize("n,seed", [(9117, 42), (8019, 43), (7052, 43)])
def test_random_frames_reproduces_the_scripts_split(n, seed):
    tr, va = T.RandomFrames(seed=seed).indices(_split(n))
    ltr, lva = _legacy_random_split(n, seed)
    assert np.array_equal(tr, ltr) and np.array_equal(va, lva)
    assert len(va) == int(n * 0.1) and len(set(tr) | set(va)) == n


def test_flights_reproduces_flight_holdout():
    rng = np.random.default_rng(5)
    meta = [{"flight": f"17.{int(f)}"} for f in rng.integers(1, 12, 500)]
    got = D.flight_holdout(meta, fraction=0.1, seed=42)
    want = _legacy_flight_holdout(meta)
    assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1]) and got[2] == want[2]
    tr, va = T.Flights().indices(_split(500, meta))
    assert np.array_equal(va, want[1]) and len(va) >= 50
    assert not ({meta[i]["flight"] for i in tr} & {meta[i]["flight"] for i in va})   # no flight on both sides


def test_flights_on_real_india_manifest():
    ds = D.Dataset("india")
    if not (ds.dir / "train_meta.csv").exists():
        pytest.skip("India metadata not present")
    meta = ds._india_meta("train")
    got = D.flight_holdout(meta)
    want = _legacy_flight_holdout(meta)
    assert np.array_equal(got[1], want[1]) and got[2] == want[2]


def test_holdout_from_config():
    assert T.holdout_from_config({"type": "random_frames", "seed": 43}) == T.RandomFrames(seed=43)
    assert T.holdout_from_config({"type": "flights", "fraction": 0.2}) == T.Flights(fraction=0.2)
    with pytest.raises(ValueError):
        T.holdout_from_config({"type": "coin_flip"})


# ---------------------------------------------------------------- the loop
def _toy_dataset(n=64, with_masks=False):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:H, 0:W]
    base = (np.sin(yy / 40.0) + np.cos(xx / 50.0)).astype(np.float32)
    x = np.stack([base * rng.uniform(0.5, 1.5) + rng.normal(0, 0.05, (H, W)) for _ in range(n)]).astype(np.float32)
    split = D.Split("train", x, np.zeros(n, int), [f"f{i}" for i in range(n)], np.array(["toy"] * n), [{}] * n)
    masks = {"train": (xx < W // 2)[None].repeat(n, axis=0)} if with_masks else None
    return D.InMemoryDataset("toy", {"train": split}, stats={"mean": 0.0, "std": 1.0}, masks=masks)


def test_fit_trains_checkpoints_and_returns_a_bundle(tmp_path):
    bundle, h = T.fit(_toy_dataset(), "toy", holdout=T.RandomFrames(0, 0.25), epochs=2, batch=16,
                      device="cpu", model_dir=tmp_path, log=tmp_path / "log.txt", seed=1)
    assert h["epochs_run"] == 2 and len(h["val_loss"]) == 2
    assert h["train_loss"][1] < h["train_loss"][0]
    assert (tmp_path / "autoencoder_toy_best.pth").exists() and (tmp_path / "autoencoder_toy_final.pth").exists()
    assert h["final_epoch"] == h["best_epoch"]                     # final="best"
    assert bundle.name == "toy" and bundle.threshold is None and bundle.weights == h["best_path"]
    assert bundle.score_normalized(_toy_dataset().split("train").x[:2]).shape == (2,)
    log = (tmp_path / "log.txt").read_text(encoding="utf-8")
    assert "Epoch   1/2" in log and "Training complete" in log


def test_fit_final_last_keeps_the_last_epoch(tmp_path):
    _, h = T.fit(_toy_dataset(), "toy", epochs=2, batch=16, device="cpu", model_dir=tmp_path, final="last", seed=1)
    assert h["final_epoch"] == 2
    _, h = T.fit(_toy_dataset(), "toy", epochs=2, batch=16, device="cpu", model_dir=tmp_path, final="best", seed=1)
    assert h["final_epoch"] == h["best_epoch"]


def test_fit_masked_mse_and_clip(tmp_path):
    _, h = T.fit(_toy_dataset(with_masks=True), "toy", loss="masked_mse", clip=1.0, lr=3e-4, epochs=1, batch=16,
                 device="cpu", model_dir=tmp_path, seed=1)
    assert h["epochs_run"] == 1 and np.isfinite(h["val_loss"][0])
    with pytest.raises(ValueError, match="panel masks"):
        T.fit(_toy_dataset(), "toy", loss="masked_mse", epochs=1, batch=16, device="cpu", model_dir=tmp_path)
    with pytest.raises(ValueError, match="unknown loss"):
        T.fit(_toy_dataset(), "toy", loss="huber", device="cpu", model_dir=tmp_path)


def test_bundle_names_keep_their_checkpoint_files(tmp_path):
    assert T.weights_paths("v2", tmp_path) == (tmp_path / "autoencoder_best.pth", tmp_path / "autoencoder_final.pth")
    assert T.weights_paths("pv_om", tmp_path)[0] == tmp_path / "autoencoder_pv_om_best.pth"
    assert T.weights_paths("toy", tmp_path)[1] == tmp_path / "autoencoder_toy_final.pth"


# ---------------------------------------------------------------- configurations
def test_training_config_entries_match_the_deleted_scripts():
    cfg = json.loads((CONFIG / "training.json").read_text(encoding="utf-8"))
    assert cfg["v2"]["holdout"] == {"type": "random_frames", "seed": 42, "fraction": 0.1}
    for name in ("pv", "pv_om"):
        assert cfg[name]["holdout"]["seed"] == 43 and cfg[name]["dataset"] == name
    for name in ("v2", "pv", "pv_om"):
        e = cfg[name]
        assert (e["lr"], e["clip"], e["seed"], e["epochs"], e["batch"], e["patience"], e["final"]) == \
               (0.001, None, None, 50, 32, 10, "best")
    assert cfg["india"]["holdout"]["type"] == "flights" and cfg["india"]["clip"] == 1.0 and cfg["india"]["final"] == "last"
    assert T.load_training_config("pv")["log"] == "training_log_pv.txt"
    with pytest.raises(ValueError, match="no training config"):
        T.load_training_config("mars")


def test_fit_config_runs_an_entry_from_a_custom_file(tmp_path, monkeypatch):
    entries = {"toy": {"dataset": "pv", "holdout": {"type": "random_frames", "seed": 0, "fraction": 0.25},
                       "epochs": 1, "batch": 16, "log": "toy.txt"}}
    (tmp_path / "t.json").write_text(json.dumps(entries), encoding="utf-8")
    monkeypatch.setattr(T, "Dataset", lambda name: _toy_dataset())
    bundle, h = T.fit_config("toy", path=tmp_path / "t.json", device="cpu", model_dir=tmp_path, log_dir=tmp_path)
    assert h["epochs_run"] == 1 and (tmp_path / "toy.txt").exists() and bundle.name == "toy"
