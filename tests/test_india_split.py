"""Invariants for docs/data_manifests/india_plot_split.csv (HER-18)."""
import csv
from collections import Counter
from pathlib import Path

SPLIT = Path(__file__).resolve().parent.parent / "docs" / "data_manifests" / "india_plot_split.csv"
ROWS = list(csv.DictReader(open(SPLIT, encoding="utf-8")))


def test_counts_match_decision():
    c = Counter((r["split"], r["label"]) for r in ROWS)
    assert c == {
        ("train", "healthy"): 7790, ("train_faulty_unused", "faulty"): 659,
        ("val", "healthy"): 1743, ("val", "faulty"): 121,
        ("test", "healthy"): 2687, ("test", "faulty"): 187,
        ("cross_site", "healthy"): 97, ("cross_site", "faulty"): 40,
    }


def test_each_plot_has_one_role():
    roles = {}
    for r in ROWS:
        if r["source"] == "india":
            role = "train" if r["split"].startswith("train") else r["split"]
            assert roles.setdefault(r["plot"], role) == role
    assert roles == {"17": "train", "16": "val", "18": "test"}


def test_no_flight_in_two_splits():
    seen = {}
    for r in ROWS:
        if r["source"] == "india":
            role = "train" if r["split"].startswith("train") else r["split"]
            assert seen.setdefault(r["flight"], role) == role, r["flight"]


def test_paths_unique():
    paths = [r["rel_path"] for r in ROWS]
    assert len(paths) == len(set(paths))


def test_no_faulty_frames_in_training():
    assert not any(r["split"] == "train" and r["label"] == "faulty" for r in ROWS)
