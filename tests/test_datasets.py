"""Tests for hlab.datasets and hlab.paths (HER-127).

Fixture datasets are written to tmp_path in the layout the preprocess scripts
produce, so label parsing, manifests, sources and mask alignment are tested
without the real data. Golden tests on the real arrays skip when absent.
"""
import csv

import numpy as np
import pytest

from hlab import datasets as D
from hlab import paths

H, W = 240, 320


# ---------------------------------------------------------------- paths
def test_paths_point_at_this_checkout():
    assert (paths.ROOT / "src" / "hlab" / "paths.py").exists()
    assert (paths.CONFIG / "thresholds_iec.json").exists()
    assert paths.PREPROCESSED == paths.DATA / "CA_Preprocessed"
    assert paths.MANIFESTS == paths.ROOT / "docs" / "data_manifests"


# ---------------------------------------------------------------- manifests
def test_manifest_round_trip(tmp_path):
    sections = {"TRAIN": ["a/x_000.tiff", "a/x_001.tiff"], "HOLDOUT": ["a/x_002.tiff"], "FAULT": []}
    D.write_manifest(tmp_path / "m.txt", sections, header=["seed=42", "2 train / 1 holdout"])
    text = (tmp_path / "m.txt").read_text(encoding="utf-8")
    assert text.startswith("# seed=42\n# 2 train / 1 holdout\n\n[TRAIN]\n")
    assert D.read_manifest(tmp_path / "m.txt") == sections


def test_adjacent_frame():
    assert D.adjacent_frame("d/dr_frame_000006.tiff", 1) == "d/dr_frame_000007.tiff"
    assert D.adjacent_frame("d/dr_frame_000006.tiff", -1) == "d/dr_frame_000005.tiff"
    assert D.adjacent_frame("d/dr_frame_000000.tiff", -1) is None
    assert D.adjacent_frame("d/no_number.tiff", 1) is None


# ---------------------------------------------------------------- fixtures
def _stats(d, mean, std):
    np.save(d / "norm_stats.npy", {"mean": mean, "std": std})


def _pv_fixture(root):
    d = root / "pv"
    d.mkdir()
    _stats(d, 100.0, 10.0)
    np.save(d / "train_normal.npy", np.zeros((3, H, W), np.float32))
    np.save(d / "test_normal.npy", np.full((3, H, W), 0.5, np.float32))
    np.save(d / "test_fault.npy", np.full((2, H, W), 2.0, np.float32))
    D.write_manifest(d / "split_manifest.txt", {
        "TRAIN": ["data/CA_Training_Data/train/normal/pv_om_inspection/dr_frame_000000.tiff",
                  "data/CA_Training_Data/train/normal/pv_om_inspection/dr_frame_000001.tiff",
                  "data/CA_Training_Data/train/normal/pv_thermal_inspection/THM_00078_00070.tif"],
        "HOLDOUT": ["data/CA_Training_Data/train/normal/pv_om_inspection/dr_frame_000002.tiff",
                    "data/CA_Training_Data/train/normal/pv_om_inspection/sr_frame_000009.tiff",
                    "data/CA_Training_Data/train/normal/pv_thermal_inspection/THM_00078_00071.tif"],
        "FAULT": ["data/CA_Training_Data/test/fault/pv/C1.jpeg", "data/CA_Training_Data/test/fault/pv/H2.jpeg"],
    })
    return D.Dataset("pv", root=root)


def _india_fixture(root):
    for name, mean, std, drop_last in (("india", 111.0, 48.0, False), ("india_temp", 39.0, 5.8, True)):
        d = root / name
        d.mkdir()
        _stats(d, mean, std)
        rows = [("data/sources/x/DJI_0005_R.JPG", "faulty", "18"), ("data/sources/x/DJI_0007_R.JPG", "healthy", "18"),
                ("data/sources/x/DJI_0009_R.JPG", "healthy", "17")]
        if drop_last:
            rows = rows[:2]   # india_temp dropped the frame with no FLIR data
        with open(d / "test_meta.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["rel_path", "label", "plot"])
            w.writerows(rows)
        x = np.arange(len(rows), dtype=np.float32)[:, None, None] * np.ones((H, W), np.float32)
        np.save(d / "test.npy", x if name == "india" else x + mean)   # india_temp stores degrees C
        if name == "india_temp":
            masks = np.zeros((len(rows), H, W), bool)
            masks[:, 0, 0] = True
            np.save(d / "test_panelmask.npy", masks)
    return D.Dataset("india", root=root), D.Dataset("india_temp", root=root)


def test_pv_split_sources_labels_and_sequence(tmp_path):
    ds = _pv_fixture(tmp_path)
    assert ds.splits() == ("train", "holdout", "fault", "test")
    ho = ds.split("holdout")
    assert ho.x.shape == (3, H, W) and ho.labels.tolist() == [0, 0, 0]
    assert ho.source.tolist() == ["om", "om", "thm"]
    assert [m["sequence"] for m in ho.meta] == ["dr", "sr", ""]
    assert ds.frames(ho, source="thm").tolist() == [False, False, True]
    assert ds.frames("holdout", sequence="dr").tolist() == [True, False, False]
    test = ds.split("test")
    assert test.x.shape == (5, H, W) and test.labels.tolist() == [0, 0, 0, 1, 1]
    assert test.source.tolist()[-2:] == ["pvmd", "pvmd"] and test.frame_ids[-1].endswith("H2.jpeg")
    assert ds.stats == {"mean": 100.0, "std": 10.0}
    assert [r for r in ds.manifest() if r[1] == "train"][2][2] == "thm"


def test_split_select_keeps_alignment(tmp_path):
    ds = _pv_fixture(tmp_path)
    sub = ds.split("test").select([True, False, True, False, True])
    assert len(sub) == 3 and sub.labels.tolist() == [0, 0, 1]
    assert sub.frame_ids[1].endswith("THM_00078_00071.tif") and sub.source.tolist() == ["om", "thm", "pvmd"]


def test_manifest_length_mismatch_is_loud(tmp_path):
    ds = _pv_fixture(tmp_path)
    np.save(tmp_path / "pv" / "test_normal.npy", np.zeros((4, H, W), np.float32))
    with pytest.raises(RuntimeError, match="manifest lists 3"):
        ds.split("holdout")


def test_india_labels_plots_and_celsius_normalisation(tmp_path):
    india, temp = _india_fixture(tmp_path)
    t = india.split("test")
    assert t.labels.tolist() == [1, 0, 0] and t.source.tolist() == ["18", "18", "17"]
    assert t.frame_ids[0].endswith("DJI_0005_R.JPG") and t.meta[2]["plot"] == "17"
    assert india.frames(t, source="17").tolist() == [False, False, True]
    assert np.array_equal(t.x[1], np.ones((H, W), np.float32))      # stored z-scores returned as-is
    tt = temp.split("test")
    assert tt.x.shape == (2, H, W)
    assert np.allclose(tt.x[1], 1.0 / 5.8, atol=1e-6)                 # (39 + 1 - 39) / 5.8: degrees C z-scored here


def test_india_masks_align_by_rel_path(tmp_path):
    india, temp = _india_fixture(tmp_path)
    assert temp.masks("test").shape == (2, H, W)
    m = india.masks("test")
    assert m.shape == (3, H, W)
    assert m[0, 0, 0] and m[1, 0, 0] and not m[2].any()               # third frame has no FLIR data: all-False
    with pytest.raises(ValueError):
        D.Dataset("pv", root=tmp_path).masks("holdout")


def test_unknown_names_raise(tmp_path):
    with pytest.raises(ValueError, match="unknown dataset"):
        D.Dataset("mars")
    with pytest.raises(ValueError, match="no split"):
        _pv_fixture(tmp_path).split("val")


def test_in_memory_dataset_round_trip():
    s = D.Split("test", np.zeros((2, H, W), np.float32), np.array([0, 1]), ["a.tif", "b.tif"],
                np.array(["om", "thm"]), [{"sequence": "dr"}, {"sequence": ""}])
    ds = D.InMemoryDataset("fake", {"test": s}, stats={"mean": 1.0, "std": 2.0})
    assert ds.split("test") is s and ds.splits() == ("test",) and ds.has_split("test")
    assert ds.manifest() == [("a.tif", "test", "om"), ("b.tif", "test", "thm")]
    assert ds.frames("test", source="thm").tolist() == [False, True]
    assert ds.frames("test", sequence="dr").tolist() == [True, False]
    assert ds.stats == {"mean": 1.0, "std": 2.0}
    with pytest.raises(ValueError):
        ds.split("train")


# ---------------------------------------------------------------- golden (real data)
def _real(name, split):
    ds = D.Dataset(name)
    if not ds.has_split(split):
        pytest.skip(f"{name}/{split} not present")
    return ds


def test_golden_pv_holdout():
    ds = _real("pv", "holdout")
    ho = ds.split("holdout")
    assert ho.x.shape == (892, H, W)
    assert int(ds.frames(ho, source="thm").sum()) == 84
    assert int(ds.frames(ho, sequence="dr").sum()) + int(ds.frames(ho, sequence="sr").sum()) == 808


def test_golden_v2_transformer_selector_matches_old_slices():
    ds = _real("v2", "test_normal")
    tn, tf = ds.split("test_normal"), ds.split("test_fault")
    assert np.flatnonzero(ds.frames(tn, source="transformer")).tolist() == [33, 34, 35, 36]
    sel = np.flatnonzero(ds.frames(tf, source="transformer"))
    assert sel.min() == 452 and sel.max() == 684 and len(sel) == 233


def test_golden_pvmd_unique_groups():
    if not (paths.RAW / "test" / "fault" / "pv").exists():
        pytest.skip("PVMD files not present")
    files, gid, groups = D.pvmd_unique_groups()
    assert len(files) == 1000 and len(groups) == 140 and gid.max() == 139
