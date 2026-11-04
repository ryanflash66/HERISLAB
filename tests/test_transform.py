"""Tests for hlab.transform (HER-125).

Three layers: exact equivalence with the transform the preprocess scripts
used to carry (one file per source, skipped without data/), golden rows of
the stored CA_Preprocessed arrays (skipped without data/), and pure checks
on synthetic arrays.

Run:  venv\\Scripts\\python.exe -m pytest tests
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hlab import transform as T

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "CA_Training_Data"
PRE = ROOT / "data" / "CA_Preprocessed"


def _legacy_load_image_raw(filepath):
    """Verbatim body of preprocess.py:load_image_raw before HER-125 (the reference)."""
    img = Image.open(filepath)
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if arr.max() > 255:
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        else:
            arr = np.zeros_like(arr)
    img_resized = Image.fromarray(arr, mode="F")
    img_resized = img_resized.resize((320, 240), Image.LANCZOS)
    return np.array(img_resized, dtype=np.float32)


def _first_file(folder):
    files = sorted(p for p in folder.iterdir() if p.is_file()) if folder.exists() else []
    if not files:
        pytest.skip(f"no data under {folder}")
    return files[0]


# ---------------------------------------------------------------- equivalence
@pytest.mark.parametrize("rel", [
    "test/normal/electric_motor",        # 8-bit PNG (RGB, averaged)
    "test/normal/transformer",           # 8-bit BMP
    "train/normal/pv_om_inspection",     # 16-bit TIFF, O&M
    "train/normal/pv_thermal_inspection",  # 16-bit TIFF, THM
    "test/fault/pv",                     # 8-bit JPEG (RGB palette)
])
def test_matches_legacy_transform_exactly(rel):
    path = _first_file(RAW / rel)
    new = T.to_model_input(T.load_gray(path))
    assert new.dtype == np.float32 and new.shape == (240, 320)
    assert np.array_equal(new, _legacy_load_image_raw(path))


# ---------------------------------------------------------------- golden rows
def _manifest_section(path, section):
    out, sec = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("["):
            sec = line
        elif line and not line.startswith("#") and sec == section:
            out.append(line)
    return out


def test_golden_v2_test_normal_rows():
    arr_path, stats_path = PRE / "test_normal.npy", PRE / "norm_stats.npy"
    folder = RAW / "test" / "normal" / "electric_motor"   # first TEST_NORMAL_FOLDERS entry in preprocess.py
    if not (arr_path.exists() and stats_path.exists() and folder.exists()):
        pytest.skip("CA_Preprocessed arrays or raw data not present")
    stored = np.load(arr_path, mmap_mode="r")
    stats = np.load(stats_path, allow_pickle=True).item()
    files = sorted(p for p in folder.iterdir() if p.is_file())[:3]
    for i, p in enumerate(files):
        row = T.normalize(T.to_model_input(T.load_gray(p)), stats)
        assert np.allclose(row, stored[i], atol=1e-2), p.name   # arrays were produced on another Pillow


def test_golden_pv_holdout_rows():
    arr_path, stats_path, manifest = PRE / "pv" / "test_normal.npy", PRE / "pv" / "norm_stats.npy", PRE / "pv" / "split_manifest.txt"
    if not (arr_path.exists() and stats_path.exists() and manifest.exists()):
        pytest.skip("PV preprocessed arrays not present")
    stored = np.load(arr_path, mmap_mode="r")
    stats = np.load(stats_path, allow_pickle=True).item()
    holdout = _manifest_section(manifest, "[HOLDOUT]")[:3]
    for i, rel in enumerate(holdout):
        p = ROOT / rel
        if not p.exists():
            pytest.skip(f"{rel} not present")
        row = T.normalize(T.to_model_input(T.load_gray(p)), stats)
        assert np.allclose(row, stored[i], atol=1e-2), rel


# ---------------------------------------------------------------- pure
def test_target_size_is_the_only_declaration():
    assert T.TARGET_SIZE == (320, 240)


def test_scale_by_max_stretches_16_bit_and_passes_8_bit_through():
    rng = np.random.default_rng(0)
    raw16 = rng.integers(6000, 14000, (512, 640)).astype(np.uint16)
    out = T.to_model_input(raw16)
    assert out.shape == (240, 320) and out.dtype == np.float32
    assert out.min() >= 0 and out.max() <= 255 + 1e-3
    raw8 = rng.integers(0, 256, (240, 320)).astype(np.uint8)
    assert np.array_equal(T.to_model_input(raw8), raw8.astype(np.float32))   # same size: untouched


def test_minmax_always_stretches():
    arr = np.full((240, 320), 100.0, dtype=np.float32)
    arr[0, 0] = 0.0
    arr[0, 1] = 200.0
    out = T.to_model_input(arr, "minmax")
    assert out.max() == pytest.approx(255.0) and out.min() == pytest.approx(0.0)
    assert T.to_model_input(np.zeros((240, 320)), "minmax").max() == 0.0     # flat image -> zeros


def test_celsius_clips_without_rescale():
    arr = np.array([[-5.0, 50.0], [130.0, 20.0]], dtype=np.float32)
    big = np.kron(arr, np.ones((120, 160), dtype=np.float32))          # 240 x 320 blocks
    out = T.to_model_input(big, "celsius")
    assert out.min() == pytest.approx(0.0) and out.max() == pytest.approx(120.0)
    assert out[60, 240] == pytest.approx(50.0)


def test_rejects_bad_representation_and_shape():
    with pytest.raises(ValueError):
        T.to_model_input(np.zeros((240, 320)), "kelvin")
    with pytest.raises(ValueError):
        T.to_model_input(np.zeros((240, 320, 3)))


def test_load_gray_averages_rgb(tmp_path):
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 30, 60, 90
    p = tmp_path / "rgb.png"
    Image.fromarray(rgb).save(p)
    out = T.load_gray(p)
    assert out.dtype == np.float32 and out.shape == (4, 4)
    assert np.all(out == 60.0)


def test_normalize_denormalize_round_trip():
    stats = {"mean": 171.73, "std": 45.97}
    x = np.random.default_rng(1).uniform(0, 255, (240, 320)).astype(np.float32)
    z = T.normalize(x, stats)
    assert z.dtype == np.float32
    assert abs(float(z.mean())) < 1 and np.allclose(T.denormalize(z, stats), x, atol=1e-3)
