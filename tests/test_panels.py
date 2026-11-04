"""Tests for hlab.panels and hlab.rjpg (HER-129): hand-built arrays on CPU, a synthetic R-JPG, no data."""
import json
import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hlab import panels as P  # noqa: E402
from hlab import rjpg  # noqa: E402

Fn = torch.nn.functional


# ---------------------------------------------------------------- labelme boxes
def _labelme(tmp_path, shapes):
    p = tmp_path / "frame.json"
    p.write_text(json.dumps({"shapes": [{"points": s} for s in shapes]}), encoding="utf-8")
    return p


def test_box_mask_scales_640x512_boxes_to_240x320(tmp_path):
    jp = _labelme(tmp_path, [[(30, 60), (10, 20)], [(600, 500), (620, 510)]])
    assert P.boxes(jp) == [(10, 20, 30, 60), (600, 500, 620, 510)]
    m = P.box_mask(jp, 0)
    assert m.shape == (240, 320) and m.dtype == bool
    assert m[9:29, 5:16].all() and not m[:9].any() and not m[:, :5].any() and not m[29:200, :].any()
    assert m[234:240, 300:311].all()
    p2 = P.box_mask(jp, 2)
    assert p2[7:31, 3:18].all() and p2.sum() > m.sum()
    assert P.largest_box(jp) == (10, 20, 30, 60)
    assert P.scale_point(511, 639) == (239, 319) and P.scale_point(256, 320) == (120, 160)
    assert P.box_mask(jp, 0, shape=(512, 640), src_shape=(512, 640))[20:61, 10:31].all()


# ---------------------------------------------------------------- roughness mask
def test_panel_mask_keeps_the_smooth_half():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:96, 0:128]
    frame = 0.2 * np.sin(yy / 9.0) + rng.normal(0, 0.01, (96, 128))        # smooth left half
    frame[:, 64:] = rng.normal(0, 1.0, (96, 64))                           # rough right half
    b = torch.from_numpy(frame.astype(np.float32))[None, None]
    m = P.panel_mask(b)
    assert m.shape == (1, 1, 96, 128) and m.dtype == torch.bool
    assert m[0, 0, :, :50].float().mean() > 0.9 and m[0, 0, :, 78:].float().mean() < 0.1
    assert P.panel_mask(b, version=1).shape == m.shape and P.MASK_VERSION == 2


# ---------------------------------------------------------------- ground blanking
def test_blank_ground_sets_ground_to_the_panel_median():
    x = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    mask = np.zeros((2, 4, 4), bool); mask[0, :, :2] = True                 # frame 1 has no panel
    out = P.blank_ground(x, mask)
    assert np.array_equal(out[0][:, :2], x[0][:, :2]) and (out[0][:, 2:] == np.median(x[0][:, :2])).all()
    assert np.array_equal(out[1], x[1]) and np.array_equal(x, np.arange(32, dtype=np.float32).reshape(2, 4, 4))


# ---------------------------------------------------------------- peak statistics
def _peak_frame():
    v = torch.zeros(1, 1, 8, 8); v[0, 0, 3:6, 3:6] = 9.0                    # a 3x3 plateau of 9
    return v


def test_masked_peak_whole_frame_and_masked_agree_with_the_formulas():
    v = _peak_frame()
    mx, md, sd = P.masked_peak(v, None, 3)
    assert mx.item() == pytest.approx(9.0) and md.item() == 0.0 and sd.item() == pytest.approx(v.flatten().std().item())
    full = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    mx2, md2, sd2 = P.masked_peak(v, full, 3)
    assert (mx2.item(), md2.item(), sd2.item()) == (mx.item(), md.item(), sd.item())
    hole = full.clone(); hole[0, 0, 2:7, 2:7] = False                     # mask out the plateau and its rim
    mx3, md3, sd3 = P.masked_peak(v, hole, 3)
    assert mx3.item() == 0.0                                              # max of the 3x3 average outside the rim
    assert md3.item() == 0.0 and sd3.item() == pytest.approx(sd.item())    # 39 px kept (<= 50): stats fall back to the frame
    small = torch.zeros(1, 1, 8, 8, dtype=torch.bool); small[0, 0, 4, 4] = True
    mx4, md4, sd4 = P.masked_peak(v, small, 3)
    assert mx4.item() == pytest.approx(9.0) and md4.item() == 0.0 and sd4.item() == pytest.approx(sd.item())
    empty = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    assert P.masked_peak(v, empty, 3)[0].item() == pytest.approx(9.0)      # empty mask: whole-frame max
    assert P.peak_ratio(v, None).item() == pytest.approx(9.0 / 1e-12)
    assert P.peak_standout(v, None).item() == pytest.approx(9.0 / sd.item())
    assert P.masked_peak(v, None, 5)[0].item() == pytest.approx(9.0 * 9 / 25)


def test_peak_scores_has_the_four_scores():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (5, 16, 16)).astype(np.float32)
    s = P.peak_scores(lambda b: torch.zeros_like(b), x, torch.device("cpu"), batch=2)
    assert set(s) == {"peak3", "peak5", "peakmax", "input_peak3 (no model)"}
    assert all(v.shape == (5,) and np.isfinite(v).all() for v in s.values())
    e = x ** 2
    assert s["peakmax"][0] == pytest.approx(e[0].max() / e[0].mean(), rel=1e-5)


# ---------------------------------------------------------------- local excess
def test_local_excess_and_hottest_spot():
    t = torch.full((1, 1, 9, 9), 20.0); t[0, 0, 4, 4] = 30.0
    m = torch.ones(1, 1, 9, 9); m[0, 0, :, 0] = 0                           # column 0 is ground
    ref, cov = P.local_mean(t, m, 3)
    assert cov[0, 0, 4, 4].item() == pytest.approx(1.0) and cov[0, 0, 4, 1].item() == pytest.approx(2 / 3)
    assert ref[0, 0, 4, 4].item() == pytest.approx((8 * 20 + 30) / 9)
    ex, _ = P.local_excess(t, m, 3, smooth=0)
    assert ex[0, 0, 4, 4].item() == pytest.approx(30 - (8 * 20 + 30) / 9)
    assert ex[0, 0, 0, 8].item() == pytest.approx(0.0)
    ex3, _ = P.local_excess(t, m, 3, smooth=3)
    assert ex3[0, 0, 4, 4].item() == pytest.approx((8 * 20 + 30) / 9 - (8 * 20 + 30) / 9)
    t[0, 0, 3:6, 3:6] = 30.0                                                # a 3x3 spot survives the smoothing
    r, c, value, dist = P.hottest_spot(t, m, 7, 0.3)
    assert (r, c) == (4, 4) and value == pytest.approx(30 - (9 * 30 + 40 * 20) / 49) and dist == pytest.approx(4.0)
    assert P.hottest_spot(t, torch.ones(1, 1, 9, 9), 7, 0.3)[3] == 999.0


# ---------------------------------------------------------------- DINO building blocks (no net)
def test_kmeans2_and_panel_of_on_synthetic_features():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.05, (300, P.FEAT_DIM)) + np.r_[np.ones(10), np.zeros(P.FEAT_DIM - 10)]
    b = rng.normal(0, 0.05, (200, P.FEAT_DIM)) - np.r_[np.ones(10), np.zeros(P.FEAT_DIM - 10)]
    z = torch.from_numpy(np.vstack([a, b]).astype(np.float32))
    cent = P.kmeans2(z, seed=0)
    assert cent.shape == (2, P.FEAT_DIM) and torch.equal(cent, P.kmeans2(z, seed=0))
    assign = torch.cdist(z, cent).argmin(1)
    assert assign[:300].eq(assign[0]).all() and assign[300:].eq(assign[300]).all() and assign[0] != assign[300]
    f = z[assign == assign[0]][:1].repeat(P.GRID[0] * P.GRID[1], 1)[None]      # one frame, all panel patches
    f[0, 17 * P.GRID[1] + 20] = z[300]                                           # one ground patch inside
    f[0, :P.GRID[1] * 10] = z[300]                                               # ten ground rows on top
    m = P.panel_of(f, cent, int(assign[0]))
    assert m.shape == (1, *P.GRID) and m[0, 17, 20] and not m[0, 5, 5] and m[0, 20:].all()
    bank = z[:300]
    q = torch.stack([z[0], z[300]])[None]
    s = P.patch_anomaly(q, bank)
    assert s.shape == (1, 2) and s[0, 0] < 0.01 < s[0, 1]


def test_to01_undoes_the_zscore():
    x = np.full((2, 24, 32), (128.0 - 100.0) / 50.0, np.float32)
    g = P.to01(x, {"mean": 100.0, "std": 50.0}, torch.device("cpu"))
    assert g.shape == (2, 1, *P.SIZE) and g.min().item() == pytest.approx(128 / 255, abs=1e-5)


# ---------------------------------------------------------------- hidden blocks
def test_hidden_block_predict_hides_every_block_exactly_once():
    b = torch.arange(2 * 32 * 48, dtype=torch.float32).view(2, 1, 32, 48)
    ones = P.hidden_block_predict(lambda t: torch.ones_like(t), b, block=8)
    assert torch.equal(ones, torch.ones_like(b))                 # every pixel predicted once
    seen = P.hidden_block_predict(lambda t: t, b, block=8)
    assert torch.equal(seen, torch.zeros_like(b))                # a hidden block never sees itself
    grid = torch.zeros(1, 4, 6, dtype=torch.bool); grid[0, 1, 2] = True
    hid = P.block_mask(grid, 8)
    assert hid.shape == (1, 1, 32, 48) and hid.sum() == 64 and hid[0, 0, 8:16, 16:24].all()


def test_error_maps_smooth_and_raw():
    x = np.zeros((3, 16, 16), np.float32); x[:, 8, 8] = 2.0
    raw = P.error_maps(lambda b: torch.zeros_like(b), x, torch.device("cpu"), batch=2, smooth=0)
    assert raw.shape == (3, 16, 16) and raw[0, 8, 8] == 4.0 and raw[0].sum() == 4.0
    sm = P.error_maps(lambda b: torch.zeros_like(b), x, torch.device("cpu"), batch=2)
    assert sm[0, 8, 8] == pytest.approx(4.0 / 9) and sm[0, 7:10, 7:10].sum() == pytest.approx(4.0)
    inp = P.error_maps(lambda b: torch.zeros_like(b), x, torch.device("cpu"), inpaint=True, smooth=0)
    assert np.array_equal(inp, raw)


def test_resolve_device():
    assert P.resolve_device("cpu").type == "cpu" and P.resolve_device(torch.device("cpu")).type == "cpu"
    assert P.resolve_device(None).type in ("cpu", "cuda")


# ---------------------------------------------------------------- R-JPG reader on a synthetic file
def _synthetic_rjpg(path, raw, calib):
    h, w = raw.shape
    raw_rec = struct.pack("<HHH", 2, w, h) + bytes(0x20 - 6) + raw.astype("<u2").tobytes()
    cam = bytearray(0x310)
    cam[0:2] = struct.pack("<H", 2)
    for off, key in ((0x20, "Emissivity"), (0x24, "ObjectDistance"), (0x28, "ReflectedTempK"),
                     (0x2C, "AtmosphericTempK"), (0x58, "PlanckR1"), (0x5C, "PlanckB"), (0x60, "PlanckF"),
                     (0x30C, "PlanckR2")):
        cam[off:off + 4] = struct.pack("<f", calib[key])
    cam[0x308:0x30C] = struct.pack("<i", calib["PlanckO"])
    dir_off = 0x40
    records = [(rjpg.REC_RAW, raw_rec), (rjpg.REC_CAMINFO, bytes(cam))]
    body_off = dir_off + 32 * len(records)
    fff = bytearray(b"FFF\x00" + bytes(0x18 - 4) + struct.pack(">II", dir_off, len(records)) + bytes(dir_off - 0x20))
    body = b""
    for rtype, rec in records:
        fff += struct.pack(">HHIIII", rtype, 0, 0, 0, body_off + len(body), len(rec)) + bytes(12)
        body += rec
    fff += body
    half = len(fff) // 2
    parts = [(1, fff[half:]), (0, fff[:half])]                  # out of order on purpose
    jpg = b"\xff\xd8"
    for idx, payload in parts:
        seg = b"FLIR\x00\x01" + bytes([idx, 0]) + bytes(payload)
        jpg += b"\xff\xe1" + struct.pack(">H", len(seg) + 2) + seg
    jpg += b"\xff\xda" + b"\x00\x02"
    path.write_bytes(jpg)


def test_read_rjpg_reassembles_segments_and_converts_to_celsius(tmp_path):
    rng = np.random.default_rng(0)
    raw = rng.integers(9000, 14000, (6, 8), dtype=np.uint16)
    calib = {"Emissivity": 1.0, "ObjectDistance": 0.0, "ReflectedTempK": 293.15, "AtmosphericTempK": 293.15,
             "PlanckR1": 16556.0, "PlanckB": 1428.0, "PlanckF": 1.0, "PlanckO": -342, "PlanckR2": 0.0122}
    p = tmp_path / "f.jpg"
    _synthetic_rjpg(p, raw, calib)
    got, c = rjpg.read_rjpg(p)
    assert np.array_equal(got, raw) and got.dtype == np.uint16
    assert c["PlanckO"] == -342 and c["PlanckR1"] == pytest.approx(16556.0) and c["PlanckR2"] == pytest.approx(0.0122)
    t = rjpg.raw_to_celsius(got, c)
    assert t.dtype == np.float32 and t.shape == raw.shape
    assert (np.diff(t.ravel()[np.argsort(raw.ravel())]) >= 0).all()         # hotter counts, hotter degrees
    s = 12000.0
    expect = c["PlanckB"] / np.log(c["PlanckR1"] / (c["PlanckR2"] * (s + c["PlanckO"])) + c["PlanckF"]) - 273.15
    assert rjpg.raw_to_celsius(np.array([[12000]], np.uint16), c)[0, 0] == pytest.approx(expect, abs=1e-3)
    with pytest.raises(ValueError, match="no FLIR"):
        q = tmp_path / "plain.jpg"; q.write_bytes(b"\xff\xd8\xff\xda\x00\x02"); rjpg.read_rjpg(q)
