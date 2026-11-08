"""Tests for docs/paper/audit.py (HER-130): every check on small in-memory inputs, no results/ or data/."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs" / "paper"))
import audit  # noqa: E402

from hlab.bundle import FakeBundle  # noqa: E402
from hlab.datasets import InMemoryDataset, Split  # noqa: E402
from hlab.metrics import from_groups  # noqa: E402
from hlab.rules import DEFAULT_CONFIG_PATH, RuleConfig  # noqa: E402

H, W = 240, 320


# ---------------------------------------------------------------- 1. PVMD duplicates
def test_check_pvmd_duplicates_counts_pixel_groups_and_label_mix(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (8, 8), dtype=np.uint8)
    b = rng.integers(0, 255, (8, 8), dtype=np.uint8)
    Image.fromarray(a).save(tmp_path / "A_001.png")
    Image.fromarray(a).save(tmp_path / "B_002.png")      # same pixels, another label
    Image.fromarray(b).save(tmp_path / "A_003.png")
    d = audit.check_pvmd_duplicates(fault_dir=tmp_path)
    assert d["n_files"] == 3 and d["n_unique"] == 2 and d["multi_label"] == 2
    assert d["label_mix"] == {"AB": 1, "A": 1}
    assert d["exif_nondefault"] == 0 and d["n_unique_exif"] == 2
    assert list(d["gid"]) == [0, 1, 0]                   # sorted file order: A_001, A_003, B_002
    assert audit.unique_first(d["gid"]) == [0, 1]
    assert np.allclose(audit.unique_fault_scores(np.array([1.0, 5.0, 3.0]), d["gid"]), [2.0, 5.0])


# ---------------------------------------------------------------- in-memory PV dataset
def _pv_dataset(n_ho=8, n_fault=6):
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:H, 0:W]
    base = (np.sin(yy / 30.0) + np.cos(xx / 45.0)).astype(np.float32)

    def frames(n, shift):
        return np.stack([base * rng.uniform(0.8, 1.2) + shift + rng.normal(0, 0.05, (H, W)) for _ in range(n)]).astype(np.float32)

    src = np.array(["om", "om", "om", "om", "om", "om", "thm", "thm"])[:n_ho]
    seq = ["dr", "sr", "dr", "sr", "dr", "sr", "", ""][:n_ho]
    ids = [f"data/raw/om/{s}_{i:03d}.jpg" if src[i] == "om" else f"data/raw/thm/THM_{i:03d}.tif" for i, s in enumerate(seq)]
    holdout = Split("holdout", frames(n_ho, 0.0), np.zeros(n_ho, int), ids, src, [{"sequence": s} for s in seq])
    train_ids = ["data/raw/om/dr_001.jpg", "data/raw/om/sr_010.jpg"]    # adjacent to dr_000 and dr_002 only
    train = Split("train", frames(2, 0.0), np.zeros(2, int), train_ids, np.array(["om", "om"]),
                  [{"sequence": "dr"}, {"sequence": "sr"}])
    fault = Split("fault", frames(n_fault, 1.5), np.ones(n_fault, int), [f"pvmd/F{i}.jpg" for i in range(n_fault)],
                  np.array(["pvmd"] * n_fault), [{}] * n_fault)
    return InMemoryDataset("pv", {"train": train, "holdout": holdout, "fault": fault}, stats={"mean": 0.0, "std": 1.0})


def _inputs(ds, thr=1.0):
    neg = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.4, 1.6])
    pos = np.array([1.1, 1.3, 2.0, 2.2, 0.9, 3.0])
    gid = np.array([0, 0, 1, 2, 2, 3])
    return audit.PVInputs(ds, FakeBundle(thr, lambda x: np.zeros(len(x))), gid=gid, errors=(neg, pos))


def test_pv_inputs_builds_unique_scores_and_source_labels():
    D = _inputs(_pv_dataset())
    assert D.n_unique == 4 and np.allclose(D.fu, [1.2, 2.0, 1.55, 3.0])
    assert list(D.seq) == ["om_dr", "om_sr", "om_dr", "om_sr", "om_dr", "om_sr", "thm", "thm"]
    assert D.n_train == 2
    negF, posF_u = D.features()
    assert negF.shape == (8, 16) and posF_u.shape == (4, 16)
    assert D.features()[0] is negF                      # computed once
    with pytest.raises(RuntimeError, match="saved errors do not match"):
        audit.PVInputs(_pv_dataset(), FakeBundle(1.0, lambda x: x), gid=np.zeros(6, int), errors=(np.zeros(3), np.zeros(6)))


# ---------------------------------------------------------------- 2. temporal leakage
def test_check_temporal_leakage_counts_adjacent_train_frames():
    ds = _pv_dataset()
    assert audit.check_temporal_leakage(ds, "holdout") == (2, 8)         # dr_000 and dr_002 flank dr_001 in TRAIN
    keep = np.array([False] + [True] * 7)
    assert audit.check_temporal_leakage(ds, ds.split("holdout"), keep=keep) == (1, 7)


# ---------------------------------------------------------------- 3. source shortcut
def test_check_source_shortcut_rows_and_fallback():
    D = _inputs(_pv_dataset())
    s = audit.check_source_shortcut(D.neg, D.src, D.fu, 1.0, order=("om", "thm"))
    assert list(s["rows"]) == ["om", "thm"] and s["rows"]["om"]["n"] == 6 and s["rows"]["thm"]["flagged"] == 2
    assert s["fp_thm"] == 2 and s["fp_om"] == 0
    assert s["auroc_om_thm"] == 1.0 and s["auroc_source_def"] == "healthy O&M vs healthy THM"
    assert s["auroc_om_fault"] == 1.0 and 0 < s["auroc_thm_fault"] < 1
    keep = D.seq != "thm"
    t = audit.check_source_shortcut(D.neg[keep], D.seq[keep], D.fu, 1.0, order=("om_dr", "om_sr", "thm"))
    assert list(t["rows"]) == ["om_dr", "om_sr"] and t["auroc_thm_fault"] is None
    assert t["auroc_source_def"] == "healthy O&M sequence dr vs sr" and 0 <= t["auroc_source"] <= 1


# ---------------------------------------------------------------- 4/5. headline and baselines
def test_check_headline_separable_scores():
    h = audit.check_headline(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.8, 0.9, 1.0]), thr=0.5, seed=0)
    assert h["auroc"] == 1.0 and h["ci"] == (1.0, 1.0)
    assert (h["saved"]["tp"], h["saved"]["fp"], h["saved"]["tn"], h["saved"]["fn"]) == (3, 0, 4, 0)
    assert h["retuned"]["f1"] == 1.0 and 0.4 <= h["retuned_threshold"] <= 0.8
    assert h["trivial_f1"] == pytest.approx(2 * 3 / (7 + 3))
    r = audit.rounded(h["saved"])
    assert set(r) == {"tp", "fn", "fp", "tn", "precision", "recall", "f1", "specificity"} and r["precision"] == 1.0


def test_global_features_and_logreg():
    flat = np.full((2, H, W), 0.5, np.float32)
    F = audit.global_features(flat, {"mean": 0.0, "std": 1.0})
    assert F.shape == (2, 16) and len(audit.FEAT) == 16
    assert F[0, audit.FEAT.index("std")] == 0 and F[0, audit.FEAT.index("hist_entropy")] == 0
    assert F[0, audit.FEAT.index("mean")] == pytest.approx(0.5)
    rng = np.random.default_rng(0)
    neg, pos = rng.normal(0, 1, (40, 16)), rng.normal(3, 1, (30, 16))
    assert audit.logreg_cv_auroc(neg, pos, seed=0) > 0.95


def test_check_trivial_baselines_orders_the_statistics():
    rng = np.random.default_rng(2)
    neg, pos = rng.normal(0, 1, (30, 16)), rng.normal(0, 1, (20, 16))
    pos[:, audit.FEAT.index("p1")] += 5                  # one statistic separates the classes
    b = audit.check_trivial_baselines(neg, pos, seed=0)
    assert set(b["single"]) == set(audit.FEAT) and all(0.5 <= v <= 1 for v in b["single"].values())
    assert b["top"][0] == ("p1", 1.0) and [v for _, v in b["top"]] == sorted((v for _, v in b["top"]), reverse=True)
    assert 0 <= b["logreg"] <= 1


def _raw_pv_files(tmp_path, n_healthy=6, n_fault=5):
    """16-bit 640x512 bright healthy frames and dark 8-bit RGB 512x512 faults, as CA-PV and PVMD differ."""
    rng = np.random.default_rng(3)
    healthy, faults = [], []
    for i in range(n_healthy):
        p = tmp_path / f"h{i}.png"
        Image.fromarray(rng.integers(20000, 24000, (512, 640)).astype(np.uint16)).save(p)
        healthy.append(p)
    for i in range(n_fault):
        p = tmp_path / f"f{i}.png"
        Image.fromarray(rng.integers(0, 120, (512, 512, 3)).astype(np.uint8)).save(p)
        faults.append(p)
    return healthy, faults


def test_check_transform_baselines_per_variant(tmp_path):
    healthy, faults = _raw_pv_files(tmp_path)
    rows = audit.check_transform_baselines(healthy, faults, seed=0)
    assert [r["variant"] for r in rows] == [v[0] for v in audit.TRANSFORM_VARIANTS]
    assert all(set(audit.TRANSFORM_COLS) <= set(r) for r in rows)
    by = {r["variant"]: r for r in rows}
    assert by["as_is"]["auroc_p1"] == 1.0                          # the stretch alone separates the formats
    assert by["as_is"]["p1_median_healthy"] > by["as_is"]["p1_median_fault"]
    m = by["matched"]
    assert m["p1_median_healthy"] == m["p1_median_fault"] and m["auroc_p1"] < by["as_is"]["auroc_p1"]
    assert m["top3"].count(",") == 2 and 0 <= m["logreg"] <= 1
    table = audit.single_statistic_table(rows).splitlines()
    assert table[0] == "| statistic | " + " | ".join(by) + " |" and len(table) == 2 + len(audit.FEAT)
    assert table[2 + audit.FEAT.index("p1")].startswith("| p1 | 1.000 |")


def test_pv_frame_paths_and_stored_baselines(tmp_path):
    ds = _pv_dataset()
    gid = np.array([0, 0, 1, 2, 2, 3])
    healthy, faults = audit.pv_frame_paths(ds, gid, root=tmp_path)
    assert healthy == [tmp_path / f for f in ds.split("holdout").frame_ids]
    assert faults == [tmp_path / f"pvmd/F{i}.jpg" for i in (0, 2, 3, 5)]
    with pytest.raises(RuntimeError, match="fault files"):
        audit.pv_frame_paths(ds, gid[:3])
    b = audit.check_stored_baselines(ds, gid, seed=0)
    assert b["single"]["mean"] == 1.0                              # faults are shifted by +1.5
    no_fault = InMemoryDataset("pv", {"holdout": ds.split("holdout")})
    assert audit.check_stored_baselines(no_fault, gid) is None


# ---------------------------------------------------------------- 6. rule layer
def test_check_rule_layer_counts_tiers_and_decisions():
    config = RuleConfig.load(DEFAULT_CONFIG_PATH)
    cool = np.full((H, W), 20.0)
    hot = np.full((H, W), 250.0)
    tiers, final, flagged = audit.check_rule_layer(config, [cool, hot, cool], "transformer", [False, True, True])
    assert sum(tiers.values()) == 3 and sum(final.values()) == 3
    assert flagged == sum(v for k, v in tiers.items() if k in ("ALARM", "CRITICAL"))
    assert tiers.get("PASS", 0) == 2 and flagged == 1                   # the 250 image trips the top-oil rule
    assert final.get("FLAG", 0) == 2 and final.get("NORMAL", 0) == 1    # ML flag wins whatever the tier


# ---------------------------------------------------------------- scenarios
def test_run_scenario_print_and_summary(tmp_path, capsys):
    D = _inputs(_pv_dataset())
    S = audit.run_scenario("a_as_is", np.ones(8, bool), D, thr=1.0, seed=0)
    assert S["n_healthy"] == 8 and S["n_thm"] == 2 and S["n_om"] == 6 and S["leak_adjacent"] == 2
    assert S["fp_thm"] == 2 and S["fp_om"] == 0 and S["fp_share_thm"] == 1.0
    assert 0.5 < S["auroc_unique"] < 1.0 and S["auroc_files"] < 1.0 and S["auroc_source"] == 1.0
    assert S["fp_files"] == 2 and S["f1_saved_files"] == round(S["_files"]["saved"]["f1"], 4)
    assert S["best_single"].split()[0] in audit.FEAT and S["auroc_files_ci"].startswith("[")
    T = audit.run_scenario("b_thm_removed", D.seq != "thm", D, thr=1.0, seed=0)
    assert T["n_thm"] == 0 and T["fp_thm"] == 0 and T["auroc_source_def"] == "healthy O&M sequence dr vs sr"
    audit.print_scenario(S)
    out = capsys.readouterr().out
    assert "healthy frames kept: 8  (O&M 6, THM 2)" in out and "AUROC healthy THM vs unique fault" in out
    audit.print_scenario(T)
    assert "AUROC healthy THM vs unique fault" not in capsys.readouterr().out
    cols = ["scenario", "n_healthy", "fp_thm", "auroc_source_def"]
    md = audit.write_summary([S, T], cols, tmp_path / "s.md", tmp_path / "s.csv")
    assert md.splitlines()[2] == "| a_as_is | 8 | 2 | healthy O&M vs healthy THM |"
    assert (tmp_path / "s.csv").read_text(encoding="utf-8").splitlines()[0] == ",".join(cols)


def test_saved_errors_rejects_unknown_bundle():
    with pytest.raises(ValueError, match="no saved errors"):
        audit.saved_errors("mars")


# ---------------------------------------------------------------- HER-110 additions
def test_check_headline_threshold_free_metrics():
    h = audit.check_headline(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.8, 0.9, 1.0]), thr=0.5, seed=0)
    assert h["auprc"] == 1.0 and h["tpr_at_fpr5"] == 1.0
    assert h["prevalence"] == pytest.approx(3 / 7)
    assert h["mean_neg"] == pytest.approx(0.25) and h["mean_pos"] == pytest.approx(0.9)


def test_check_cv_threshold_is_seeded_and_bounded():
    rng = np.random.default_rng(0)
    neg, pos = rng.normal(0, 1, 60), rng.normal(2, 1, 40)
    a = audit.check_cv_threshold(neg, pos, repeats=5, seed=1)
    assert a == audit.check_cv_threshold(neg, pos, repeats=5, seed=1)
    assert 0 < a["f1_lo"] <= a["f1_mean"] <= a["f1_hi"] <= 1 and a["repeats"] == 5
    # separable scores: the search keeps the lowest winning threshold (the top TRAIN healthy score),
    # so a held-out healthy above it is a false positive; F1 stays high but need not be 1
    sep = audit.check_cv_threshold(np.arange(10) / 10, 5 + np.arange(10), repeats=3)
    assert 0.9 < sep["f1_mean"] <= 1.0


def test_check_cv_threshold_groups_keep_copies_in_one_fold():
    rng = np.random.default_rng(4)
    neg, pos = rng.normal(0, 1, 50), rng.normal(2, 1, 30)
    # one item per group draws the same folds as no groups at all
    assert audit.check_cv_threshold(neg, pos, repeats=4, seed=2) ==         audit.check_cv_threshold(neg, pos, repeats=4, seed=2, pos_groups=np.arange(30))
    gid = np.repeat(np.arange(6), 5)                     # 6 fault images, 5 copies each
    for f in audit._group_folds(np.random.default_rng(0), np.arange(30), gid, 5):
        assert set(np.flatnonzero(np.isin(gid, gid[f]))) == set(f)     # whole groups only
    g = audit.check_cv_threshold(neg, pos, repeats=4, seed=2, pos_groups=gid)
    assert 0 <= g["f1_lo"] <= g["f1_mean"] <= g["f1_hi"] <= 1
    with pytest.raises(ValueError, match="fault groups"):
        audit.check_cv_threshold(neg, pos, pos_groups=np.arange(5))


def test_source_auroc_cis_bracket_the_point_estimates():
    D = _inputs(_pv_dataset())
    cis = audit.source_auroc_cis(D.neg, D.src, D.fu, seed=0)
    src = audit.check_source_shortcut(D.neg, D.src, D.fu, 1.0)
    for key, point in (("om_thm", "auroc_om_thm"), ("om_fault", "auroc_om_fault"), ("thm_fault", "auroc_thm_fault")):
        lo, hi = cis[key]
        assert lo <= src[point] + 1e-9 and hi >= src[point] - 1e-9
    assert audit.source_auroc_cis(D.neg[:6], D.src[:6], D.fu) == {}


def test_check_temporal_leakage_within_two_frames():
    ds = _pv_dataset()
    # dr_001 in TRAIN is one frame from dr_000 / dr_002 and three from dr_004; sr_010 is far from sr_001..sr_005
    assert audit.check_temporal_leakage(ds, "holdout", within=2) == (2, 8)
    assert audit.check_temporal_leakage(ds, "holdout", within=3) == (3, 8)


def test_check_frame_correlation_neighbour_beats_random():
    ho = _pv_dataset().split("holdout")
    rng = np.random.default_rng(5)
    train_ids = ["data/raw/om/dr_001.jpg"] + [f"data/raw/om/dr_{i:03d}.jpg" for i in range(50, 56)]
    x = rng.normal(0, 1, (len(train_ids), H, W)).astype(np.float32)   # far frames: noise
    x[0] = ho.x[0] + rng.normal(0, 0.01, (H, W))                     # dr_001 almost equals dr_000
    out = audit.check_frame_correlation(ho, train_ids, x, n_per_sequence=15, seed=0)
    assert set(out) == {"dr"}                 # no sr holdout frame has a TRAIN neighbour
    r = out["dr"]
    assert r["n"] == 2 and len(r["adjacent"]) == len(r["random"]) == 2   # dr_000 and dr_002 flank dr_001
    assert min(r["adjacent"]) > 0.9 and r["median_adjacent"] > r["median_random"]
    assert out == audit.check_frame_correlation(ho, train_ids, x, n_per_sequence=15, seed=0)


def test_check_test_hashes_duplicates_labels_and_train_pool(tmp_path):
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 255, (6, 6), dtype=np.uint8) for _ in range(4)]

    def save(name, a):
        p = tmp_path / name
        Image.fromarray(a).save(p)
        return p
    healthy = [save("h0.png", imgs[0]), save("h1.png", imgs[1])]
    fault = [save("f0.png", imgs[1]), save("f1.png", imgs[2]), save("f2.png", imgs[2])]
    train = [save("t0.png", imgs[2]), save("t1.png", imgs[3])]
    r = audit.check_test_hashes({"healthy": (healthy, 0), "fault": (fault, 1)}, {"train/x": train})
    h, f = r["sets"]["healthy"], r["sets"]["fault"]
    assert (h["n_files"], h["n_unique"], h["opposite_label"], h["in_train"]) == (2, 2, 1, 0)
    assert (f["n_files"], f["n_unique"], f["opposite_label"], f["in_train"]) == (3, 2, 1, 2)
    assert h["shared"] == {"fault": 1} and f["in_train_by_pool"] == {"train/x": 2}
    assert r["total"] == {"n_files": 5, "n_unique": 3, "in_train": 2}


def test_rule_signals_match_the_verdicts():
    config = RuleConfig.load(DEFAULT_CONFIG_PATH)
    flat = np.full((H, W), 120.0)
    spot = flat.copy(); spot[:100, :200] = 250.0     # 26% of pixels, so p95 lands in the spot
    s = audit.rule_signals(config, [flat, spot], "pv")
    assert s.shape == (2,) and s[0] == pytest.approx(0.0) and s[1] > s[0]


def test_read_training_log(tmp_path):
    p = tmp_path / "log.txt"
    p.write_text("  Train: 70 images\n  Val:   8 images\n  Parameters: 2,352,353\n"
                 "Epoch   1/50 -- train_loss: 0.5  val_loss: 0.30  lr: 1e-3\n"
                 "Epoch   2/50 -- train_loss: 0.2  val_loss: 0.10  lr: 1e-3\n"
                 "Epoch   3/50 -- train_loss: 0.9  val_loss: 1.20  lr: 1e-3\n"
                 "Early stopping at epoch 3 (no improvement)\nTraining complete in 1.5 min\n", encoding="utf-8")
    log = audit.read_training_log(p)
    assert log == dict(n_train=70, n_val=8, parameters=2352353, best_epoch=2, best_val_loss=0.10, spike_epoch=3,
                       spike_val_loss=1.20, stop_epoch=3, minutes=1.5)


def test_figures_are_written(tmp_path):
    rng = np.random.default_rng(0)
    om, thm, fu = rng.lognormal(-4.5, 0.4, 50), rng.lognormal(-3, 0.4, 10), rng.lognormal(-2, 0.6, 14)
    p2 = audit.fig_error_by_source(om, thm, fu, 0.03, tmp_path / "f2.png")
    s, y = from_groups(np.r_[om, thm], fu)
    p3 = audit.fig_roc_vs_baselines([("AE", s, y), ("p1", -s, y), ("LR", s ** 2, y)], tmp_path / "f3.png", title="t")
    assert p2.stat().st_size > 0 and p3.stat().st_size > 0


def test_image_formats_counts_mode_and_size(tmp_path):
    Image.fromarray(np.zeros((4, 6), np.uint16)).save(tmp_path / "a.png")
    Image.fromarray(np.zeros((4, 6, 3), np.uint8)).save(tmp_path / "b.png")
    Image.fromarray(np.zeros((4, 6, 3), np.uint8)).save(tmp_path / "c.png")
    f = audit.image_formats([tmp_path / n for n in ("a.png", "b.png", "c.png")])
    assert f == {("I;16", "6x4"): 1, ("RGB", "6x4"): 2}


# ---------------------------------------------------------------- India audit (HER-131)
def test_logreg_refactor_keeps_cv_and_group_cv_holds_out_whole_groups():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 3)); y = (X[:, 0] + 0.3 * rng.normal(size=60) > 0).astype(int)
    groups = np.repeat(np.arange(12), 5)
    s = audit.logreg_group_cv_scores(X, y, groups, k=4, iters=300)
    assert s.shape == (60,) and audit.auroc(s, y) > 0.8
    fit = audit.logreg_fit(X, y, iters=300)
    assert np.allclose(audit.logreg_predict(fit, X[:2]), audit.logreg_predict(fit, X)[:2])
    lo, hi, used = audit.auroc_cluster_ci(s, y, groups, n_boot=200)
    assert lo <= audit.auroc(s, y) <= hi and used > 150


def test_box_spread_and_controls():
    t = np.full((512, 640), 30.0, np.float32); t[10, 20] = 35.0
    mm, mn, px = audit.box_spread(t, (19.6, 9.4, 21.2, 11.0))       # rows 9-11, cols 20-21
    assert (mm, mn, px) == (5.0, 5.0, 6)
    assert audit.box_pixels((630.0, 500.0, 700.0, 600.0), t.shape) == (500, 512, 630, 640)
    fb = [("f1", "16.1", "16", (0, 0, 1, 1)), ("f2", "16.9", "16", (0, 0, 1, 1))]
    healthy = [{"rel_path": "h1", "flight": "16.1", "plot": "16"}, {"rel_path": "h2", "flight": "16.2", "plot": "16"}]
    ctrl, fallback = audit.control_boxes(fb, healthy)
    assert fallback == 1 and ctrl["h1"][0][0] == 0 and sum(len(v) for v in ctrl.values()) == 2
    sp = audit.check_box_spread({"max_minus_median": [3, 4, 5], "max_minus_min": [4, 5, 6]},
                                {"max_minus_median": [0.5, 1, 1.5], "max_minus_min": [1, 2, 3]})
    assert sp["max_minus_median"]["auroc"] == 1.0 and sp["max_minus_median"]["control"]["frac_under"] == 1.0


def test_shot_order_and_fault_duplicates():
    names = ["DJI_0003_R.JPG", "DJI_0001_R.JPG", "DJI_0001_R (2).JPG"]
    taken = {"DJI_0001_R.JPG": "2019:02:16 10:00:00", "DJI_0003_R.JPG": "2019:02:16 10:00:02",
             "DJI_0001_R (2).JPG": "2019:02:16 11:00:00"}
    assert sorted(names, key=lambda n: audit.shot_key(n, taken[n])) == ["DJI_0001_R.JPG", "DJI_0003_R.JPG",
                                                                       "DJI_0001_R (2).JPG"]
    assert audit.consecutive_groups([5, 1, 2, 8], gap=1) == [[1, 2], [5], [8]]
    assert audit.consecutive_groups([5, 1, 2, 8], gap=3) == [[1, 2, 5, 8]]
    order = {"a": [f"DJI_{i:04d}_R.JPG" for i in range(10)]}
    boxed = {"a": ["DJI_0001_R.JPG", "DJI_0002_R.JPG", "DJI_0006_R.JPG"]}
    split_of = {("a", n): "test" for n in boxed["a"]}
    d = audit.check_fault_duplicates(boxed, order, split_of, gaps=(1, 5))
    assert d[1]["distinct"] == 2 and d[1]["sizes"] == {1: 1, 2: 1} and d[5]["distinct"] == 1
    assert d[1]["by_split"] == {"test": (3, 2)}


def test_check_india_baselines_protocols():
    rng = np.random.default_rng(4)
    n = 240
    y = (rng.random(n) < 0.2).astype(int)
    F = rng.normal(size=(n, len(audit.FEAT))); F[:, 0] += 2 * y
    flight = np.array([f"{p}.{i % 4}" for p, i in zip(np.repeat(["16", "17", "18"], n // 3), range(n))])
    split = np.where(np.char.startswith(flight, "17"), "train", np.where(np.char.startswith(flight, "16"), "val", "test"))
    b = audit.check_india_baselines(F, y, flight, split, F[:30], y[:30])
    assert {"group_cv", "random_cv", "plot16", "plot18", "plot16_18", "cross_site", "single"} <= set(b)
    assert b["group_cv"]["auroc"] > 0.8 and "ci_flight" in b["plot16_18"] and "ci_flight" not in b["cross_site"]
    assert b["plot16"]["n_healthy"] + b["plot16"]["n_faulty"] == (split == "val").sum()
    assert max(b["single"], key=b["single"].get) == "mean"
