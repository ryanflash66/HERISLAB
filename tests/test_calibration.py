"""Tests for the validation / test halves and the frozen threshold (HER-97).

hlab.datasets: blocked_roles, group_roles, the calibration manifest and Dataset.calibration.
evaluate_autoencoder.calibrate: threshold chosen on VAL only, test half scored at it.
A golden test on the tracked manifests skips when the raw data is absent.
"""
import numpy as np
import pytest

from evaluate_autoencoder import calibrate, saved_metrics
from hlab import datasets as D
from hlab.metrics import best_f1_threshold, from_groups


def test_blocked_roles_alternate_contiguous_blocks_in_frame_order():
    order = np.array([9, 0, 5, 1, 8, 2, 7, 3, 6, 4])            # shuffled frame numbers
    roles = D.blocked_roles(order, n_blocks=5)
    by_frame = dict(zip(order, roles))
    assert [by_frame[i] for i in range(10)] == ["val", "val", "test", "test", "val", "val",
                                                "test", "test", "val", "val"]


def test_group_roles_keep_groups_together_and_balance_each_stratum():
    groups = [0, 0, 1, 2, 2, 2, 3, 4, 5, 5]
    strata = ["a", "a", "a", "a", "a", "a", "b", "b", "b", "b"]
    roles = D.group_roles(groups, strata, seed=0)
    for g in set(groups):
        assert len({r for gg, r in zip(groups, roles) if gg == g}) == 1
    for s in "ab":
        gs = {g: r for g, st, r in zip(groups, strata, roles) if st == s}
        assert sorted(gs.values()).count("val") == round(len(gs) / 2)
    assert list(D.group_roles(groups, strata, seed=0)) == list(roles)      # deterministic


def test_calibration_manifest_round_trip_and_missing_frames(tmp_path):
    rows = [dict(frame_id="a/x_001.tiff", label="healthy", stratum="om_dr", group="x_001.tiff", role="val"),
            dict(frame_id="a/x_002.tiff", label="healthy", stratum="om_dr", group="x_002.tiff", role="test")]
    D.write_calibration(tmp_path / "pv_calibration_split.csv", rows)
    assert D.read_calibration(tmp_path / "pv_calibration_split.csv")["a/x_002.tiff"]["role"] == "test"
    ds = D.Dataset("pv")
    split = D.Split("holdout", np.zeros((2, 1, 1)), np.zeros(2, int), ["a/x_002.tiff", "a/x_001.tiff"],
                    np.array(["om", "om"]))
    assert list(ds.calibration(split, manifests=tmp_path)) == ["test", "val"]
    split.frame_ids.append("a/x_003.tiff")
    with pytest.raises(RuntimeError, match="not in pv_calibration_split.csv"):
        ds.calibration(split, manifests=tmp_path)
    with pytest.raises(ValueError, match="India uses plot 16"):
        D.Dataset("india").calibration("val")


def test_calibrate_chooses_on_val_and_reports_test():
    neg = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    pos = np.array([0.9, 0.35, 0.8, 0.45, 0.85])
    neg_role = np.array(["val", "test"] * 3)
    pos_role = np.array(["val", "test", "val", "test", "val"])
    cal = calibrate(neg, pos, neg_role, pos_role)
    assert cal["threshold"] == best_f1_threshold(*from_groups(neg[neg_role == "val"], pos[pos_role == "val"]))
    assert cal["threshold"] == 0.5
    assert cal["val"]["f1"] == 1.0 and (cal["val"]["n_normal"], cal["val"]["n_fault"]) == (3, 3)
    assert (cal["test"]["tp"], cal["test"]["fp"], cal["test"]["fn"], cal["test"]["tn"]) == (0, 1, 2, 2)
    m = saved_metrics(cal)
    assert m["threshold"] == 0.5 and m["f1"] == cal["test"]["f1"] and m["auroc"] == cal["test"]["auroc"]
    assert "test_files" not in m


def test_calibrate_counts_each_unique_fault_once():
    neg = np.array([0.1, 0.2, 0.3, 0.4])
    pos = np.array([0.9, 0.9, 0.9, 0.25, 0.8, 0.45])            # group 0 has three copies
    gid = np.array([0, 0, 0, 1, 2, 3])
    neg_role = np.array(["val", "test", "val", "test"])
    pos_role = np.array(["val", "val", "val", "test", "val", "test"])
    cal = calibrate(neg, pos, neg_role, pos_role, fault_groups=gid)
    assert cal["val"]["n_fault"] == 2 and cal["test"]["n_fault"] == 2
    assert cal["test_files"]["n_fault"] == 2                       # TEST holds single-copy groups only
    assert cal["threshold"] == 0.3


def test_tracked_manifests_cover_the_test_pools():
    for name, parts in (("pv", ("holdout", "fault")), ("v2", ("test_normal", "test_fault"))):
        ds = D.Dataset(name)
        if not ds.has_split(parts[0]):
            pytest.skip(f"{name} arrays absent")
        roles = np.concatenate([ds.calibration(p) for p in parts])
        assert set(roles) == {"val", "test"}
        assert abs((roles == "val").mean() - 0.5) < 0.05
