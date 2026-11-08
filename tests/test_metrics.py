"""Tests for hlab.metrics (HER-122).

Three layers: hand-computed fixtures, golden numbers from the saved error
arrays in results/ (skipped when absent), and bootstrap determinism.

Run:  venv\\Scripts\\python.exe -m pytest tests
"""
from pathlib import Path

import numpy as np
import pytest

from hlab import metrics as M
from hlab.paths import RESULTS


# ---------------------------------------------------------------- fixtures
def test_from_groups_round_trip():
    s, y = M.from_groups([0.1, 0.2], [0.9])
    assert s.tolist() == [0.1, 0.2, 0.9]
    assert y.tolist() == [0, 0, 1]
    assert y.dtype.kind == "i"


def test_auroc_perfect_and_reversed():
    assert M.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert M.auroc([0.9, 0.1], [0, 1]) == 0.0


def test_auroc_ties_count_half():
    # pairs (neg, pos): (1,1) tie, (1,2) win, (2,1) loss, (2,2) tie -> (0.5 + 1 + 0 + 0.5) / 4
    assert M.auroc([1, 1, 2, 2], [0, 1, 0, 1]) == 0.5


def test_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 20, 300) / 10.0          # many ties
    y = rng.integers(0, 2, 300)
    pos, neg = s[y == 1], s[y == 0]
    pairwise = ((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    assert M.auroc(s, y) == pytest.approx(pairwise, abs=1e-12)


def test_auroc_rejects_bad_input():
    with pytest.raises(ValueError):
        M.auroc([0.1, 0.2], [0, 1, 1])
    with pytest.raises(ValueError):
        M.auroc([0.1, 0.2], [0, 2])
    with pytest.raises(ValueError):
        M.auroc([0.1, 0.2], [1, 1])


def test_confusion_counts_and_rates():
    s, y = [0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1]
    c = M.confusion(s, y, 0.5)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 0, 0, 2)
    assert c["precision"] == c["recall"] == c["f1"] == c["specificity"] == 1.0
    c = M.confusion(s, y, 0.3)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 0, 1)
    assert c["precision"] == pytest.approx(2 / 3)
    assert c["f1"] == pytest.approx(0.8)
    assert c["specificity"] == 0.5
    assert M.confusion(s, y, 1.0)["precision"] == 0.0  # nothing predicted: rate defined as 0


def test_best_f1_threshold_exact_search_picks_lowest_winner():
    s, y = [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]
    assert M.best_f1_threshold(s, y) == 0.2          # 0.2 and 0.8 both give F1 = 1; lowest wins
    assert M.confusion(s, y, M.best_f1_threshold(s, y))["f1"] == 1.0


def test_best_f1_threshold_quantile_fallback():
    rng = np.random.default_rng(2)
    s = np.concatenate([rng.normal(0, 1, 3000), rng.normal(2, 1, 3000)])
    y = np.r_[np.zeros(3000, int), np.ones(3000, int)]
    t = M.best_f1_threshold(s, y)
    assert s.min() <= t <= s.max()
    assert M.confusion(s, y, t)["f1"] > 0.8


def test_trivial_f1_closed_form():
    assert M.trivial_f1([0, 1]) == pytest.approx(2 * 0.5 / 1.5)
    assert M.trivial_f1([1, 1, 1]) == 1.0
    assert M.trivial_f1([0, 0]) == 0.0


# ---------------------------------------------------------------- golden numbers
GOLDEN = [
    # (healthy file, fault file, AUROC, frozen threshold, F1 at it, exact best-F1 threshold)
    ("normal_errors.npy", "fault_errors.npy", 0.902427, 0.0003388182958588004, 0.9807, 0.000336),
    ("pv_normal_errors.npy", "pv_fault_errors.npy", 0.973614, 0.027865182608366013, 0.9441, 0.028182),
]


@pytest.mark.parametrize("neg_file,pos_file,au,thr,f1,best", GOLDEN)
def test_golden_numbers(neg_file, pos_file, au, thr, f1, best):
    if not (RESULTS / neg_file).exists() or not (RESULTS / pos_file).exists():
        pytest.skip("saved error arrays not present")
    s, y = M.from_groups(np.load(RESULTS / neg_file), np.load(RESULTS / pos_file))
    assert round(M.auroc(s, y), 6) == au
    assert round(M.confusion(s, y, thr)["f1"], 4) == f1
    assert round(M.best_f1_threshold(s, y), 6) == best


# ---------------------------------------------------------------- bootstrap
def _toy():
    rng = np.random.default_rng(3)
    s = np.concatenate([rng.normal(0, 1, 60), rng.normal(1.5, 1, 40)])
    y = np.r_[np.zeros(60, int), np.ones(40, int)]
    return s, y


def test_auroc_ci_is_deterministic_per_seed():
    s, y = _toy()
    a = M.auroc_ci(s, y, n_boot=200, seed=0)
    b = M.auroc_ci(s, y, n_boot=200, seed=0)
    c = M.auroc_ci(s, y, n_boot=200, seed=1)
    assert a == b
    assert a != c
    assert a[0] <= M.auroc(s, y) <= a[1]


def test_bootstrap_cis_reproduces_auroc_ci():
    s, y = _toy()
    out = M.bootstrap_cis(s, y, {"auroc": M.auroc, "ap": M.average_precision}, n_boot=200, seed=0)
    assert out["auroc"][0] == M.auroc(s, y)
    assert out["auroc"][1:] == M.auroc_ci(s, y, n_boot=200, seed=0)
    assert out["ap"][1] <= out["ap"][0] <= out["ap"][2]


def test_paired_delta_of_self_is_zero():
    s, y = _toy()
    assert M.paired_delta_ci(s, s, y, n_boot=100) == (0.0, 0.0, 0.0)


def test_paired_delta_sign():
    s, y = _toy()
    better = s + 2.0 * y                        # pushes faults up: b separates better than a
    delta, lo, hi = M.paired_delta_ci(s, better, y, n_boot=200, seed=0)
    assert delta == pytest.approx(M.auroc(better, y) - M.auroc(s, y))
    assert delta > 0 and lo > 0


def test_roc_points_and_tpr_at_fpr():
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.35, 0.8, 0.9])
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    fpr, tpr = M.roc_points(s, y)
    assert (fpr[0], tpr[0]) == (0.0, 0.0) and (fpr[-1], tpr[-1]) == (1.0, 1.0)
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)
    assert M.tpr_at_fpr(s, y, 0.0) == pytest.approx(2 / 3)       # 0.8 and 0.9 above every healthy
    assert M.tpr_at_fpr(s, y, 0.25) == 1.0                       # 0.35 passes with one healthy (0.4) above it


def test_average_precision_hand_computed_and_chance_level():
    s = np.array([0.9, 0.8, 0.7, 0.6])
    y = np.array([1, 0, 1, 0])
    assert M.average_precision(s, y) == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))
    assert M.average_precision(np.ones(10), np.r_[np.ones(3), np.zeros(7)].astype(int)) == pytest.approx(0.3)
    assert M.average_precision([0.1, 0.9], [0, 1]) == 1.0
