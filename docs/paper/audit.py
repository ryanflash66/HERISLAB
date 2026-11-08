"""
Paper audit checks (HER-130): every check is a function of a Dataset, a Bundle and a keep-mask.

The entry points (audit_eval_validity.py, the script the paper cites; audit_her107_baselines.py,
audit_her114_reruns.py, audit_her114_retrain.py) only choose scenarios and print; the numbers come from
here, and from here only through hlab: metrics (AUROC, bootstrap CI, confusion,
threshold search), rules (tiers and decisions), datasets (splits, manifests,
PVMD pixel groups, adjacent frames) and bundle (saved thresholds, never weights).
No check retrains or re-runs a model: the reconstruction errors are the ones the
evaluators saved under results/ (saved_errors).

Checks
  check_pvmd_duplicates()                       byte- and pixel-identical PVMD files, with and without the EXIF orientation tag
  image_formats(paths)                          PIL mode and size counts (16-bit vs 8-bit, RGB vs grayscale)
  check_test_hashes(test_sets, train_pool)      duplicates, cross-label copies and training-pool copies on any test set (HER-105)
  check_temporal_leakage(dataset, split, keep)  held-out frames with a video frame within `within` steps in TRAIN
  check_frame_correlation(holdout, ids, x)      pixel correlation of held-out O&M frames with their TRAIN neighbour vs a random frame
  check_source_shortcut(neg, src, fu, thr)      errors by acquisition source, source AUROCs, false positives by source
  source_auroc_cis(neg, src, fu)                bootstrap intervals of the three source AUROCs
  check_headline(neg, pos, thr, seed)           AUROC with bootstrap CI, AUPRC, TPR at FPR <= 5%, confusion at the saved and
                                                re-tuned thresholds, trivial F1
  check_cv_threshold(neg, pos, k, repeats, pos_groups)   F1 with the threshold chosen by repeated k-fold
                                                cross-validation; pos_groups keeps duplicate faults in one fold
  check_unique_intervals(neg, fu, thr)          bootstrap intervals on the deduplicated set, unique fault image as the unit (HER-120)
  check_seed_runs(runs, gid, thr)               per-seed best epoch, val-loss spike and test metrics of repeated runs, and their spread
  check_trivial_baselines(negF, posF, seed)     single global statistics and the 16-statistic logistic regression (no model)
  check_transform_baselines(healthy, faults)    the trivial baselines per hlab.transform variant, from the raw files (HER-107)
  check_stored_baselines(dataset, gid)          the trivial baselines on a PV dataset's stored arrays (pv/, pv_matched/)
  rule_signals(config, images, equipment)       the rule layer's temperature signal per image (dT proxy for PV)
  check_rule_layer(config, images, equipment, ml_flags)   severity tiers and ensemble decisions, hlab.rules per image
  check_india_baselines(F, y, flight, split)    the 16-statistic logistic regression on the India frames, folds of whole
                                                flights, plot 17 -> plots 16/18, India -> other sites (HER-131)
  check_box_spread(fault, control)              temperature spread inside the India fault boxes vs the same boxes in
                                                healthy frames of the same flight (HER-131)
  check_fault_duplicates(boxed, order, split)   distinct faults when boxed frames in consecutive shots count once (HER-131)

Scenarios
  PVInputs(dataset, bundle)                     one PV dataset's holdout, faults, saved errors, unique-fault means, features
  run_scenario(name, keep, inputs, thr, seed)   every check above a keep-mask over the held-out healthy frames
  print_scenario(S)                             the report block the HER-114 reruns print
  write_summary(rows, cols, md, csv)            the side-by-side table
  pv_frame_paths(dataset, gid)                  raw files of a PV dataset's held-out frames and unique faults
  single_statistic_table(rows)                  every statistic's AUROC per transform variant, as Markdown
  read_training_log(path)                       the numbers the training script logged
  fig_error_by_source(...), fig_roc_vs_baselines(...)   the paper's Fig. 2 and Fig. 3 (matplotlib, imported on use)
  fig_pr_curves(curves, path)                   precision-recall curves with each set's chance level (HER-116)

Feature extraction (global_features, FEAT) and the no-model baselines stay
here: they belong to the audit, not to a detector. Seeds are arguments
(SEED = 0 is the default the paper numbers were produced with); nothing here
is configured by assigning a module global from outside.
"""
import csv
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # until hlab is installable
from hlab.datasets import PV_FAULT_FOLDER, adjacent_frame, pvmd_unique_groups  # noqa: E402
from hlab.metrics import (auroc, auroc_ci, average_precision, average_precision_ci,  # noqa: E402
                          best_f1_threshold, bootstrap_cis, confusion, from_groups, pr_points, roc_points, tpr_at_fpr,
                          trivial_f1)
from hlab.paths import RAW, RESULTS, ROOT  # noqa: E402
from hlab.rules import combine, evaluate  # noqa: E402
from hlab.transform import CELSIUS_RANGE, denormalize, load_gray, to_model_input  # noqa: E402

SEED = 0                        # bootstrap / cross-validation seed the paper numbers used

# The evaluators' saved per-image errors (healthy, fault) by bundle name.
ERROR_FILES = {
    "v2": ("normal_errors.npy", "fault_errors.npy"),
    "pv": ("pv_normal_errors.npy", "pv_fault_errors.npy"),
    "pv_om": ("pv_om_normal_errors.npy", "pv_om_fault_errors.npy"),
}


def saved_errors(bundle, results=None):
    """(healthy errors, fault errors) the evaluator saved for `bundle` (a Bundle or its name)."""
    name = bundle if isinstance(bundle, str) else bundle.name
    if name not in ERROR_FILES:
        raise ValueError(f"no saved errors known for bundle {name!r}; known: {', '.join(ERROR_FILES)}")
    neg, pos = ERROR_FILES[name]
    results = results or RESULTS
    return np.load(results / neg), np.load(results / pos)


def rounded(c):
    """A hlab.metrics confusion dict as the audit prints it: counts, then rates to 4 decimals."""
    return dict(tp=c["tp"], fn=c["fn"], fp=c["fp"], tn=c["tn"], precision=round(c["precision"], 4),
                recall=round(c["recall"], 4), f1=round(c["f1"], 4), specificity=round(c["specificity"], 4))


# ---------------------------------------------------------------- 1. PVMD duplicates
def _multi_label(groups):
    return sum(len(v) for v in groups.values() if len({x[0].upper() for x in v}) > 1)


def _label_mix(groups):
    return dict(Counter("".join(sorted({x[0].upper() for x in v})) for v in groups.values()))


def check_pvmd_duplicates(fault_dir=None):
    """Pixel-identical PVMD files and files whose pixels appear under more than one fault label.

    PVMD stores some rotation augmentations only as an EXIF orientation tag, which the
    preprocessing ignores; the count is repeated with the tag applied so both views are reported.
    Returns the group index per file (`gid`, preprocess_pv.py file order) with the counts.
    """
    fault_dir = fault_dir or RAW / "test" / "fault" / PV_FAULT_FOLDER
    files, gid, groups = pvmd_unique_groups(fault_dir=fault_dir)
    _, _, groups_t = pvmd_unique_groups(exif_transpose=True, fault_dir=fault_dir)
    exif_tag = {f: Image.open(fault_dir / f).getexif().get(274, 1) for f in files}
    byte_hash = {f: hashlib.md5((fault_dir / f).read_bytes()).hexdigest() for f in files}
    by_label = {}
    for f in files:
        by_label.setdefault(f[0].upper(), []).append(byte_hash[f])
    return dict(files=files, gid=gid, n_files=len(files), n_unique=len(groups),
                n_byte_unique=len(set(byte_hash.values())),
                byte_unique_by_label={k: (len(set(v)), len(v)) for k, v in sorted(by_label.items())},
                multi_label=_multi_label(groups), label_mix=_label_mix(groups),
                exif_nondefault=sum(1 for v in exif_tag.values() if v != 1),
                n_unique_exif=len(groups_t), multi_label_exif=_multi_label(groups_t),
                label_mix_exif=_label_mix(groups_t))


def unique_fault_scores(pos, gid):
    """One score per pixel-unique fault image: the mean over its duplicate files."""
    return np.array([pos[gid == g].mean() for g in range(gid.max() + 1)])


def unique_first(gid):
    """Index of the first file of every pixel-unique group."""
    return [int(np.flatnonzero(gid == g)[0]) for g in range(gid.max() + 1)]


# ---------------------------------------------------------------- 2. temporal leakage
def check_temporal_leakage(dataset, split="holdout", keep=None, train="train", within=1):
    """(frames with a video frame at most `within` frames away in the TRAIN manifest, frames checked).

    `split` is a split name or a Split; `keep` restricts it to a scenario's frames. within=1 is
    the immediately adjacent frame.
    """
    s = dataset.split(split) if isinstance(split, str) else split
    ids = s.frame_ids if keep is None else [f for f, k in zip(s.frame_ids, keep) if k]
    trs = {f for f, sp, _src in dataset.manifest() if sp == train}
    steps = [d for k in range(1, within + 1) for d in (-k, k)]
    adj = sum(any(adjacent_frame(p, d) in trs for d in steps) for p in ids)
    return adj, len(ids)


def _pearson(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.sqrt((a @ a) * (b @ b)) + 1e-12))


def check_frame_correlation(holdout, train_ids, train_x, n_per_sequence=15, seed=SEED):
    """How close a held-out O&M frame is to its adjacent TRAIN frame, against a random TRAIN frame.

    Definition (the Method's frame-correlation sample): per O&M sequence (dr = double-row,
    sr = single-row), `n_per_sequence` held-out frames that have an adjacent TRAIN frame are
    drawn with np.random.default_rng(seed). For each, r is the Pearson correlation over all
    pixels of the preprocessed model inputs (Split.x, z-scored; r does not change under the
    z-score) between the frame and its TRAIN neighbour (step -1 first, then +1), and between
    the frame and one TRAIN frame of the same sequence drawn uniformly at random.
    `train_x` may be a memory map; only the sampled rows are read.
    Returns {sequence: dict(n, median_adjacent, median_random, adjacent=[...], random=[...])}.
    """
    rng = np.random.default_rng(seed)
    row_of = {f: i for i, f in enumerate(train_ids)}
    train_seq = {}
    for i, f in enumerate(train_ids):
        train_seq.setdefault(Path(f).name.split("_")[0], []).append(i)
    out = {}
    for seq in sorted({m.get("sequence", "") for m in holdout.meta} - {""}):
        cands = []
        for i, (f, m) in enumerate(zip(holdout.frame_ids, holdout.meta)):
            if m.get("sequence") != seq:
                continue
            nb = next((adjacent_frame(f, d) for d in (-1, 1) if adjacent_frame(f, d) in row_of), None)
            if nb is not None:
                cands.append((i, row_of[nb]))
        if not cands:
            continue
        pick = rng.choice(len(cands), min(n_per_sequence, len(cands)), replace=False)
        adj, rnd = [], []
        for k in pick:
            i, j = cands[k]
            r_j = int(rng.choice(train_seq[seq]))
            adj.append(_pearson(holdout.x[i], train_x[j]))
            rnd.append(_pearson(holdout.x[i], train_x[r_j]))
        out[seq] = dict(n=len(adj), median_adjacent=float(np.median(adj)), median_random=float(np.median(rnd)),
                        adjacent=adj, random=rnd)
    return out


# ---------------------------------------------------------------- 3. source shortcut
def source_auroc_cis(neg, src, pos_unique, seed=SEED):
    """95% stratified bootstrap intervals (hlab.metrics.auroc_ci, 2,000 resamples) of the three source AUROCs.

    Same orientation as check_source_shortcut: O&M vs THM scores THM as the positive class;
    O&M vs fault and THM vs fault score the unique faults as positive. {} when there is no THM frame.
    """
    src = np.asarray(src)
    is_thm = src == "thm"
    om, thm = neg[~is_thm], neg[is_thm]
    if not len(thm):
        return {}
    return dict(om_thm=auroc_ci(*from_groups(om, thm), seed=seed),
                om_fault=auroc_ci(*from_groups(om, pos_unique), seed=seed),
                thm_fault=auroc_ci(*from_groups(thm, pos_unique), seed=seed))


def check_source_shortcut(neg, src, pos_unique, thr, order=("om", "thm")):
    """Healthy errors by acquisition source and the AUROCs that expose a source shortcut.

    neg: healthy errors; src: a source label per healthy frame ("thm" marks the aerial survey,
    anything else is O&M); pos_unique: one error per unique fault; thr: the saved threshold.
    rows: per source in `order` (sources with no frames are skipped): n, median error, flagged.
    auroc_source separates healthy O&M from healthy THM, or O&M sequence dr from sr when the
    scenario holds no THM frame.
    """
    src = np.asarray(src)
    rows = {}
    for s in order:
        e = neg[src == s]
        if len(e):
            rows[s] = dict(n=len(e), median=float(np.median(e)), flagged=int((e > thr).sum()))
    is_thm = src == "thm"
    om, thm = neg[~is_thm], neg[is_thm]
    flagged = neg > thr
    out = dict(rows=rows, fp_thm=int((flagged & is_thm).sum()), fp_om=int((flagged & ~is_thm).sum()),
               auroc_om_fault=auroc(*from_groups(om, pos_unique)),
               auroc_thm_fault=auroc(*from_groups(thm, pos_unique)) if len(thm) else None,
               auroc_om_thm=auroc(*from_groups(om, thm)) if len(thm) else None)
    if len(thm):
        out["auroc_source"], out["auroc_source_def"] = out["auroc_om_thm"], "healthy O&M vs healthy THM"
    else:
        out["auroc_source"] = auroc(*from_groups(neg[src == "om_dr"], neg[src == "om_sr"]))
        out["auroc_source_def"] = "healthy O&M sequence dr vs sr"
    return out


# ---------------------------------------------------------------- 4/5. headline and baselines
AUPRC_DEF = ("average precision: the sum over distinct score thresholds t (flag when score >= t; tied scores "
             "enter together) of (recall step) x precision; chance level is the fault prevalence")


def check_headline(neg, pos, thr, seed=SEED):
    """AUROC with bootstrap 95% CI, trivial flag-everything F1, confusion at the saved and re-tuned thresholds.

    Also the threshold-free AUPRC (hlab.metrics.average_precision, defined in AUPRC_DEF) with its
    bootstrap 95% CI (hlab.metrics.average_precision_ci, resampled like the AUROC CI) and the fault
    prevalence (its chance level), the TPR at FPR <= 5% (hlab.metrics.tpr_at_fpr) and the mean score per class.
    """
    s, y = from_groups(neg, pos)
    lo, hi = auroc_ci(s, y, seed=seed)
    retuned_thr = best_f1_threshold(s, y)
    return dict(auroc=auroc(s, y), ci=(lo, hi), trivial_f1=trivial_f1(y), saved=confusion(s, y, thr),
                retuned_threshold=retuned_thr, retuned=confusion(s, y, retuned_thr),
                auprc=average_precision(s, y), auprc_ci=average_precision_ci(s, y, seed=seed),
                prevalence=float(y.mean()), tpr_at_fpr5=tpr_at_fpr(s, y, 0.05),
                mean_neg=float(np.mean(neg)), mean_pos=float(np.mean(pos)))


def _group_folds(rng, idx, groups, k):
    """`idx` split into k folds of whole groups (groups permuted with `rng`). With one item per
    group this draws the same permutation as rng.permutation(idx), so ungrouped folds are unchanged."""
    ug = np.unique(groups)
    return [idx[np.isin(groups, ug[part])] for part in np.array_split(rng.permutation(len(ug)), k)]


def check_cv_threshold(neg, pos, k=5, repeats=200, seed=SEED, pos_groups=None):
    """F1 when the threshold is chosen by cross-validation instead of on the whole test set.

    Definition (the Method's cross-validated threshold): each of `repeats` repetitions splits
    the images into `k` stratified folds (healthy and faults permuted separately with
    np.random.default_rng(seed), as logreg_cv_auroc does). For each fold the threshold is
    hlab.metrics.best_f1_threshold on the other k-1 folds (the evaluator's own search) and
    flags the held-out fold at score > threshold. The k held-out folds are pooled, so every image
    is flagged once per repetition, and F1 is computed on that pooled confusion.
    pos_groups: a group index per fault (the PVMD pixel group, gid). Folds then split whole groups,
    so copies of one fault image never sit on both sides of a fold; None folds faults one by one.
    Healthy frames are always folded one by one.
    Returns dict(f1_mean, f1_lo, f1_hi (2.5 / 97.5 percentiles over repetitions), f1_mean_fold
    (the mean of per-fold F1 instead of pooled), repeats, k).
    """
    s, y = from_groups(neg, pos)
    rng = np.random.default_rng(seed)
    pooled, per_fold = [], []
    i_neg, i_pos = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    g_pos = np.arange(len(i_pos)) if pos_groups is None else np.asarray(pos_groups)
    if len(g_pos) != len(i_pos):
        raise ValueError(f"{len(g_pos)} fault groups for {len(i_pos)} faults")
    for _ in range(repeats):
        folds = [np.r_[a, b] for a, b in zip(np.array_split(rng.permutation(i_neg), k),
                                             _group_folds(rng, i_pos, g_pos, k))]
        pred = np.zeros(len(y), bool)
        for f in folds:
            tr = np.setdiff1d(np.arange(len(y)), f)
            t = best_f1_threshold(s[tr], y[tr])
            pred[f] = s[f] > t
            per_fold.append(confusion(s[f], y[f], t)["f1"])
        pooled.append(confusion(pred.astype(float), y, 0.5)["f1"])
    lo, hi = np.percentile(pooled, [2.5, 97.5])
    return dict(f1_mean=float(np.mean(pooled)), f1_lo=float(lo), f1_hi=float(hi),
                f1_mean_fold=float(np.mean(per_fold)), repeats=repeats, k=k)


FEAT = ["mean", "std", "min", "max", "p1", "p5", "p50", "p95", "p99", "p95_minus_mean",
        "frac_ge250", "frac_le5", "grad_x", "grad_y", "laplacian", "hist_entropy"]


def global_features(x, stats):
    """16 global statistics per image of a z-scored array, computed on the 0-255 values."""
    rows = []
    for i in range(len(x)):
        r = denormalize(x[i], stats)  # back to 0-255
        p = np.percentile(r, [1, 5, 50, 95, 99])
        h, _ = np.histogram(np.clip(r, 0, 255), bins=32, range=(0, 255)); h = h / h.sum()
        lap = np.abs(4 * r[1:-1, 1:-1] - r[:-2, 1:-1] - r[2:, 1:-1] - r[1:-1, :-2] - r[1:-1, 2:]).mean()
        rows.append([r.mean(), r.std(), r.min(), r.max(), *p, p[3] - r.mean(),
                     (r >= 250).mean(), (r <= 5).mean(),
                     np.abs(np.diff(r, axis=1)).mean(), np.abs(np.diff(r, axis=0)).mean(),
                     lap, -(h[h > 0] * np.log2(h[h > 0])).sum()])
    return np.array(rows)


def logreg_cv_auroc(X_neg, X_pos, k=5, l2=1.0, iters=3000, lr=0.1, seed=SEED):
    """5-fold CV logistic regression (standardized, L2), numpy only."""
    return auroc(*logreg_cv_scores(X_neg, X_pos, k=k, l2=l2, iters=iters, lr=lr, seed=seed))


def logreg_cv_scores(X_neg, X_pos, k=5, l2=1.0, iters=3000, lr=0.1, seed=SEED):
    """(out-of-fold probabilities, labels) of the logistic regression logreg_cv_auroc scores."""
    rng = np.random.default_rng(seed)
    X = np.vstack([X_neg, X_pos]); y = np.r_[np.zeros(len(X_neg)), np.ones(len(X_pos))]
    idx_n = rng.permutation(len(X_neg)); idx_p = rng.permutation(len(X_pos)) + len(X_neg)
    folds = [np.r_[a, b] for a, b in zip(np.array_split(idx_n, k), np.array_split(idx_p, k))]
    out = np.zeros(len(y))
    for f in folds:
        tr = np.setdiff1d(np.arange(len(y)), f)
        out[f] = logreg_predict(logreg_fit(X[tr], y[tr], l2=l2, iters=iters, lr=lr), X[f])
    return out, y


def logreg_fit(X, y, l2=1.0, iters=3000, lr=0.1):
    """(mu, sd, w) of the standardized L2 logistic regression every baseline here uses (gradient descent)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ w))
        g = A.T @ (p - y) / len(X) + l2 * np.r_[0, w[1:]] / len(X)
        w -= lr * g
    return mu, sd, w


def logreg_predict(fit, X):
    """Probabilities of a logreg_fit result on rows X."""
    mu, sd, w = fit
    return 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), (X - mu) / sd] @ w)))


def logreg_group_cv_scores(X, y, groups, k=5, seed=SEED, **kw):
    """Out-of-fold probabilities with whole groups (flights) per fold: no group is in both a fit and its scoring.

    The distinct groups are permuted with np.random.default_rng(seed) and cut into k folds (np.array_split).
    """
    groups = np.asarray(groups)
    g = np.random.default_rng(seed).permutation(np.unique(groups))
    out = np.zeros(len(y))
    for fold in np.array_split(g, k):
        te = np.isin(groups, fold)
        out[te] = logreg_predict(logreg_fit(X[~te], y[~te], **kw), X[te])
    return out


def auroc_cluster_ci(scores, labels, groups, n_boot=2000, seed=SEED):
    """95% bootstrap interval of the AUROC resampling whole groups (flights) with replacement.

    Frames of one flight are not independent, so this is wider than hlab.metrics.auroc_ci.
    Draws without both classes are skipped; returns (lo, hi, draws used).
    """
    s, y, groups = np.asarray(scores, float), np.asarray(labels, int), np.asarray(groups)
    ug = np.unique(groups)
    rows = [np.flatnonzero(groups == u) for u in ug]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(ug), len(ug))])
        if 0 < y[idx].sum() < len(idx):
            vals.append(auroc(s[idx], y[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), len(vals)


def check_trivial_baselines(neg_features, pos_features, seed=SEED):
    """No-model baselines on global_features rows: every single statistic's AUROC (oriented
    above 0.5), the three best, and the 5-fold logistic regression on all 16."""
    aj = [auroc(*from_groups(neg_features[:, j], pos_features[:, j])) for j in range(len(FEAT))]
    single = {k: float(max(v, 1 - v)) for k, v in zip(FEAT, aj)}
    top = sorted(single.items(), key=lambda kv: -kv[1])[:3]
    return dict(single=single, top=top, logreg=float(logreg_cv_auroc(neg_features, pos_features, seed=seed)))


# HER-107: the same frames under each (representation, fit) of hlab.transform, applied to both classes.
TRANSFORM_VARIANTS = [
    ("as_is", "scale_by_max", "stretch"),      # what pv/ holds
    ("crop", "scale_by_max", "crop"),          # aspect-ratio fix only
    ("minmax_crop", "minmax", "crop"),         # one stretch rule for both classes, plus crop
    ("equalize", "equalize", "stretch"),       # intensity fix only
    ("matched", "equalize", "crop"),           # what pv_matched/ holds (preprocess_pv_matched.py)
]
TRANSFORM_COLS = ["variant", "representation", "fit", "p1_median_healthy", "p1_median_fault", "auroc_p1",
                  "top3", "logreg"]
_IDENTITY = {"mean": 0.0, "std": 1.0}     # global_features denormalises; to_model_input is already 0-255


def transform_features(paths, representation, fit):
    """global_features of the raw files at `paths` after to_model_input(representation, fit)."""
    return global_features([to_model_input(load_gray(p), representation, fit) for p in paths], _IDENTITY)


def check_transform_baselines(healthy_paths, fault_paths, variants=TRANSFORM_VARIANTS, seed=SEED):
    """check_trivial_baselines for every (name, representation, fit) in `variants`, reloading the raw files.

    One row per variant: the p1 medians (0-255), the p1 AUROC, the three best single statistics and the
    16-statistic logistic regression, rounded as results/her-107/baselines.md shows them; the full
    check_trivial_baselines result is kept under "_baselines".
    """
    j = FEAT.index("p1")
    rows = []
    for name, representation, fit in variants:
        negF = transform_features(healthy_paths, representation, fit)
        posF = transform_features(fault_paths, representation, fit)
        b = check_trivial_baselines(negF, posF, seed=seed)
        rows.append(dict(variant=name, representation=representation, fit=fit,
                         p1_median_healthy=round(float(np.median(negF[:, j])), 1),
                         p1_median_fault=round(float(np.median(posF[:, j])), 1),
                         auroc_p1=round(b["single"]["p1"], 3),
                         top3=", ".join(f"{k} {v:.3f}" for k, v in b["top"]),
                         logreg=round(b["logreg"], 3), _baselines=b))
    return rows


def check_stored_baselines(dataset, gid, seed=SEED):
    """check_trivial_baselines on a PV dataset's stored arrays (holdout vs unique faults), None if not built."""
    if not (dataset.has_split("holdout") and dataset.has_split("fault")):
        return None
    negF = global_features(dataset.split("holdout").x, dataset.stats)
    posF = global_features(dataset.split("fault").x, dataset.stats)[unique_first(np.asarray(gid))]
    return check_trivial_baselines(negF, posF, seed=seed)


def image_formats(paths):
    """{(PIL mode, "WxH"): count} over image files, read from the headers only."""
    out = Counter()
    for p in paths:
        with Image.open(p) as im:
            out[(im.mode, f"{im.width}x{im.height}")] += 1
    return dict(out)


# ---------------------------------------------------------------- test-set hashes (HER-105)
_HASHES = {}


def pixel_hash(path):
    """md5 of a file's decoded pixel array (shape, dtype and bytes), as the loader receives it (EXIF tag ignored)."""
    key = str(path)
    if key not in _HASHES:
        a = np.array(Image.open(path))
        _HASHES[key] = hashlib.md5(f"{a.shape}{a.dtype}".encode() + a.tobytes()).hexdigest()
    return _HASHES[key]


def check_test_hashes(test_sets, train_pool):
    """Duplicate, cross-label and training-pool checks over any number of test sets.

    test_sets: {name: (paths, label)} with label 0 = healthy, 1 = fault; train_pool: {name: paths}
    of the images the model under test was trained on. Images are compared by pixel_hash.
    Per test set: n_files, n_unique (distinct images inside the set), shared (files whose image
    is also in another test set, by set), opposite_label (files whose image also appears in a
    test set of the other label), in_train (files whose image is in the training pool, by pool
    folder). total: n_files and n_unique over all test sets together, and in_train.
    """
    train_where = {}
    for pool, paths in train_pool.items():
        for p in paths:
            train_where.setdefault(pixel_hash(p), set()).add(pool)
    h = {name: [pixel_hash(p) for p in paths] for name, (paths, _lab) in test_sets.items()}
    label = {name: lab for name, (_paths, lab) in test_sets.items()}
    sets_of = {}
    for name, hs in h.items():
        for x in set(hs):
            sets_of.setdefault(x, set()).add(name)
    rows = {}
    for name, hs in h.items():
        shared = Counter(o for x in hs for o in sets_of[x] - {name})
        in_train = Counter(pool for x in hs for pool in train_where.get(x, ()))
        rows[name] = dict(n_files=len(hs), n_unique=len(set(hs)), shared=dict(shared),
                          opposite_label=sum(any(label[o] != label[name] for o in sets_of[x]) for x in hs),
                          in_train=sum(x in train_where for x in hs), in_train_by_pool=dict(in_train))
    every = [x for hs in h.values() for x in hs]
    total = dict(n_files=len(every), n_unique=len(set(every)), in_train=sum(x in train_where for x in every))
    return dict(sets=rows, total=total)


# ---------------------------------------------------------------- 6. rule layer
def rule_signals(config, images_0_255, equipment):
    """The rule layer's temperature signal (Verdict.signal_c, e.g. the PV hot-spot dT proxy) per image."""
    return np.array([evaluate(config, img, equipment).signal_c for img in images_0_255])


def check_rule_layer(config, images_0_255, equipment, ml_flags):
    """(severity counts, ensemble decision counts, images at ALARM or CRITICAL) via hlab.rules per image."""
    sev, final = [], []
    for img, ml in zip(images_0_255, ml_flags):
        v = evaluate(config, img, equipment)
        sev.append(v.severity)
        final.append({"flag": "FLAG", "monitor": "MONITOR", "pass": "NORMAL"}[combine(bool(ml), v).action])
    return dict(Counter(sev)), dict(Counter(final)), sum(s in ("ALARM", "CRITICAL") for s in sev)


# ---------------------------------------------------------------- 7. India radiometric audit (HER-131)
INDIA_READINGS = ("minmax", "celsius")     # the india/ model input and the india_temp/ fixed scale
FIT_SPLITS = ("train", "train_faulty_unused")   # plot 17, the plot the India models were fit on
SPREAD_UNDER_C = 2.0                       # the ticket's "fraction under 2 degrees C"


def celsius_0_255(t):
    """Degrees C on the fixed CELSIUS_RANGE scale (india_temp/), mapped linearly to 0-255 so global_features applies.

    frac_ge250 then counts pixels at or above 117.6 C and frac_le5 those at or below 2.4 C.
    """
    lo, hi = CELSIUS_RANGE
    return (to_model_input(t, "celsius") - lo) * (255.0 / (hi - lo))


def india_frame_features(raw, celsius):
    """{reading: global_features row} of one R-JPG: minmax of the raw counts, and degrees C on the fixed scale."""
    x = {"minmax": to_model_input(raw, "minmax"), "celsius": celsius_0_255(celsius)}
    return {k: global_features([x[k]], _IDENTITY)[0] for k in INDIA_READINGS}


def box_pixels(box, shape):
    """(r0, r1, c0, c1) slice bounds of a labelme box (x0, y0, x1, y1 source pixels, inclusive), rounded and clipped."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    r0, c0 = min(max(y0, 0), shape[0] - 1), min(max(x0, 0), shape[1] - 1)
    return r0, max(r0 + 1, min(y1 + 1, shape[0])), c0, max(c0 + 1, min(x1 + 1, shape[1]))


def box_spread(celsius, box):
    """(max - median, max - min, pixels) of the temperatures inside one box."""
    r0, r1, c0, c1 = box_pixels(box, celsius.shape)
    v = celsius[r0:r1, c0:c1]
    return float(v.max() - np.median(v)), float(v.max() - v.min()), int(v.size)


def control_boxes(fault_boxes, healthy_rows, seed=SEED):
    """One same-size, same-position box per fault box, in a healthy frame of the same flight.

    fault_boxes: [(rel_path, flight, plot, box)]; healthy_rows: manifest rows (rel_path, flight, plot).
    The healthy frame is drawn uniformly with np.random.default_rng(seed) from the box's flight, or
    from its plot when the flight has no healthy frame. Returns {healthy rel_path: [(index into
    fault_boxes, box)]} and the number of boxes that fell back to the plot.
    """
    rng = np.random.default_rng(seed)
    by_flight, by_plot = {}, {}
    for r in healthy_rows:
        by_flight.setdefault(r["flight"], []).append(r["rel_path"])
        by_plot.setdefault(r["plot"], []).append(r["rel_path"])
    out, fallback = {}, 0
    for i, (_p, flight, plot, box) in enumerate(fault_boxes):
        pool = by_flight.get(flight)
        if not pool:
            pool, fallback = by_plot[plot], fallback + 1
        out.setdefault(pool[int(rng.integers(len(pool)))], []).append((i, box))
    return out, fallback


def spread_summary(v, under=SPREAD_UNDER_C):
    """n, mean, 5/25/50/75/95th percentiles and the fraction under `under` degrees C."""
    v = np.asarray(v, float)
    p = np.percentile(v, [5, 25, 50, 75, 95])
    return dict(n=len(v), mean=float(v.mean()), p5=float(p[0]), p25=float(p[1]), p50=float(p[2]),
                p75=float(p[3]), p95=float(p[4]), frac_under=float((v < under).mean()))


def check_box_spread(fault, control, under=SPREAD_UNDER_C):
    """Temperature spread inside the fault boxes vs the same boxes placed in healthy frames.

    fault / control: {"max_minus_median": [...], "max_minus_min": [...]} in degrees C, one value per box.
    Per measure: both summaries and the AUROC of fault over control boxes (0.5 = no difference).
    """
    return {m: dict(fault=spread_summary(fault[m], under), control=spread_summary(control[m], under),
                    auroc=auroc(*from_groups(control[m], fault[m])))
            for m in ("max_minus_median", "max_minus_min")}


_SHOT = re.compile(r"DJI_(\d+)_R(?: \((\d+)\))?\.JPG$", re.I)


def shot_key(name, taken=""):
    """Sort key of a frame in its flight: (EXIF DateTime, copy suffix, shot number).

    232 frames are named like "DJI_0325_R (2).JPG" and are different shots from their namesakes
    (a second counter in the same folder), so the capture time orders the shots first.
    """
    m = _SHOT.search(name)
    if not m:
        raise ValueError(f"not a DJI R-JPG name: {name}")
    return taken, int(m.group(2) or 0), int(m.group(1))


def place_unreadable(order, unreadable):
    """`order` (readable names in capture order) with every name in `unreadable` given a slot of its own.

    A file without a capture time keeps a place instead of being dropped, so the shots on either side of
    it never become adjacent. Its slot is an estimate: right after the readable shot with the nearest
    lower number of the same counter (same copy suffix and number parity, as the two DJI counters in one
    folder step by 2), or first when there is none. Unreadable names are placed in number order.
    """
    out = list(order)
    num = lambda n: (int(_SHOT.search(n).group(2) or 0), int(_SHOT.search(n).group(1)))  # noqa: E731
    for u in sorted(unreadable, key=num):
        cu, nu = num(u)
        before = [i for i, n in enumerate(out) if num(n)[0] == cu and num(n)[1] % 2 == nu % 2 and num(n)[1] < nu]
        out.insert(max(before, key=lambda i: num(out[i])[1]) + 1 if before else 0, u)
    return out


def consecutive_groups(positions, gap=1):
    """Groups of positions (sorted) where each member is at most `gap` shots after the previous one."""
    pos = sorted(positions)
    groups = []
    for p in pos:
        if groups and p - groups[-1][-1] <= gap:
            groups[-1].append(p)
        else:
            groups.append([p])
    return groups


def check_fault_duplicates(boxed, order, split_of, gaps=(1, 2, 5)):
    """Distinct faults when boxed frames in consecutive shots of one flight are counted once.

    boxed: {folder: [boxed frame names]}; order: {folder: [every frame name on disk, in shot order]};
    split_of: {(folder, name): split}. gap g joins boxed frames at most g shots apart (g = 1: the
    very next shot; g = 5: the +-5 shots the reviewed-healthy selection treats as the same modules).
    A group's split is the split of its frames (a flight is in one split only).
    Per gap: distinct groups, group-size counts, largest group, and boxed frames / groups by split.
    """
    missing = [f"{folder}/{n}" for folder, names in boxed.items() for n in names if n not in set(order[folder])]
    if missing:
        raise ValueError(f"{len(missing)} boxed frame(s) missing from the shot order: {', '.join(missing[:5])}")
    out = {}
    for g in gaps:
        sizes, by_split = Counter(), {}
        for folder, names in boxed.items():
            idx = {n: i for i, n in enumerate(order[folder])}
            for grp in consecutive_groups([idx[n] for n in names], g):
                sizes[len(grp)] += 1
                s = split_of[(folder, order[folder][grp[0]])]
                f, d = by_split.get(s, (0, 0))
                by_split[s] = (f + len(grp), d + 1)
        out[g] = dict(boxed=sum(k * v for k, v in sizes.items()), distinct=sum(sizes.values()),
                      sizes=dict(sorted(sizes.items())), largest=max(sizes), by_split=dict(sorted(by_split.items())))
    return out


def _score_row(s, y, flights=None, seed=SEED):
    lo, hi = auroc_ci(s, y, seed=seed)
    row = dict(n_healthy=int((y == 0).sum()), n_faulty=int(y.sum()), auroc=auroc(s, y), ci=(lo, hi))
    if flights is not None and len(np.unique(flights)) > 2:
        row["ci_flight"] = auroc_cluster_ci(s, y, flights, seed=seed)[:2]
    return row


def check_india_baselines(F, y, flight, split, cert_F=None, cert_y=None, seed=SEED):
    """The 16-statistic logistic regression on the India frames, split by flight, with the cross-site check.

    F: global_features rows; y: 1 = faulty; flight, split: per row, from india_plot_split.csv.
      group_cv     every India frame scored out of fold, 5 folds of whole flights (logreg_group_cv_scores)
      random_cv    the same with frames shuffled into folds (logreg_cv_scores), the HER-104 protocol, for contrast
      plot16 / plot18 / plot16_18
                   fit on plot 17 (train + train_faulty_unused), scored on plot 16 (val), plot 18 (test), both
      cross_site   fit on every India frame, scored on cert_F / cert_y (other sites)
    Each row: n_healthy, n_faulty, AUROC, frame bootstrap CI (hlab.metrics.auroc_ci) and, in-site,
    the flight bootstrap CI (auroc_cluster_ci). single: every statistic's AUROC on all India frames,
    oriented above 0.5 (orientation chosen on the same frames, so slightly optimistic).
    """
    F, y, flight, split = np.asarray(F, float), np.asarray(y, int), np.asarray(flight), np.asarray(split)
    out = {"group_cv": _score_row(logreg_group_cv_scores(F, y, flight, seed=seed), y, flight, seed)}
    s, yy = logreg_cv_scores(F[y == 0], F[y == 1], seed=seed)
    out["random_cv"] = _score_row(s, yy, np.r_[flight[y == 0], flight[y == 1]], seed)
    fit_rows = np.isin(split, FIT_SPLITS)
    fit = logreg_fit(F[fit_rows], y[fit_rows])
    for name, keep in (("plot16", split == "val"), ("plot18", split == "test"),
                       ("plot16_18", np.isin(split, ("val", "test")))):
        out[name] = _score_row(logreg_predict(fit, F[keep]), y[keep], flight[keep], seed)
    if cert_F is not None and len(cert_F):
        out["cross_site"] = _score_row(logreg_predict(logreg_fit(F, y), np.asarray(cert_F, float)),
                                       np.asarray(cert_y, int), seed=seed)
    aj = [auroc(F[:, j], y) for j in range(len(FEAT))]
    out["single"] = {k: float(max(v, 1 - v)) for k, v in zip(FEAT, aj)}
    return out


# ---------------------------------------------------------------- scenarios
def source_labels(split):
    """'thm' | 'om_dr' | 'om_sr' per healthy frame of a PV holdout split."""
    return np.array([s if s == "thm" else f"om_{m['sequence']}" for s, m in zip(split.source, split.meta)])


class PVInputs:
    """One PV dataset's held-out healthy frames, PVMD faults and the errors its bundle's evaluator saved.

    neg / pos: saved errors (healthy, 1000 fault files); fu: one error per pixel-unique fault;
    src: "om" | "thm"; seq: "om_dr" | "om_sr" | "thm". features() computes the global statistics
    once (negF over the holdout, posF_u over the unique faults' first file).
    `errors` and `gid` can be passed directly (tests); by default they come from saved_errors(bundle)
    and pvmd_unique_groups().
    """

    def __init__(self, dataset, bundle, gid=None, errors=None):
        self.dataset, self.bundle = dataset, bundle
        self.holdout, self.fault = dataset.split("holdout"), dataset.split("fault")
        self.ho = self.holdout.frame_ids
        self.src, self.seq = self.holdout.source, source_labels(self.holdout)
        self.n_train = sum(1 for _f, split, _src in dataset.manifest() if split == "train")
        self.gid = pvmd_unique_groups()[1] if gid is None else np.asarray(gid)
        self.neg, self.pos = saved_errors(bundle) if errors is None else errors
        if len(self.neg) != len(self.holdout) or len(self.pos) != len(self.fault) or len(self.pos) != len(self.gid):
            raise RuntimeError(f"saved errors do not match {dataset}: {len(self.neg)} healthy for "
                               f"{len(self.holdout)} frames, {len(self.pos)} faults for {len(self.fault)} frames "
                               f"and {len(self.gid)} PVMD files")
        self.n_unique = int(self.gid.max()) + 1
        self.fu = unique_fault_scores(self.pos, self.gid)
        self._negF = self._posF_u = None

    def features(self):
        """(healthy features, unique-fault features), computed on first call."""
        if self._negF is None:
            self._negF = global_features(self.holdout.x, self.dataset.stats)
            posF = global_features(self.fault.x, self.dataset.stats)
            self._posF_u = posF[unique_first(self.gid)]
        return self._negF, self._posF_u


def run_scenario(name, keep, inputs, thr, seed=SEED):
    """Every check over the held-out healthy frames selected by `keep`, scored at threshold `thr`.

    Returns the summary row (scenario, n_*, auroc_*, fp_*, f1_*, auroc_source*, p1_baseline,
    best_single, logreg16, leak_adjacent) plus the full check results under "_files", "_unique",
    "_source" and "_baselines" for print_scenario.
    """
    keep = np.asarray(keep, bool)
    n = inputs.neg[keep]
    src = inputs.seq[keep]
    negF, posF_u = inputs.features()
    S = {"scenario": name, "n_healthy": int(len(n)), "n_thm": int((src == "thm").sum()),
         "n_om": int((src != "thm").sum()), "n_fault_files": int(len(inputs.pos)),
         "n_fault_unique": int(len(inputs.fu))}
    S["leak_adjacent"], S["_n_checked"] = check_temporal_leakage(inputs.dataset, inputs.holdout, keep=keep)
    for key, pos in (("files", inputs.pos), ("unique", inputs.fu)):
        h = check_headline(n, pos, thr, seed=seed)
        c = rounded(h["saved"])
        S[f"_{key}"] = h
        S[f"auroc_{key}"] = round(h["auroc"], 4); S[f"auroc_{key}_ci"] = "[{:.3f}, {:.3f}]".format(*h["ci"])
        S[f"f1_trivial_{key}"] = round(h["trivial_f1"], 4); S[f"f1_saved_{key}"] = c["f1"]
        S[f"fp_{key}"] = c["fp"]; S[f"recall_{key}"] = c["recall"]; S[f"precision_{key}"] = c["precision"]
    src_check = check_source_shortcut(n, src, inputs.fu, thr, order=("om_dr", "om_sr", "thm"))
    S["_source"] = src_check
    S["fp_thm"], S["fp_om"] = src_check["fp_thm"], src_check["fp_om"]
    S["fp_share_thm"] = round(S["fp_thm"] / max(S["fp_thm"] + S["fp_om"], 1), 3)
    S["auroc_source"] = round(src_check["auroc_source"], 3); S["auroc_source_def"] = src_check["auroc_source_def"]
    base = check_trivial_baselines(negF[keep], posF_u, seed=seed)
    S["_baselines"] = base
    S["p1_baseline"] = round(base["single"]["p1"], 3)
    S["best_single"] = f"{base['top'][0][0]} {base['top'][0][1]:.3f}"
    S["logreg16"] = round(base["logreg"], 3)
    return S


def print_scenario(S):
    """The report block of one scenario, as results/her-114/rerun_<scenario>.txt shows it."""
    print(f"healthy frames kept: {S['n_healthy']}  (O&M {S['n_om']}, THM {S['n_thm']})")
    print(f"held-out healthy with an adjacent frame in TRAIN: {S['leak_adjacent']} / {S['_n_checked']}")
    for lab, key in ((f"{S['n_fault_files']} files", "files"), (f"{S['n_fault_unique']} unique", "unique")):
        h = S[f"_{key}"]
        print(f"\n{lab}: AUROC {h['auroc']:.4f} [{h['ci'][0]:.3f}, {h['ci'][1]:.3f}]  "
              f"flag-everything F1 {h['trivial_f1']:.4f}")
        print("  at saved threshold:", rounded(h["saved"]))
    src = S["_source"]
    print(f"\nfalse positives at saved threshold: THM {S['fp_thm']} / {S['n_thm']}   O&M {S['fp_om']} / {S['n_om']}"
          f"   THM share {S['fp_share_thm']:.1%}")
    for s, r in src["rows"].items():
        print(f"  {s:5s}: n={r['n']:4d}  median err {r['median']:.4f}  flagged {r['flagged']}")
    print(f"source AUROC ({S['auroc_source_def']}): {S['auroc_source']}")
    print(f"AUROC healthy O&M vs unique fault: {src['auroc_om_fault']:.3f}")
    if src["auroc_thm_fault"] is not None:
        print(f"AUROC healthy THM vs unique fault: {src['auroc_thm_fault']:.3f}")
    base = S["_baselines"]
    print(f"single-statistic AUROC (unique faults): p1 = {base['single']['p1']:.3f}; "
          f"best {[(k, round(v, 3)) for k, v in base['top']]}")
    print(f"5-fold logistic regression on 16 statistics: AUROC {base['logreg']:.3f}")


def pv_frame_paths(dataset, gid, root=None):
    """(held-out healthy files, first file of every unique fault) of a PV dataset, in its manifest order."""
    root = ROOT if root is None else root
    rows = dataset.manifest()
    healthy = [root / f for f, split, _src in rows if split == "holdout"]
    faults = [root / f for f, split, _src in rows if split == "fault"]
    if len(gid) != len(faults):
        raise RuntimeError(f"{len(faults)} fault files in the {dataset.name} manifest, {len(gid)} PVMD files")
    return healthy, [faults[i] for i in unique_first(np.asarray(gid))]


def single_statistic_table(rows):
    """Markdown: one line per global statistic, its AUROC under every check_transform_baselines row."""
    out = ["| statistic | " + " | ".join(r["variant"] for r in rows) + " |", "|---|" + "---|" * len(rows)]
    for k in FEAT:
        out.append(f"| {k} | " + " | ".join(f"{r['_baselines']['single'][k]:.3f}" for r in rows) + " |")
    return "\n".join(out)


def read_training_log(path):
    """The numbers the training script logged: train/val sizes, parameters, best and spike epochs, stop, minutes."""
    text = Path(path).read_text(encoding="utf-8")
    epochs = [(int(e), float(t), float(v)) for e, t, v in
              re.findall(r"Epoch\s+(\d+)/\d+ -- train_loss: ([\d.]+)\s+val_loss: ([\d.]+)", text)]
    best = min(epochs, key=lambda r: r[2])
    worst = max(epochs, key=lambda r: r[2])
    num = lambda pat: re.search(pat, text).group(1)  # noqa: E731
    stop = re.search(r"Early stopping at epoch (\d+)", text)
    return dict(n_train=int(num(r"Train:\s+(\d+)")), n_val=int(num(r"Val:\s+(\d+)")),
                parameters=int(num(r"Parameters: ([\d,]+)").replace(",", "")),
                best_epoch=best[0], best_val_loss=best[2], spike_epoch=worst[0], spike_val_loss=worst[2],
                stop_epoch=int(stop.group(1)) if stop else None, minutes=float(num(r"complete in ([\d.]+) min")),
                val_loss=[v for _e, _t, v in epochs])


# ---------------------------------------------------------------- repeated runs and intervals (HER-120)
SPIKE_RATIO = 2.0               # a val-loss spike: an epoch at least this many times the lowest val loss before it


def val_loss_jump(val_loss):
    """(largest ratio of an epoch's val loss to the lowest val loss before it, the epoch it happens at), 1-based.

    (0.0, None) for fewer than two epochs. The recorded PV run's epoch-6 spike is 1.04 / 0.0170, about 61x.
    """
    v = np.asarray(val_loss, dtype=float)
    if len(v) < 2:
        return 0.0, None
    ratio = v[1:] / np.minimum.accumulate(v)[:-1]
    i = int(np.argmax(ratio))
    return float(ratio[i]), i + 2


def threshold_free_stats():
    """The statistics that need no alarm threshold: AUROC, AUPRC, TPR at FPR <= 5%."""
    return {"auroc": auroc, "auprc": average_precision, "tpr_at_fpr5": lambda s, y: tpr_at_fpr(s, y, 0.05)}


def threshold_stats(thr):
    """Precision, recall and F1 at score > thr."""
    return {k: (lambda s, y, k=k: confusion(s, y, thr)[k]) for k in ("precision", "recall", "f1")}


def check_unique_intervals(neg, pos_unique, thr=None, n_boot=2000, seed=SEED):
    """95% bootstrap intervals on the deduplicated set: healthy frames and one score per unique fault.

    Resampling is stratified, with the unique fault image (not the PVMD file) as the fault unit, so
    the fault side of every interval rests on the 140 distinct images. Always AUROC, AUPRC and TPR at
    FPR <= 5%; precision, recall and F1 too when a threshold is given. {name: (point, lo, hi)}.
    The AUROC interval is the one check_headline reports for the same inputs and seed.
    """
    stats = threshold_free_stats()
    if thr is not None:
        stats.update(threshold_stats(thr))
    return bootstrap_cis(*from_groups(neg, pos_unique), stats, n_boot=n_boot, seed=seed)


def check_seed_runs(runs, gid, thr=None, spike_ratio=SPIKE_RATIO):
    """Per-seed training and test numbers of repeated runs, and their spread.

    runs: [(label, healthy errors, fault errors per PVMD file, read_training_log dict)], every run
    scored on the same test set. gid: the PVMD pixel groups, so each run is scored on one error per
    unique fault (unique_fault_scores). thr: a frozen alarm threshold for precision / recall / F1, or
    None (threshold-free statistics only; a per-run best-F1 threshold would tune on the test set).

    Returns dict(rows=[{label, best_epoch, best_val_loss, epochs_run, jump, jump_epoch, spike, auroc,
    auprc, tpr_at_fpr5[, precision, recall, f1]}], summary={stat: (mean, sd, min, max)}, n, n_spike).
    sd is the sample standard deviation (ddof 1; 0.0 for one run).
    """
    stats = threshold_free_stats()
    if thr is not None:
        stats.update(threshold_stats(thr))
    rows = []
    for label, neg, pos, log in runs:
        s, y = from_groups(neg, unique_fault_scores(pos, gid))
        jump, jump_epoch = val_loss_jump(log["val_loss"])
        row = dict(label=label, best_epoch=log["best_epoch"], best_val_loss=log["best_val_loss"],
                   epochs_run=len(log["val_loss"]), jump=jump, jump_epoch=jump_epoch, spike=jump >= spike_ratio)
        row.update({k: float(f(s, y)) for k, f in stats.items()})
        rows.append(row)
    summary = {}
    for k in ("best_epoch", "best_val_loss", *stats):
        v = np.array([r[k] for r in rows], dtype=float)
        summary[k] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, float(v.min()), float(v.max()))
    return dict(rows=rows, summary=summary, n=len(rows), n_spike=sum(r["spike"] for r in rows))


# ---------------------------------------------------------------- paper figures
_COLORS = {"om": "#2a7bd6", "thm": "#e8683a", "fault": "#1aaf7a", "grid": "#e6e6e6", "ink": "#4d4d4d"}


def _figure_style(plt):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": _COLORS["ink"], "axes.labelcolor": "black",
                         "xtick.color": _COLORS["ink"], "ytick.color": _COLORS["ink"],
                         "axes.spines.top": False, "axes.spines.right": False})


def fig_error_by_source(neg_om, neg_thm, fault_unique, thr, path):
    """Fig. 2: per-image reconstruction error (log scale) of healthy O&M, healthy THM and unique faults, with tau."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _figure_style(plt)
    groups = [(neg_om, f"Healthy, O&M source (n={len(neg_om)})", _COLORS["om"]),
              (neg_thm, f"Healthy, THM source (n={len(neg_thm)})", _COLORS["thm"]),
              (fault_unique, f"PVMD fault, unique images (n={len(fault_unique)})", _COLORS["fault"])]
    every = np.concatenate([g[0] for g in groups])
    bins = np.logspace(np.log10(every.min()), np.log10(every.max()), 45)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for e, lab, c in groups:
        ax.hist(e, bins=bins, weights=np.full(len(e), 1 / len(e)), histtype="step", lw=1.8, color=c, label=lab)
    ax.axvline(thr, ls="--", lw=1.0, color=_COLORS["ink"])
    ax.text(thr * 0.95, ax.get_ylim()[1] * 0.97, "F1-optimal\nthreshold", ha="right", va="top", fontsize=8,
            color=_COLORS["ink"])
    ax.set_xscale("log")
    ax.yaxis.grid(True, color=_COLORS["grid"]); ax.set_axisbelow(True)
    ax.set_xlabel("Per-image reconstruction MSE (z-scored units, log scale)")
    ax.set_ylabel("Fraction of group")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def fig_roc_vs_baselines(curves, path, title=None):
    """Fig. 3: ROC curves; `curves` is [(label, scores, labels), ...] and each legend entry carries its AUROC."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _figure_style(plt)
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    for (lab, s, y), c in zip(curves, (_COLORS["om"], _COLORS["thm"], _COLORS["fault"])):
        fpr, tpr = roc_points(s, y)
        ax.step(fpr, tpr, where="post", lw=1.8, color=c, label=f"{lab} ({auroc(s, y):.3f})")
    ax.plot([0, 1], [0, 1], ls=":", lw=0.8, color="#bbbbbb")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
    ax.grid(True, color=_COLORS["grid"]); ax.set_axisbelow(True)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower right", title="AUROC in parentheses", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def fig_pr_curves(curves, path, title=None):
    """Precision-recall curves on one set of axes; `curves` is [(label, scores, labels), ...]. Each legend entry
    carries its AUPRC and each curve's chance level (its fault prevalence) is a dashed line in the same colour."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _figure_style(plt)
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    for (lab, s, y), c in zip(curves, (_COLORS["om"], _COLORS["fault"], _COLORS["thm"])):
        rec, prec = pr_points(s, y)
        prev = float(np.mean(y))
        ax.plot(rec, prec, drawstyle="steps-post", lw=1.8, color=c, label=f"{lab} ({average_precision(s, y):.3f})")
        ax.axhline(prev, ls="--", lw=0.9, color=c)
        ax.text(0.02, prev + 0.012, f"chance {prev:.3f}", fontsize=7, color=c, va="bottom")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
    ax.grid(True, color=_COLORS["grid"]); ax.set_axisbelow(True)
    ax.set_xlabel("Recall (fault)"); ax.set_ylabel("Precision")
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 0.18),
              title="AUPRC in parentheses", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def write_summary(rows, cols, md_path, csv_path):
    """Write the scenario rows as a Markdown table and a CSV; returns the Markdown."""
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    text = "\n".join(md)
    md_path.write_text(text + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return text


if __name__ == "__main__":      # `python docs/paper/audit.py` prints the paper report, as audit_eval_validity.py does
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audit_eval_validity import main
    main()
