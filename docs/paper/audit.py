"""
Paper audit checks (HER-130): every check is a function of a Dataset, a Bundle and a keep-mask.

The three entry points (audit_eval_validity.py, audit_her114_reruns.py,
audit_her114_retrain.py) only choose scenarios and print; the numbers come from
here, and from here only through hlab: metrics (AUROC, bootstrap CI, confusion,
threshold search), rules (tiers and decisions), datasets (splits, manifests,
PVMD pixel groups, adjacent frames) and bundle (saved thresholds, never weights).
No check retrains or re-runs a model: the reconstruction errors are the ones the
evaluators saved under results/ (saved_errors).

Checks
  check_pvmd_duplicates()                       pixel-identical PVMD files, with and without the EXIF orientation tag
  check_temporal_leakage(dataset, split, keep)  held-out frames with an adjacent video frame in TRAIN
  check_source_shortcut(neg, src, fu, thr)      errors by acquisition source, source AUROCs, false positives by source
  check_headline(neg, pos, thr, seed)           AUROC with bootstrap CI, confusion at the saved and re-tuned thresholds, trivial F1
  check_trivial_baselines(negF, posF, seed)     single global statistics and the 16-statistic logistic regression (no model)
  check_rule_layer(config, images, equipment, ml_flags)   severity tiers and ensemble decisions, hlab.rules per image

Scenarios
  PVInputs(dataset, bundle)                     one PV dataset's holdout, faults, saved errors, unique-fault means, features
  run_scenario(name, keep, inputs, thr, seed)   every check above a keep-mask over the held-out healthy frames
  print_scenario(S)                             the report block the HER-114 reruns print
  write_summary(rows, cols, md, csv)            the side-by-side table

Feature extraction (global_features, FEAT) and the no-model baselines stay
here: they belong to the audit, not to a detector. Seeds are arguments
(SEED = 0 is the default the paper numbers were produced with); nothing here
is configured by assigning a module global from outside.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # until hlab is installable
from hlab.datasets import PV_FAULT_FOLDER, adjacent_frame, pvmd_unique_groups  # noqa: E402
from hlab.metrics import auroc, auroc_ci, best_f1_threshold, confusion, from_groups, trivial_f1  # noqa: E402
from hlab.paths import RAW, RESULTS  # noqa: E402
from hlab.rules import combine, evaluate  # noqa: E402
from hlab.transform import denormalize  # noqa: E402

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
    return dict(files=files, gid=gid, n_files=len(files), n_unique=len(groups),
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
def check_temporal_leakage(dataset, split="holdout", keep=None, train="train"):
    """(frames with an immediately adjacent video frame in the TRAIN manifest, frames checked).

    `split` is a split name or a Split; `keep` restricts it to a scenario's frames.
    """
    s = dataset.split(split) if isinstance(split, str) else split
    ids = s.frame_ids if keep is None else [f for f, k in zip(s.frame_ids, keep) if k]
    trs = {f for f, sp, _src in dataset.manifest() if sp == train}
    adj = sum(any(adjacent_frame(p, d) in trs for d in (-1, 1)) for p in ids)
    return adj, len(ids)


# ---------------------------------------------------------------- 3. source shortcut
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
def check_headline(neg, pos, thr, seed=SEED):
    """AUROC with bootstrap 95% CI, trivial flag-everything F1, confusion at the saved and re-tuned thresholds."""
    s, y = from_groups(neg, pos)
    lo, hi = auroc_ci(s, y, seed=seed)
    retuned_thr = best_f1_threshold(s, y)
    return dict(auroc=auroc(s, y), ci=(lo, hi), trivial_f1=trivial_f1(y), saved=confusion(s, y, thr),
                retuned_threshold=retuned_thr, retuned=confusion(s, y, retuned_thr))


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
    rng = np.random.default_rng(seed)
    X = np.vstack([X_neg, X_pos]); y = np.r_[np.zeros(len(X_neg)), np.ones(len(X_pos))]
    idx_n = rng.permutation(len(X_neg)); idx_p = rng.permutation(len(X_pos)) + len(X_neg)
    folds = [np.r_[a, b] for a, b in zip(np.array_split(idx_n, k), np.array_split(idx_p, k))]
    out = np.zeros(len(y))
    for f in folds:
        tr = np.setdiff1d(np.arange(len(y)), f)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.c_[np.ones(len(tr)), (X[tr] - mu) / sd]
        w = np.zeros(A.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-A @ w))
            g = A.T @ (p - y[tr]) / len(tr) + l2 * np.r_[0, w[1:]] / len(tr)
            w -= lr * g
        out[f] = 1 / (1 + np.exp(-(np.c_[np.ones(len(f)), (X[f] - mu) / sd] @ w)))
    return auroc(out, y)


def check_trivial_baselines(neg_features, pos_features, seed=SEED):
    """No-model baselines on global_features rows: every single statistic's AUROC (oriented
    above 0.5), the three best, and the 5-fold logistic regression on all 16."""
    aj = [auroc(*from_groups(neg_features[:, j], pos_features[:, j])) for j in range(len(FEAT))]
    single = {k: float(max(v, 1 - v)) for k, v in zip(FEAT, aj)}
    top = sorted(single.items(), key=lambda kv: -kv[1])[:3]
    return dict(single=single, top=top, logreg=float(logreg_cv_auroc(neg_features, pos_features, seed=seed)))


# ---------------------------------------------------------------- 6. rule layer
def check_rule_layer(config, images_0_255, equipment, ml_flags):
    """(severity counts, ensemble decision counts, images at ALARM or CRITICAL) via hlab.rules per image."""
    sev, final = [], []
    for img, ml in zip(images_0_255, ml_flags):
        v = evaluate(config, img, equipment)
        sev.append(v.severity)
        final.append({"flag": "FLAG", "monitor": "MONITOR", "pass": "NORMAL"}[combine(bool(ml), v).action])
    return dict(Counter(sev)), dict(Counter(final)), sum(s in ("ALARM", "CRITICAL") for s in sev)


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
         "n_om": int((src != "thm").sum())}
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
    for lab, key in (("1000 files", "files"), ("140 unique", "unique")):
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
