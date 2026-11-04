"""
Evaluation-validity audit for the HERIS Lab thermal anomaly detectors.

Reproduces every number in the paper draft (Problem Statement, Method, Results) that is
not already printed by evaluate_autoencoder.py / evaluate_autoencoder_pv.py.
It never retrains or re-runs a model: it reads the saved reconstruction errors
in results/, the preprocessed arrays and split manifests through
hlab.datasets, the saved thresholds through hlab.bundle (weights are never
loaded), and the raw PVMD files.

Checks:
  1. PVMD duplicates: pixel-identical files, and files whose pixels appear
     under more than one fault label
  2. Temporal leakage: held-out PV normals with an adjacent video frame in TRAIN
  3. Source shortcut: AE error split by acquisition source (O&M vs THM)
  4. Trivial baselines: single global intensity statistics and a 5-fold
     logistic regression on 16 global statistics (no model)
  5. Headline metrics with bootstrap 95% CIs, on the full and deduplicated set
  6. Rule layer: tier distribution and ensemble verdicts on the test sets,
     through hlab.rules.evaluate per image (the same code as the demo)

Metrics (AUROC, bootstrap CI, confusion, threshold search) come from
src/hlab/metrics.py (HER-122) and the rule layer from src/hlab/rules.py
(HER-124); this file keeps feature extraction and the no-model baselines. Dependencies: numpy, Pillow (already in the project venv).

Usage:
    venv\\Scripts\\python.exe docs\\paper\\audit_eval_validity.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # until hlab is installable
from hlab.bundle import Bundle  # noqa: E402
from hlab.datasets import Dataset, adjacent_frame, pvmd_unique_groups  # noqa: E402
from hlab.metrics import auroc, auroc_ci, best_f1_threshold, confusion, from_groups, trivial_f1  # noqa: E402
from hlab.paths import PREPROCESSED as PRE, RAW, RESULTS as RES, ROOT  # noqa: E402,F401
from hlab.rules import RuleConfig, combine, evaluate  # noqa: E402
from hlab.transform import denormalize  # noqa: E402

# The evaluators' frozen best-F1 thresholds (results/*_eval_metrics.npy), read through the bundles.
PV_THR = Bundle.load("pv").require_threshold()
V2_THR = Bundle.load("v2").require_threshold()
SEED = 0                        # bootstrap / cross-validation seed, passed explicitly


# ---------------------------------------------------------------- metrics
def rounded(c):
    """A hlab.metrics confusion dict as the audit prints it: counts, then rates to 4 decimals."""
    return dict(tp=c["tp"], fn=c["fn"], fp=c["fp"], tn=c["tn"], precision=round(c["precision"], 4),
                recall=round(c["recall"], 4), f1=round(c["f1"], 4), specificity=round(c["specificity"], 4))


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


# ---------------------------------------------------------------- features
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


def rule_counts(config, images, equipment, ml_flags):
    """Severity and ensemble counts over images, via hlab.rules (one implementation of the tiers)."""
    sev, final = [], []
    for img, ml in zip(images, ml_flags):
        v = evaluate(config, img, equipment)
        sev.append(v.severity)
        final.append({"flag": "FLAG", "monitor": "MONITOR", "pass": "NORMAL"}[combine(bool(ml), v).action])
    return dict(Counter(sev)), dict(Counter(final)), sum(s in ("ALARM", "CRITICAL") for s in sev)


def main():
    print("=" * 70, "\n1. PVMD duplicates\n" + "=" * 70)
    fault_dir = RAW / "test" / "fault" / "pv"
    files, gid, groups = pvmd_unique_groups()   # same file order as preprocess_pv.py
    multi = sum(len(v) for v in groups.values() if len({x[0].upper() for x in v}) > 1)
    print(f"files: {len(files)}   pixel-unique images: {len(groups)}")
    print(f"files whose pixels also appear under another fault label: {multi}")
    print("label mix of unique images:", dict(Counter("".join(sorted({x[0].upper() for x in v})) for v in groups.values())))
    # PVMD stores some rotation augmentations only as an EXIF orientation tag, which the
    # preprocessing ignores. Recount with the tag applied so both views are reported.
    exif_tag = {f: Image.open(fault_dir / f).getexif().get(274, 1) for f in files}
    _, _, groups_t = pvmd_unique_groups(exif_transpose=True)
    multi_t = sum(len(v) for v in groups_t.values() if len({x[0].upper() for x in v}) > 1)
    print(f"files with a non-default EXIF orientation tag: {sum(1 for v in exif_tag.values() if v != 1)}")
    print(f"unique images with EXIF orientation applied: {len(groups_t)}   "
          f"files sharing an image with another label: {multi_t}")
    print("label mix with EXIF applied:", dict(Counter("".join(sorted({x[0].upper() for x in v})) for v in groups_t.values())))
    G = len(groups)

    print("\n" + "=" * 70, "\n2. Temporal leakage in the PV normal split\n" + "=" * 70)
    pvds = Dataset("pv")
    holdout = pvds.split("holdout")
    ho = holdout.frame_ids
    trs = {f for f, split, _src in pvds.manifest() if split == "train"}
    adj = sum(any(adjacent_frame(p, d) in trs for d in (-1, 1)) for p in ho)
    print(f"held-out normals with an immediately adjacent frame in TRAIN: {adj} / {len(ho)}")

    print("\n" + "=" * 70, "\n3. Source shortcut (PV specialist)\n" + "=" * 70)
    pvn = np.load(RES / "pv_normal_errors.npy"); pvf = np.load(RES / "pv_fault_errors.npy")
    fu = np.array([pvf[gid == g].mean() for g in range(G)])
    src = holdout.source
    for s in ("om", "thm"):
        e = pvn[src == s]
        print(f"healthy {s:3s}: n={len(e):4d}  median err {np.median(e):.4f}  flagged at thr {(e > PV_THR).sum()}")
    print(f"AUROC healthy O&M vs healthy THM : {auroc(*from_groups(pvn[src == 'om'], pvn[src == 'thm'])):.3f}")
    print(f"AUROC healthy O&M vs unique fault: {auroc(*from_groups(pvn[src == 'om'], fu)):.3f}")
    print(f"AUROC healthy THM vs unique fault: {auroc(*from_groups(pvn[src == 'thm'], fu)):.3f}")

    print("\n" + "=" * 70, "\n4/5. Headline metrics, CIs, trivial baselines\n" + "=" * 70)
    gn = np.load(RES / "normal_errors.npy"); gf = np.load(RES / "fault_errors.npy")
    for name, n, f, thr in [("v2 general", gn, gf, V2_THR), ("PV, 1000 files", pvn, pvf, PV_THR),
                            ("PV, 140 unique", pvn, fu, PV_THR)]:
        s, y = from_groups(n, f)
        lo, hi = auroc_ci(s, y, seed=SEED)
        print(f"\n{name}: AUROC {auroc(s, y):.4f} [{lo:.3f}, {hi:.3f}]   flag-everything F1 = {trivial_f1(y):.4f}")
        print("  at saved threshold :", rounded(confusion(s, y, thr)))
        print("  re-tuned best-F1   :", rounded(confusion(s, y, best_f1_threshold(s, y))))

    print("\nComputing global image statistics (no model)...")
    pv_fault = pvds.split("fault")
    pvnF = global_features(holdout.x, pvds.stats)
    pvfF = global_features(pv_fault.x, pvds.stats)
    pvfF_u = np.array([pvfF[gid == g][0] for g in range(G)])
    v2ds = Dataset("v2")
    v2n, v2f = v2ds.split("test_normal"), v2ds.split("test_fault")
    gnF = global_features(v2n.x, v2ds.stats)
    gfF = global_features(v2f.x, v2ds.stats)
    for name, a, b in [("PV (unique faults)", pvnF, pvfF_u), ("v2 general", gnF, gfF)]:
        aj = [auroc(*from_groups(a[:, j], b[:, j])) for j in range(len(FEAT))]
        single = {k: max(v, 1 - v) for k, v in zip(FEAT, aj)}
        top = sorted(((k, float(v)) for k, v in single.items()), key=lambda kv: -kv[1])[:3]
        print(f"{name}: best single statistics {[(k, round(v, 3)) for k, v in top]}")
        print(f"{name}: 5-fold logistic regression on 16 statistics AUROC {logreg_cv_auroc(a, b):.3f}")

    config = RuleConfig.load()
    print("\n" + "=" * 70, f"\n6. Rule layer ({Path(config.source).relative_to(ROOT).as_posix()} defaults)\n" + "=" * 70)
    unique_first = [int(np.flatnonzero(gid == g)[0]) for g in range(G)]
    for name, imgs, e in [("PV healthy", denormalize(holdout.x, pvds.stats), pvn),
                          ("PV unique faults", denormalize(pv_fault.x[unique_first], pvds.stats), fu)]:
        tiers, final, _ = rule_counts(config, imgs, "pv", e > PV_THR)
        print(f"{name}: tiers {tiers}  ensemble {final}")
    ceiling = float(config.ambient_c) + float(config.equipment("transformer")["ceiling_over_ambient_c"])
    for s in ("transformer",):
        for lab, split, e in [("healthy", v2n, gn), ("fault", v2f, gf)]:
            keep = v2ds.frames(split, source=s)
            _, _, flagged = rule_counts(config, denormalize(split.x[keep], v2ds.stats), s, e[keep] > V2_THR)
            print(f"{s} {lab}: n={int(keep.sum())} AE flags {(e[keep] > V2_THR).sum()}  "
                  f"top-oil rule (>{ceiling:.0f} C) flags {flagged}")


if __name__ == "__main__":
    main()
