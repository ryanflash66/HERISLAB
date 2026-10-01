"""
Evaluation-validity audit for the HERIS Lab thermal anomaly detectors.

Reproduces every number in the paper draft (Problem Statement, Method, Results) that is
not already printed by evaluate_autoencoder.py / evaluate_autoencoder_pv.py.
It never retrains or re-runs a model: it reads the saved reconstruction errors
in results/, the preprocessed arrays in data/CA_Preprocessed/, the PV split
manifest, and the raw PVMD files.

Checks:
  1. PVMD duplicates: pixel-identical files, and files whose pixels appear
     under more than one fault label
  2. Temporal leakage: held-out PV normals with an adjacent video frame in TRAIN
  3. Source shortcut: AE error split by acquisition source (O&M vs THM)
  4. Trivial baselines: single global intensity statistics and a 5-fold
     logistic regression on 16 global statistics (no model)
  5. Headline metrics with bootstrap 95% CIs, on the full and deduplicated set
  6. Rule layer: tier distribution and ensemble verdicts on the test sets

Dependencies: numpy, Pillow (already in the project venv). No sklearn needed.

Usage:
    venv\\Scripts\\python.exe docs\\paper\\audit_eval_validity.py
"""

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PRE = ROOT / "data" / "CA_Preprocessed"
RAW = ROOT / "data" / "CA_Training_Data"
RES = ROOT / "results"

PV_THR = 0.027865182608366013   # results/pv_eval_metrics.npy
V2_THR = 0.0003388182958588004  # results/eval_metrics.npy
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- metrics
def auroc(neg, pos):
    """Mann-Whitney AUROC with tie correction."""
    s = np.concatenate([neg, pos])
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    rp = ranks[len(neg):].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def auroc_ci(neg, pos, B=2000):
    vals = [auroc(RNG.choice(neg, len(neg)), RNG.choice(pos, len(pos))) for _ in range(B)]
    return np.percentile(vals, [2.5, 97.5])


def confusion(neg, pos, thr):
    tp = int((pos > thr).sum()); fn = len(pos) - tp
    fp = int((neg > thr).sum()); tn = len(neg) - fp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / len(pos)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, precision=round(p, 4), recall=round(r, 4),
                f1=round(f1, 4), specificity=round(tn / len(neg), 4))


def best_f1_threshold(neg, pos):
    s = np.concatenate([neg, pos])
    best = (0.0, None)
    for t in np.linspace(s.min(), s.max(), 1000):
        f1 = confusion(neg, pos, t)["f1"]
        if f1 > best[0]:
            best = (f1, t)
    return best[1]


def logreg_cv_auroc(X_neg, X_pos, k=5, l2=1.0, iters=3000, lr=0.1):
    """5-fold CV logistic regression (standardized, L2), numpy only."""
    X = np.vstack([X_neg, X_pos]); y = np.r_[np.zeros(len(X_neg)), np.ones(len(X_pos))]
    idx_n = RNG.permutation(len(X_neg)); idx_p = RNG.permutation(len(X_pos)) + len(X_neg)
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
    return auroc(out[y == 0], out[y == 1])


# ---------------------------------------------------------------- features
FEAT = ["mean", "std", "min", "max", "p1", "p5", "p50", "p95", "p99", "p95_minus_mean",
        "frac_ge250", "frac_le5", "grad_x", "grad_y", "laplacian", "hist_entropy"]


def global_features(arr_path, stats_path):
    st = np.load(stats_path, allow_pickle=True).item()
    A = np.load(arr_path, mmap_mode="r")
    rows = []
    for i in range(A.shape[0]):
        r = np.asarray(A[i], dtype=np.float32) * st["std"] + st["mean"]  # back to 0-255
        p = np.percentile(r, [1, 5, 50, 95, 99])
        h, _ = np.histogram(np.clip(r, 0, 255), bins=32, range=(0, 255)); h = h / h.sum()
        lap = np.abs(4 * r[1:-1, 1:-1] - r[:-2, 1:-1] - r[2:, 1:-1] - r[1:-1, :-2] - r[1:-1, 2:]).mean()
        rows.append([r.mean(), r.std(), r.min(), r.max(), *p, p[3] - r.mean(),
                     (r >= 250).mean(), (r <= 5).mean(),
                     np.abs(np.diff(r, axis=1)).mean(), np.abs(np.diff(r, axis=0)).mean(),
                     lap, -(h[h > 0] * np.log2(h[h > 0])).sum()])
    return np.array(rows)


def read_manifest():
    tr, ho, sec = [], [], None
    for line in (PRE / "pv" / "split_manifest.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line == "[TRAIN]": sec = tr; continue
        if line == "[HOLDOUT]": sec = ho; continue
        if line and not line.startswith("#") and sec is not None:
            sec.append(line)
    return tr, ho


def main():
    print("=" * 70, "\n1. PVMD duplicates\n" + "=" * 70)
    fault_dir = RAW / "test" / "fault" / "pv"
    files = sorted(p.name for p in fault_dir.iterdir() if p.is_file())  # same order as preprocess_pv.py
    pix = {f: hashlib.md5(np.array(Image.open(fault_dir / f)).tobytes()).hexdigest() for f in files}
    groups = defaultdict(list)
    for f, h in pix.items():
        groups[h].append(f)
    multi = sum(len(v) for v in groups.values() if len({x[0].upper() for x in v}) > 1)
    print(f"files: {len(files)}   pixel-unique images: {len(groups)}")
    print(f"files whose pixels also appear under another fault label: {multi}")
    print("label mix of unique images:", dict(Counter("".join(sorted({x[0].upper() for x in v})) for v in groups.values())))
    # PVMD stores some rotation augmentations only as an EXIF orientation tag, which the
    # preprocessing ignores. Recount with the tag applied so both views are reported.
    from PIL import ImageOps
    exif_tag = {f: Image.open(fault_dir / f).getexif().get(274, 1) for f in files}
    pix_t = {f: hashlib.md5(np.array(ImageOps.exif_transpose(Image.open(fault_dir / f))).tobytes()).hexdigest()
             for f in files}
    groups_t = defaultdict(list)
    for f, h in pix_t.items():
        groups_t[h].append(f)
    multi_t = sum(len(v) for v in groups_t.values() if len({x[0].upper() for x in v}) > 1)
    print(f"files with a non-default EXIF orientation tag: {sum(1 for v in exif_tag.values() if v != 1)}")
    print(f"unique images with EXIF orientation applied: {len(groups_t)}   "
          f"files sharing an image with another label: {multi_t}")
    print("label mix with EXIF applied:", dict(Counter("".join(sorted({x[0].upper() for x in v})) for v in groups_t.values())))
    gid_of = {h: i for i, h in enumerate(groups)}
    gid = np.array([gid_of[pix[f]] for f in files])
    G = len(groups)

    print("\n" + "=" * 70, "\n2. Temporal leakage in the PV normal split\n" + "=" * 70)
    tr, ho = read_manifest()
    trs = set(tr)

    def neighbour(p, d):
        m = re.search(r"(\d+)(\.tiff?)$", p)
        return p[:m.start(1)] + str(int(m.group(1)) + d).zfill(len(m.group(1))) + m.group(2)
    adj = sum(any(neighbour(p, d) in trs for d in (-1, 1)) for p in ho)
    print(f"held-out normals with an immediately adjacent frame in TRAIN: {adj} / {len(ho)}")

    print("\n" + "=" * 70, "\n3. Source shortcut (PV specialist)\n" + "=" * 70)
    pvn = np.load(RES / "pv_normal_errors.npy"); pvf = np.load(RES / "pv_fault_errors.npy")
    fu = np.array([pvf[gid == g].mean() for g in range(G)])
    src = np.array(["thm" if "THM_" in p else "om" for p in ho])
    for s in ("om", "thm"):
        e = pvn[src == s]
        print(f"healthy {s:3s}: n={len(e):4d}  median err {np.median(e):.4f}  flagged at thr {(e > PV_THR).sum()}")
    print(f"AUROC healthy O&M vs healthy THM : {auroc(pvn[src == 'om'], pvn[src == 'thm']):.3f}")
    print(f"AUROC healthy O&M vs unique fault: {auroc(pvn[src == 'om'], fu):.3f}")
    print(f"AUROC healthy THM vs unique fault: {auroc(pvn[src == 'thm'], fu):.3f}")

    print("\n" + "=" * 70, "\n4/5. Headline metrics, CIs, trivial baselines\n" + "=" * 70)
    gn = np.load(RES / "normal_errors.npy"); gf = np.load(RES / "fault_errors.npy")
    for name, n, f, thr in [("v2 general", gn, gf, V2_THR), ("PV, 1000 files", pvn, pvf, PV_THR),
                            ("PV, 140 unique", pvn, fu, PV_THR)]:
        lo, hi = auroc_ci(n, f)
        prev = len(f) / (len(n) + len(f)); triv = 2 * prev / (1 + prev)
        print(f"\n{name}: AUROC {auroc(n, f):.4f} [{lo:.3f}, {hi:.3f}]   flag-everything F1 = {triv:.4f}")
        print("  at saved threshold :", confusion(n, f, thr))
        print("  re-tuned best-F1   :", confusion(n, f, best_f1_threshold(n, f)))

    print("\nComputing global image statistics (no model)...")
    pvnF = global_features(PRE / "pv" / "test_normal.npy", PRE / "pv" / "norm_stats.npy")
    pvfF = global_features(PRE / "pv" / "test_fault.npy", PRE / "pv" / "norm_stats.npy")
    pvfF_u = np.array([pvfF[gid == g][0] for g in range(G)])
    gnF = global_features(PRE / "test_normal.npy", PRE / "norm_stats.npy")
    gfF = global_features(PRE / "test_fault.npy", PRE / "norm_stats.npy")
    for name, a, b in [("PV (unique faults)", pvnF, pvfF_u), ("v2 general", gnF, gfF)]:
        single = {k: max(auroc(a[:, j], b[:, j]), 1 - auroc(a[:, j], b[:, j])) for j, k in enumerate(FEAT)}
        top = sorted(((k, float(v)) for k, v in single.items()), key=lambda kv: -kv[1])[:3]
        print(f"{name}: best single statistics {[(k, round(v, 3)) for k, v in top]}")
        print(f"{name}: 5-fold logistic regression on 16 statistics AUROC {logreg_cv_auroc(a, b):.3f}")

    print("\n" + "=" * 70, "\n6. Rule layer (config/thresholds_iec.json defaults)\n" + "=" * 70)
    im, i95, imax = FEAT.index("mean"), FEAT.index("p95"), FEAT.index("max")
    for name, F, e in [("PV healthy", pvnF, pvn), ("PV unique faults", pvfF_u, fu)]:
        d = (F[:, i95] - F[:, im]) * 100 / 255
        tier = np.select([d < 0, d < 15, d < 40, d < 75], ["OUT_OF_RANGE", "NORMAL", "WARNING", "ALARM"], "CRITICAL")
        ml = e > PV_THR; rule = np.isin(tier, ["ALARM", "CRITICAL"])
        final = np.where(ml | rule, "FLAG", np.where((tier == "WARNING") & ~ml, "MONITOR", "NORMAL"))
        print(f"{name}: tiers {dict(Counter(tier.tolist()))}  ensemble {dict(Counter(final.tolist()))}")
    for s, n_idx, f_idx in [("transformer", slice(33, 37), slice(452, 685))]:
        for lab, F, e in [("healthy", gnF[n_idx], gn[n_idx]), ("fault", gfF[f_idx], gf[f_idx])]:
            oil = 20 + F[:, imax] / 255 * 100 + 10
            print(f"{s} {lab}: n={len(F)} AE flags {(e > V2_THR).sum()}  top-oil rule (>120 C) flags {(oil > 120).sum()}")


if __name__ == "__main__":
    main()
