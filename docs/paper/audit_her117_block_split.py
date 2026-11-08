"""
HER-117: the PV evaluation repeated on a temporally separated split (reviewer comment 2).

Reads only saved errors; never trains or loads weights. Compares three models of the same recipe
(config/training.json "pv"), each on its own held-out healthy frames and the same 1,000 PVMD fault
files (140 pixel-unique images):

  paper model      random-frame split (data/CA_Preprocessed/pv/split_manifest.txt), results/pv_*_errors.npy
  random, seed 0   the same split, torch seed 0 (HER-120, results/her-120/pv_seed0_*)
  block, seed 0    the block split of src/experiments/block_split_pv.py (docs/data_manifests/pv_block_split.csv),
                   torch seed 0, results/her-117/pv_block_seed0_*

Per model: the temporal-leakage check under both neighbour definitions (a TRAIN frame immediately
adjacent, i.e. 1 frame away; a TRAIN frame at most 2 frames away) and the distance to the nearest TRAIN
frame of the same recording; threshold-free AUROC (1,000 files and 140 unique faults) and AUPRC with
95% stratified bootstrap intervals (2,000 resamples, seed 0, unique fault as the fault unit); precision,
recall and F1 on the TEST half at the threshold chosen as the best-F1 point of the VAL half with one
score per unique fault (HER-97 protocol); AUROC by recording; no-model baselines on the same frames.

Which threshold the paper uses (0.0279 or 0.0762) is not decided here: every model below takes its own
VAL-half threshold, so the rows are comparable to each other, not to a printed paper threshold.

Usage:
    venv/Scripts/python.exe docs/paper/audit_her117_block_split.py
Writes results/her-117/her117_output.txt
"""
import csv
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit  # noqa: E402
from audit import (SEED, check_pvmd_duplicates, check_trivial_baselines, check_unique_intervals,  # noqa: E402
                   global_features, saved_errors, unique_fault_scores, unique_first)
from hlab.bundle import Bundle  # noqa: E402
from hlab.datasets import Dataset  # noqa: E402
from hlab.metrics import auroc, auroc_ci, best_f1_threshold, confusion, from_groups, trivial_f1  # noqa: E402
from hlab.paths import MANIFESTS, RESULTS  # noqa: E402
from hlab.transform import denormalize, normalize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "experiments"))
from block_split_pv import frame_key, read_split  # noqa: E402

OUT = RESULTS / "her-117"
BLOCK = "pv_block_seed0"


def leakage(holdout_ids, train_ids):
    """(n with a TRAIN frame 1 away, n with one at most 2 away, min / median distance to the nearest TRAIN frame)."""
    by_rec = {}
    for f in train_ids:
        r, n = frame_key(f)
        by_rec.setdefault(r, []).append(n)
    by_rec = {r: np.sort(v) for r, v in by_rec.items()}
    d = []
    for f in holdout_ids:
        r, n = frame_key(f)
        v = by_rec[r]
        i = np.searchsorted(v, n)
        d.append(min(abs(n - v[j]) for j in (i - 1, i) if 0 <= j < len(v)))
    d = np.array(d)
    return int((d <= 1).sum()), int((d <= 2).sum()), int(d.min()), float(np.median(d))


def evaluate(label, neg, pos, gid, ho_ids, tr_ids, neg_role, pos_role, negF, posF_u):
    fu = unique_fault_scores(pos, gid)
    fu_role = np.array([pos_role[i] for i in unique_first(gid)], dtype=object)
    rec = np.array([frame_key(f)[0] for f in ho_ids])
    a1, a2, dmin, dmed = leakage(ho_ids, tr_ids)
    R = dict(model=label, n_healthy=len(neg), n_train=len(tr_ids), leak_adjacent=a1, leak_within2=a2,
             min_train_distance=dmin, median_train_distance=dmed)
    s, y = from_groups(neg, pos)
    R["auroc_files"] = auroc(s, y); R["auroc_files_ci"] = auroc_ci(s, y, seed=SEED)
    iv = check_unique_intervals(neg, fu, seed=SEED)
    R["auroc_unique"], R["auroc_unique_ci"] = iv["auroc"][0], iv["auroc"][1:]
    R["auprc_unique"], R["auprc_unique_ci"] = iv["auprc"][0], iv["auprc"][1:]
    hv, ht = neg_role == "val", neg_role == "test"
    fv, ft = fu_role == "val", fu_role == "test"
    thr = best_f1_threshold(*from_groups(neg[hv], fu[fv]))
    st, yt = from_groups(neg[ht], fu[ft])
    c = confusion(st, yt, thr)
    R.update(threshold=thr, n_test=(int(ht.sum()), int(ft.sum())), test_auroc=auroc(st, yt),
             precision=c["precision"], recall=c["recall"], f1=c["f1"], fp=c["fp"], fn=c["fn"],
             test_trivial_f1=trivial_f1(yt))
    ivt = check_unique_intervals(neg[ht], fu[ft], thr=thr, seed=SEED)
    for k in ("precision", "recall", "f1"):
        R[f"{k}_ci"] = ivt[k][1:]
    R["by_recording"] = {r: (int((rec == r).sum()), auroc(*from_groups(neg[rec == r], fu)),
                             int(((neg > thr) & (rec == r) & ht).sum()), int(((rec == r) & ht).sum()))
                         for r in ("dr", "sr", "thm")}
    base = check_trivial_baselines(negF, posF_u, seed=SEED)
    R["best_single"] = base["top"][0]
    R["logreg16"] = base["logreg"]
    R["trivial_f1_all"] = trivial_f1(from_groups(neg, fu)[1])
    return R


def ci(v):
    return f"[{v[0]:.3f}, {v[1]:.3f}]"


def report(R):
    print(f"\n--- {R['model']} ---")
    print(f"healthy held out {R['n_healthy']}   TRAIN {R['n_train']}")
    print(f"leakage: TRAIN frame immediately adjacent (1 away) {R['leak_adjacent']} / {R['n_healthy']};  "
          f"TRAIN frame at most 2 away {R['leak_within2']} / {R['n_healthy']};  "
          f"nearest TRAIN frame min {R['min_train_distance']}, median {R['median_train_distance']:.0f} frames")
    print(f"AUROC, every image, 1,000 files  {R['auroc_files']:.4f} {ci(R['auroc_files_ci'])}")
    print(f"AUROC, every image, 140 unique   {R['auroc_unique']:.4f} {ci(R['auroc_unique_ci'])}")
    print(f"AUPRC, every image, 140 unique   {R['auprc_unique']:.4f} {ci(R['auprc_unique_ci'])}")
    print(f"threshold: best F1 on the VAL half (one score per unique fault) = {R['threshold']:.6f}")
    print(f"TEST half ({R['n_test'][0]} healthy, {R['n_test'][1]} unique faults): AUROC {R['test_auroc']:.4f}  "
          f"precision {R['precision']:.4f} {ci(R['precision_ci'])}  recall {R['recall']:.4f} {ci(R['recall_ci'])}  "
          f"F1 {R['f1']:.4f} {ci(R['f1_ci'])}  FP {R['fp']}  FN {R['fn']}   flag-everything F1 {R['test_trivial_f1']:.4f}")
    for r, (n, a, fl, nt) in R["by_recording"].items():
        print(f"  {r:3s}: n={n:4d}  AUROC vs 140 unique {a:.4f}   test half flagged {fl} / {nt}")
    print(f"no-model baselines on the same frames: best single statistic {R['best_single'][0]} "
          f"{R['best_single'][1]:.3f}   16-statistic logistic regression {R['logreg16']:.3f}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pv = Dataset("pv")
    tr, ho, fa = pv.split("train"), pv.split("holdout"), pv.split("fault")
    gid = check_pvmd_duplicates()["gid"]
    pos_role = pv.calibration("fault")
    first = unique_first(gid)
    posF_old = global_features(fa.x, pv.stats)[first]
    negF_old = global_features(ho.x, pv.stats)
    neg_role_old = pv.calibration("holdout")

    split = read_split()
    ids = list(tr.frame_ids) + list(ho.frame_ids)
    block_ids = [f.strip() for f in (OUT / f"{BLOCK}_holdout_ids.txt").read_text(encoding="utf-8").splitlines() if f.strip()]
    block_tr = [f for f, v in split.items() if v[2] == "train"]
    assert block_ids == [f for f in ids if split[f][2] == "holdout"], "holdout order differs from the split manifest"
    stats = np.load(OUT / "norm_stats_pv_block.npy", allow_pickle=True).item()
    row = {f: i for i, f in enumerate(ids)}
    sel = np.array([row[f] for f in block_ids])
    x_raw = np.concatenate([tr.x, ho.x])[sel]
    negF_blk = global_features(normalize(denormalize(x_raw, pv.stats), stats), stats)
    posF_blk = global_features(normalize(denormalize(fa.x, pv.stats), stats), stats)[first]
    neg_role_blk = np.array([split[f][3] for f in block_ids], dtype=object)

    rows = []
    buf = io.StringIO()
    with redirect_stdout(buf):
        print("=" * 70 + "\nHER-117: PV specialist on a temporally separated split\n" + "=" * 70)
        counts = {}
        for rec, _n, sp, _r in split.values():
            counts.setdefault(rec, {}).setdefault(sp, 0)
            counts[rec][sp] += 1
        print("block split (docs/data_manifests/pv_block_split.csv): one held-out block of 10% per ~900-frame "
              "segment of each recording, 100 frames dropped on each side")
        for rec, c in sorted(counts.items()):
            print(f"  {rec:3s}: train {c.get('train', 0):5d}  holdout {c.get('holdout', 0):4d}  gap (unused) {c.get('gap', 0):5d}")
        neg_p, pos_p = saved_errors(Bundle.load("pv"))
        rows.append(evaluate("paper model, random-frame split", neg_p, pos_p, gid, ho.frame_ids, tr.frame_ids,
                             neg_role_old, pos_role, negF_old, posF_old))
        q = RESULTS / "her-120" / "pv_seed0_normal_errors.npy"
        if q.exists():
            rows.append(evaluate("random-frame split, torch seed 0 (HER-120)", np.load(q),
                                 np.load(q.with_name("pv_seed0_fault_errors.npy")), gid, ho.frame_ids, tr.frame_ids,
                                 neg_role_old, pos_role, negF_old, posF_old))
        rows.append(evaluate("block split, torch seed 0 (HER-117)", np.load(OUT / f"{BLOCK}_normal_errors.npy"),
                             np.load(OUT / f"{BLOCK}_fault_errors.npy"), gid, block_ids, block_tr,
                             neg_role_blk, pos_role, negF_blk, posF_blk))
        for R in rows:
            report(R)
        print("\nNotes: the block model trains on 6,023 frames instead of 8,019 (gap frames are unused). "
              "Faults are the same 1,000 PVMD files for every model; their VAL / TEST halves come from "
              "docs/data_manifests/pv_calibration_split.csv. Fault-side generalisation is not tested: every fault is "
              "from PVMD (HER-95).")
    txt = buf.getvalue()
    (OUT / "her117_output.txt").write_text(txt, encoding="utf-8")
    print(txt)
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        keys = [k for k in rows[0] if k != "by_recording"]
        w = csv.writer(fh)
        w.writerow(keys)
        for R in rows:
            w.writerow([R[k] for k in keys])


if __name__ == "__main__":
    main()
