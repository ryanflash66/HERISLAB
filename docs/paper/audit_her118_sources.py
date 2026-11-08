"""
HER-118: per-source and cross-source (leave-one-source-out) validation of the PV specialist
(reviewer comments 4 and 12). Reads saved errors only; no model is trained or loaded here.

Sections:
  1. Source shift: what the two healthy sources are, and the THM PV-content screen
     (docs/data_manifests/thm_screened_frames.csv, HER-114: pv_content none / partial / major).
  2. Per-source table for the paper's PV specialist (bundle "pv", results/pv_*_errors.npy):
     each healthy source of the 892-frame holdout (O&M dr, O&M sr, THM) against the 140 unique
     PVMD faults: AUROC with stratified bootstrap 95% CI, median error, and false positives by
     source. Two thresholds are printed, labelled, because the paper's threshold is not chosen
     yet: 0.027865 (v1, best F1 tuned on the whole test set) and the HER-97 threshold the "pv"
     bundle now holds (best F1 on the validation half, frozen; counted on the test half).
  3. Leave-one-source-out.
     a. Train O&M, score unseen THM: the HER-114 O&M-only model (bundle "pv_om"), its saved
        errors on the 784-frame O&M holdout and the 1,000 PVMD files, and on all 1,075 THM
        frames (results/pv_om_thm_errors.npy, THM file order).
     b. Train THM, score unseen O&M: src/experiments/thm_only_pv.py (results/her-118/
        thm_only_errors.npz): THM TRAIN, a 108-frame contiguous THM holdout block, all 7,836
        O&M healthy frames, the 1,000 PVMD files.
     Threshold rule (one rule, both directions, fixed before looking at the unseen source):
     the 95th percentile of the in-source held-out healthy errors. It uses no fault and no
     unseen-source frame; by construction about 5% of in-source holdout frames are flagged.

Fault-side generalisation is not testable here: every fault image comes from one source (PVMD).

Usage:
    venv/Scripts/python.exe docs/paper/audit_her118_sources.py
Writes results/her-118/her118_output.txt
"""
import csv
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import SEED, saved_errors, source_labels, unique_fault_scores  # noqa: E402
from hlab.bundle import Bundle  # noqa: E402
from hlab.datasets import Dataset, pvmd_unique_groups  # noqa: E402
from hlab.metrics import auroc, auroc_ci, from_groups  # noqa: E402
from hlab.paths import MANIFESTS, RESULTS  # noqa: E402

OUT = RESULTS / "her-118"
V1_THRESHOLD = 0.027865   # the v1 paper threshold (results/her-114/rerun_summary.csv), test-tuned


def header(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def auc_ci(neg, pos):
    s, y = from_groups(neg, pos)
    lo, hi = auroc_ci(s, y, seed=SEED)
    return f"{auroc(s, y):.3f} [{lo:.3f}, {hi:.3f}]"


def flag(e, thr):
    k = int((np.asarray(e) > thr).sum())
    return f"{k} / {len(e)} ({k / len(e):.1%})"


def source_shift():
    header("1. Source shift")
    print("O&M (pv_om_inspection): two close-range inspection videos, dr = double-row (2,541 frames), "
          "sr = single-row (5,295 frames).")
    print("THM (pv_thermal_inspection): 1,075 frames of one fixed-wing aerial survey (senseFly eBee + ThermoMAP).")
    rows = list(csv.DictReader(open(MANIFESTS / "thm_screened_frames.csv", encoding="utf-8")))
    for key in ("pv_content", "hotspot"):
        c = {}
        for r in rows:
            c[r[key]] = c.get(r[key], 0) + 1
        print(f"THM screen (HER-114, thm_screened_frames.csv) {key}: {dict(sorted(c.items()))}  of {len(rows)}")
    return {r["frame"]: r["pv_content"] for r in rows}


def per_source(gid):
    header("2. Per-source table, paper PV specialist (bundle pv, 892 held-out healthy vs 140 unique faults)")
    ds = Dataset("pv")
    ho = ds.split("holdout")
    neg, pos = saved_errors("pv")
    fu = unique_fault_scores(pos, gid)
    seq = source_labels(ho)
    role = ds.calibration(ho)
    pos_role = ds.calibration("fault")
    fu_role = np.array([pos_role[int(np.flatnonzero(gid == g)[0])] for g in range(gid.max() + 1)], dtype=object)
    thr97 = Bundle.load("pv").require_threshold()
    print(f"thresholds: v1 {V1_THRESHOLD:.6f} (best F1 on the whole test set)   HER-97 {thr97:.6f} "
          f"(best F1 on the validation half, frozen)")
    print(f"{'source':8s} {'n':>4s} {'AUROC vs 140 unique faults [95% CI]':>36s} {'median err':>10s}  "
          f"{'FP at v1 thr (all)':>20s}  {'FP at HER-97 thr (test half)':>28s}")
    groups = [("om_dr", seq == "om_dr"), ("om_sr", seq == "om_sr"), ("om_all", seq != "thm"),
              ("thm", seq == "thm"), ("pooled", np.ones(len(seq), bool))]
    for name, m in groups:
        mt = m & (role == "test")
        print(f"{name:8s} {m.sum():4d} {auc_ci(neg[m], fu):>36s} {np.median(neg[m]):10.4f}  "
              f"{flag(neg[m], V1_THRESHOLD):>20s}  {flag(neg[mt], thr97):>28s}")
    print(f"PVMD     {len(fu):4d} unique faults: median err {np.median(fu):.4f}   flagged at v1 thr "
          f"{flag(fu, V1_THRESHOLD)}   at HER-97 thr (test half) {flag(fu[fu_role == 'test'], thr97)}")
    for lab, thr, m in (("v1", V1_THRESHOLD, np.ones(len(seq), bool)), ("HER-97", thr97, role == "test")):
        fp_thm = int(((neg > thr) & m & (seq == "thm")).sum())
        fp_all = int(((neg > thr) & m).sum())
        print(f"THM share of false positives at {lab} threshold: {fp_thm} / {fp_all}"
              f" = {fp_thm / fp_all:.1%}" if fp_all else f"no false positives at {lab} threshold")
    print(f"AUROC healthy O&M vs healthy THM (THM as the positive class): "
          f"{auc_ci(neg[seq != 'thm'], neg[seq == 'thm'])}")


def loso(name, in_ho, unseen, fu, extra=()):
    """One direction of leave-one-source-out; `extra` = [(label, unseen-mask), ...] sub-rows."""
    thr = float(np.percentile(in_ho, 95))
    print(f"threshold = 95th percentile of the {len(in_ho)} in-source held-out healthy errors: {thr:.6f}")
    print(f"AUROC in-source holdout vs 140 unique faults : {auc_ci(in_ho, fu)}")
    print(f"AUROC unseen-source healthy vs 140 unique faults: {auc_ci(unseen, fu)}")
    print(f"AUROC unseen-source healthy vs in-source holdout (unseen as positive): {auc_ci(in_ho, unseen)}")
    print(f"median error: in-source holdout {np.median(in_ho):.4f}   unseen source {np.median(unseen):.4f}   "
          f"unique faults {np.median(fu):.4f}")
    print(f"flagged at the threshold: in-source holdout {flag(in_ho, thr)}   unseen source {flag(unseen, thr)}   "
          f"unique faults {flag(fu, thr)}")
    for lab, m in extra:
        print(f"  unseen {lab:18s} flagged {flag(unseen[m], thr)}   median err {np.median(unseen[m]):.4f}")
    return thr


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print("HER-118: per-source and cross-source validation (saved errors only)")
        content = source_shift()
        _files, gid, _groups = pvmd_unique_groups()
        per_source(gid)

        header("3a. Leave-one-source-out: train O&M (HER-114 pv_om model), score unseen THM")
        om_neg, om_pos = saved_errors("pv_om")
        thm_err = np.load(RESULTS / "pv_om_thm_errors.npy")
        thm_rows = list(csv.DictReader(open(RESULTS / "pv_om_thm_errors.csv", encoding="utf-8")))
        if len(thm_rows) != len(thm_err) or len(thm_err) != 1075:
            raise RuntimeError("pv_om THM errors do not cover the 1,075 THM frames")
        pc = np.array([content[r["frame"]] for r in thm_rows])
        print(f"in-source: {len(om_neg)} O&M holdout frames (random-frame split, seed 42)   unseen: "
              f"{len(thm_err)} THM frames   faults: {len(om_pos)} files / {gid.max() + 1} unique")
        loso("om->thm", om_neg, thm_err, unique_fault_scores(om_pos, gid),
             extra=[(f"THM pv_content={k}", pc == k) for k in ("major", "partial", "none")])

        header("3b. Leave-one-source-out: train THM (src/experiments/thm_only_pv.py), score unseen O&M")
        z = np.load(OUT / "thm_only_errors.npz")
        role = z["thm_role"]
        om_seq = np.array([Path(str(f)).name.split("_")[0] for f in z["om_frames"]])
        print(f"THM train {(role == 'train').sum()}  gap {(role == 'gap').sum()}  in-source holdout "
              f"{(role == 'holdout').sum()} (last 10% of frame numbers, one contiguous block)   unseen O&M "
              f"{len(z['om_err'])}   faults {len(z['fault_err'])} files")
        log = (OUT / "training_log_pv_thm_only.txt").read_text(encoding="utf-8").strip().splitlines()
        print("training: " + " | ".join(l.strip() for l in log if "Best val_loss" in l or "Early" in l))
        loso("thm->om", z["thm_err"][role == "holdout"], z["om_err"], unique_fault_scores(z["fault_err"], gid),
             extra=[("O&M dr", om_seq == "dr"), ("O&M sr", om_seq == "sr")])
        print(f"(reference) THM TRAIN frames' own median error: {np.median(z['thm_err'][role == 'train']):.4f}")
    txt = buf.getvalue()
    (OUT / "her118_output.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
