"""
HER-114 retrain: validity controls for the O&M-only PV specialist.

Runs the same controls as audit_her114_reruns.py (metric code imported
unchanged from audit_eval_validity.py) on the errors of the model trained
WITHOUT the THM aerial-survey frames:

  split      data/CA_Preprocessed/pv_om/split_manifest.txt   (seed 42, O&M only)
  checkpoint models/autoencoder_pv_om_best.pth               (seed 43)
  errors     results/pv_om_normal_errors.npy, results/pv_om_fault_errors.npy
  threshold  re-tuned best-F1 threshold from results/pv_om_eval_metrics.npy

Reported: AUROC with bootstrap 95% CI (1,000 files and 140 unique faults),
confusion at the re-tuned threshold, flag-everything F1, single-statistic
baselines incl. p1, 16-statistic logistic regression, adjacent-frame leakage,
source AUROC (O&M sequence dr vs sr), THM flag rate of the new model.

The THM+O&M rows (scenarios a and b) are read from results/her-114/rerun_summary.csv
so the three can be shown side by side.

Usage:
    venv/Scripts/python.exe docs/paper/audit_her114_retrain.py
Writes results/her-114_retrain/audit_retrain_om_only.txt, summary.md, summary.csv
"""
import csv
import hashlib
import io
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_eval_validity as A  # noqa: E402
import audit_her114_reruns as R  # noqa: E402

ROOT = A.ROOT
PRE_OM = A.PRE / "pv_om"
OUT = ROOT / "results" / "her-114_retrain"
PREV = ROOT / "results" / "her-114" / "rerun_summary.csv"


def read_manifest(path):
    tr, ho, sec = [], [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line == "[TRAIN]": sec = tr; continue
        if line == "[HOLDOUT]": sec = ho; continue
        if line and not line.startswith("#") and sec is not None:
            sec.append(line)
    return tr, ho


def load_inputs():
    fault_dir = A.RAW / "test" / "fault" / "pv"
    files = sorted(p.name for p in fault_dir.iterdir() if p.is_file())
    pix = {f: hashlib.md5(np.array(Image.open(fault_dir / f)).tobytes()).hexdigest() for f in files}
    groups = defaultdict(list)
    for f, h in pix.items():
        groups[h].append(f)
    gid_of = {h: i for i, h in enumerate(groups)}
    gid = np.array([gid_of[pix[f]] for f in files])
    G = len(groups)
    tr, ho = read_manifest(PRE_OM / "split_manifest.txt")
    assert not any("THM_" in p for p in tr + ho), "THM frame found in the O&M-only split"
    pvn = np.load(A.RES / "pv_om_normal_errors.npy")
    pvf = np.load(A.RES / "pv_om_fault_errors.npy")
    assert len(pvn) == len(ho) and len(pvf) == len(files) == 1000, (len(pvn), len(ho), len(pvf))
    fu = np.array([pvf[gid == g].mean() for g in range(G)])
    print(f"O&M-only split: {len(tr)} train / {len(ho)} holdout; faults {len(files)} files / {G} unique")
    print("Computing global image statistics (no model)...")
    pvnF = A.global_features(PRE_OM / "test_normal.npy", PRE_OM / "norm_stats.npy")
    pvfF = A.global_features(PRE_OM / "test_fault.npy", PRE_OM / "norm_stats.npy")
    pvfF_u = np.array([pvfF[gid == g][0] for g in range(G)])
    return dict(tr=tr, ho=ho, pvn=pvn, pvf=pvf, fu=fu, pvnF=pvnF, pvfF_u=pvfF_u)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    M = np.load(A.RES / "pv_om_eval_metrics.npy", allow_pickle=True).item()
    A.PV_THR = float(M["threshold"])  # run_scenario reads the threshold from here
    buf = io.StringIO()
    with redirect_stdout(buf):
        print("=" * 70, "\nHER-114 retrain: O&M-only PV specialist\n" + "=" * 70)
        print(f"threshold used below ('saved threshold' in the printout) = re-tuned best-F1 "
              f"threshold of the O&M-only model: {A.PV_THR:.6f}")
        D = load_inputs()
        S = R.run_scenario("retrain_om_only", np.ones(len(D["ho"]), bool), D)
        c = A.confusion(D["pvn"], D["pvf"], A.PV_THR)
        print(f"\nre-tuned threshold {A.PV_THR:.6f}: {c}")
        print(f"THM flag rate at that threshold: all 1,075 frames {M['thm_flag_rate_all']:.1%}; "
              f"the 84 frames held out by the original split {M['thm_flag_rate_holdout84']:.1%}")
    txt = buf.getvalue()
    (OUT / "audit_retrain_om_only.txt").write_text(txt, encoding="utf-8")
    print(txt)

    S["threshold"] = round(A.PV_THR, 6)
    S["thm_flag_rate_holdout84"] = f"{M['thm_flag_rate_holdout84']:.1%}"
    S["thm_flag_rate_all"] = f"{M['thm_flag_rate_all']:.1%}"
    S["model"] = "O&M-only retrain (seed 43), 7,052 O&M train frames"

    rows = []
    with PREV.open(encoding="utf-8") as fh:
        prev = {r["scenario"]: r for r in csv.DictReader(fh)}
    for key, model in [("a_as_is", "THM+O&M model, holdout as is"),
                       ("b_thm_removed", "THM+O&M model, THM dropped at evaluation")]:
        r = dict(prev[key]); r["model"] = model; r["threshold"] = "0.027865"
        r["thm_flag_rate_holdout84"] = "96.4% (81/84)" if key == "a_as_is" else "n/a (not scored)"
        r["thm_flag_rate_all"] = "n/a"
        rows.append(r)
    rows.append(S)

    cols = ["model", "scenario", "n_healthy", "n_thm", "threshold", "auroc_files", "auroc_files_ci",
            "auroc_unique", "auroc_unique_ci", "fp_files", "recall_files", "precision_files", "f1_saved_files",
            "f1_trivial_files", "f1_trivial_unique", "auroc_source", "auroc_source_def", "p1_baseline",
            "best_single", "logreg16", "leak_adjacent", "thm_flag_rate_holdout84", "thm_flag_rate_all"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print("\n".join(md))


if __name__ == "__main__":
    main()
