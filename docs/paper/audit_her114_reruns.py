"""
HER-114: rerun the evaluation-validity audit with the THM "healthy" frames
removed or screened.

Nothing is retrained and the split is untouched: the PV specialist checkpoint,
the seed-42 split (src/preprocess_pv.py) and the seed-43 training run are the
same ones behind results/pv_*_errors.npy. Each scenario only changes WHICH
held-out healthy frames are counted at evaluation time:

  a  as is                        892 healthy (808 O&M + 84 THM)
  b  THM removed                  808 O&M only
  c  flagged THM removed          O&M + THM frames whose flag is not "drop"
                                  (drop = no PV modules in frame, or a visually
                                  confirmed hot-spot candidate; see
                                  docs/data_manifests/thm_screened_frames.csv)
  c2 hot-spot THM removed         O&M + THM frames without a confirmed hot-spot
                                  candidate (no-PV frames kept)
  c3 strict                       O&M + THM frames flagged "keep" only (array
                                  fills the frame, no hot-spot candidate)

Reported per scenario: AUROC (1,000 files and 140 unique faults, bootstrap
95% CI), confusion at the saved threshold, false-positive share by source,
source AUROC (O&M vs THM; for b, O&M sequence dr vs sr), flag-everything F1,
single-statistic baselines incl. the 1st-percentile one, the 16-statistic
logistic regression, and temporal leakage of the retained healthy frames.

Usage:
    venv\\Scripts\\python.exe docs\\paper\\audit_her114_reruns.py
Writes results/her-114/rerun_<scenario>.txt and results/her-114/rerun_summary.md
"""
import csv
import hashlib
import io
import re
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_eval_validity as A  # noqa: E402

ROOT = A.ROOT
OUT = ROOT / "results" / "her-114"
SCREEN = ROOT / "docs" / "data_manifests" / "thm_screened_frames.csv"


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
    tr, ho = A.read_manifest()
    pvn = np.load(A.RES / "pv_normal_errors.npy")
    pvf = np.load(A.RES / "pv_fault_errors.npy")
    assert len(pvn) == len(ho) == 892 and len(pvf) == len(files) == 1000
    fu = np.array([pvf[gid == g].mean() for g in range(G)])
    print("Computing global image statistics (no model)...")
    pvnF = A.global_features(A.PRE / "pv" / "test_normal.npy", A.PRE / "pv" / "norm_stats.npy")
    pvfF = A.global_features(A.PRE / "pv" / "test_fault.npy", A.PRE / "pv" / "norm_stats.npy")
    pvfF_u = np.array([pvfF[gid == g][0] for g in range(G)])
    return dict(tr=tr, ho=ho, pvn=pvn, pvf=pvf, fu=fu, pvnF=pvnF, pvfF_u=pvfF_u)


def neighbour(p, d):
    m = re.search(r"(\d+)(\.tiff?)$", p)
    return p[:m.start(1)] + str(int(m.group(1)) + d).zfill(len(m.group(1))) + m.group(2)


def run_scenario(name, keep, D):
    """keep: boolean mask over the 892 held-out healthy frames."""
    A.RNG = np.random.default_rng(0)
    ho = [p for p, k in zip(D["ho"], keep) if k]
    n = D["pvn"][keep]
    nF = D["pvnF"][keep]
    src = np.array(["thm" if "THM_" in p else ("om_dr" if "/dr_frame" in p else "om_sr") for p in ho])
    S = {"scenario": name, "n_healthy": len(n), "n_thm": int((src == "thm").sum()),
         "n_om": int((src != "thm").sum())}
    print(f"healthy frames kept: {len(n)}  (O&M {S['n_om']}, THM {S['n_thm']})")

    trs = set(D["tr"])
    adj = sum(any(neighbour(p, d) in trs for d in (-1, 1)) for p in ho)
    S["leak_adjacent"] = adj
    print(f"held-out healthy with an adjacent frame in TRAIN: {adj} / {len(ho)}")

    for lab, f in [("1000 files", D["pvf"]), ("140 unique", D["fu"])]:
        au = A.auroc(n, f); lo, hi = A.auroc_ci(n, f)
        prev = len(f) / (len(n) + len(f)); triv = 2 * prev / (1 + prev)
        c = A.confusion(n, f, A.PV_THR)
        print(f"\n{lab}: AUROC {au:.4f} [{lo:.3f}, {hi:.3f}]  flag-everything F1 {triv:.4f}")
        print("  at saved threshold:", c)
        key = "files" if lab == "1000 files" else "unique"
        S[f"auroc_{key}"] = round(au, 4); S[f"auroc_{key}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        S[f"f1_trivial_{key}"] = round(triv, 4); S[f"f1_saved_{key}"] = c["f1"]
        S[f"fp_{key}"] = c["fp"]; S[f"recall_{key}"] = c["recall"]; S[f"precision_{key}"] = c["precision"]

    flagged = n > A.PV_THR
    fp_thm = int((flagged & (src == "thm")).sum()); fp_om = int((flagged & (src != "thm")).sum())
    S["fp_thm"] = fp_thm; S["fp_om"] = fp_om
    S["fp_share_thm"] = round(fp_thm / max(fp_thm + fp_om, 1), 3)
    print(f"\nfalse positives at saved threshold: THM {fp_thm} / {S['n_thm']}   O&M {fp_om} / {S['n_om']}"
          f"   THM share {S['fp_share_thm']:.1%}")
    for s in ("om_dr", "om_sr", "thm"):
        e = n[src == s]
        if len(e):
            print(f"  {s:5s}: n={len(e):4d}  median err {np.median(e):.4f}  flagged {(e > A.PV_THR).sum()}")

    om = n[src != "thm"]; thm = n[src == "thm"]
    if len(thm):
        S["auroc_source"] = round(A.auroc(om, thm), 3); S["auroc_source_def"] = "healthy O&M vs healthy THM"
    else:
        S["auroc_source"] = round(A.auroc(n[src == "om_dr"], n[src == "om_sr"]), 3)
        S["auroc_source_def"] = "healthy O&M sequence dr vs sr"
    print(f"source AUROC ({S['auroc_source_def']}): {S['auroc_source']}")
    print(f"AUROC healthy O&M vs unique fault: {A.auroc(om, D['fu']):.3f}")
    if len(thm):
        print(f"AUROC healthy THM vs unique fault: {A.auroc(thm, D['fu']):.3f}")

    single = {k: max(A.auroc(nF[:, j], D["pvfF_u"][:, j]), 1 - A.auroc(nF[:, j], D["pvfF_u"][:, j]))
              for j, k in enumerate(A.FEAT)}
    top = sorted(single.items(), key=lambda kv: -kv[1])[:3]
    S["p1_baseline"] = round(float(single["p1"]), 3)
    S["best_single"] = f"{top[0][0]} {top[0][1]:.3f}"
    print(f"single-statistic AUROC (unique faults): p1 = {single['p1']:.3f}; best {[(k, round(v, 3)) for k, v in top]}")
    lr = A.logreg_cv_auroc(nF, D["pvfF_u"])
    S["logreg16"] = round(float(lr), 3)
    print(f"5-fold logistic regression on 16 statistics: AUROC {lr:.3f}")
    return S


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = load_inputs()
    ho_names = [Path(p).name for p in D["ho"]]
    is_thm = np.array(["THM_" in p for p in D["ho"]])
    screen = {r["frame"]: r for r in csv.DictReader(SCREEN.open(encoding="utf-8"))}
    flag = np.array([screen[f]["flag"] if f in screen else "n/a" for f in ho_names])
    hot = np.array([(screen[f]["hotspot"] == "yes") if f in screen else False for f in ho_names])
    scenarios = [
        ("a_as_is", np.ones(len(ho_names), bool)),
        ("b_thm_removed", ~is_thm),
        ("c_flagged_thm_removed", ~is_thm | (flag != "drop")),
        ("c2_hotspot_thm_removed", ~is_thm | ~hot),
        ("c3_strict_keep_only", ~is_thm | (flag == "keep")),
    ]
    rows = []
    for name, keep in scenarios:
        buf = io.StringIO()
        with redirect_stdout(buf):
            print("=" * 70, f"\nHER-114 rerun {name}\n" + "=" * 70)
            S = run_scenario(name, keep, D)
        (OUT / f"rerun_{name}.txt").write_text(buf.getvalue(), encoding="utf-8")
        print(buf.getvalue())
        rows.append(S)

    cols = ["scenario", "n_healthy", "n_thm", "auroc_files", "auroc_files_ci", "auroc_unique", "auroc_unique_ci",
            "fp_files", "fp_thm", "fp_share_thm", "f1_saved_files", "f1_trivial_files", "f1_trivial_unique",
            "auroc_source", "auroc_source_def", "p1_baseline", "best_single", "logreg16", "leak_adjacent"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for S in rows:
        md.append("| " + " | ".join(str(S.get(c, "")) for c in cols) + " |")
    (OUT / "rerun_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with (OUT / "rerun_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for S in rows:
            w.writerow({c: S.get(c, "") for c in cols})
    print("\n".join(md))


if __name__ == "__main__":
    main()
