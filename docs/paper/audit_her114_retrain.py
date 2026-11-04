"""
HER-114 retrain: validity controls for the O&M-only PV specialist.

Runs the same controls as audit_her114_reruns.py (metrics from hlab.metrics,
features and baselines from audit_eval_validity.py) on the errors of the model
trained WITHOUT the THM aerial-survey frames:

  split      data/CA_Preprocessed/pv_om/split_manifest.txt   (seed 42, O&M only)
  checkpoint models/autoencoder_pv_om_best.pth               (seed 43)
  errors     results/pv_om_normal_errors.npy, results/pv_om_fault_errors.npy
  threshold  re-tuned best-F1 threshold from results/pv_om_eval_metrics.npy

Reported: AUROC with bootstrap 95% CI (1,000 files and 140 unique faults),
confusion at the re-tuned threshold, flag-everything F1, single-statistic
baselines incl. p1, 16-statistic logistic regression, adjacent-frame leakage,
source AUROC (O&M sequence dr vs sr), THM flag rate of the new model. The
split and arrays come from hlab.datasets ("pv_om"), the threshold from the
"pv_om" bundle.

The THM+O&M rows (scenarios a and b) are read from results/her-114/rerun_summary.csv
so the three can be shown side by side.

Usage:
    venv/Scripts/python.exe docs/paper/audit_her114_retrain.py
Writes results/her-114_retrain/audit_retrain_om_only.txt, summary.md, summary.csv
"""
import csv
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_eval_validity as A  # noqa: E402
import audit_her114_reruns as R  # noqa: E402
from hlab.bundle import Bundle  # noqa: E402
from hlab.datasets import Dataset  # noqa: E402
from hlab.metrics import confusion, from_groups  # noqa: E402

ROOT = A.ROOT
OUT = A.RES / "her-114_retrain"
PREV = A.RES / "her-114" / "rerun_summary.csv"


def load_inputs():
    ds = Dataset("pv_om")
    assert not any(src == "thm" for _f, _split, src in ds.manifest()), "THM frame found in the O&M-only split"
    D = R.load_inputs(ds, errors_prefix="pv_om")
    print(f"O&M-only split: {D['n_train']} train / {len(D['ho'])} holdout; "
          f"faults {len(D['pvf'])} files / {D['n_unique']} unique")
    return D


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    om = Bundle.load("pv_om")
    thr = om.require_threshold()
    M = np.load(om.metrics_path, allow_pickle=True).item()   # THM flag rates recorded by the evaluator
    buf = io.StringIO()
    with redirect_stdout(buf):
        print("=" * 70, "\nHER-114 retrain: O&M-only PV specialist\n" + "=" * 70)
        print(f"threshold used below ('saved threshold' in the printout) = re-tuned best-F1 "
              f"threshold of the O&M-only model: {thr:.6f}")
        D = load_inputs()
        S = R.run_scenario("retrain_om_only", np.ones(len(D["ho"]), bool), D, thr)
        c = A.rounded(confusion(*from_groups(D["pvn"], D["pvf"]), thr))
        print(f"\nre-tuned threshold {thr:.6f}: {c}")
        print(f"THM flag rate at that threshold: all 1,075 frames {M['thm_flag_rate_all']:.1%}; "
              f"the 84 frames held out by the original split {M['thm_flag_rate_holdout84']:.1%}")
    txt = buf.getvalue()
    (OUT / "audit_retrain_om_only.txt").write_text(txt, encoding="utf-8")
    print(txt)

    S["threshold"] = round(thr, 6)
    S["thm_flag_rate_holdout84"] = f"{M['thm_flag_rate_holdout84']:.1%}"
    S["thm_flag_rate_all"] = f"{M['thm_flag_rate_all']:.1%}"
    S["model"] = "O&M-only retrain (seed 43), 7,052 O&M train frames"

    rows = []
    with PREV.open(encoding="utf-8") as fh:
        prev = {r["scenario"]: r for r in csv.DictReader(fh)}
    for key, model in [("a_as_is", "THM+O&M model, holdout as is"),
                       ("b_thm_removed", "THM+O&M model, THM dropped at evaluation")]:
        r = dict(prev[key]); r["model"] = model; r["threshold"] = f"{A.PV_THR:.6f}"
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
