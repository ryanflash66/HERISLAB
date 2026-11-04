"""
HER-114: rerun the evaluation-validity audit with the THM "healthy" frames
removed or screened.

Nothing is retrained and the split is untouched: the PV specialist checkpoint,
the seed-42 split (src/preprocess_pv.py) and the seed-43 training run are the
same ones behind results/pv_*_errors.npy. Each scenario is a keep-mask over the
held-out healthy frames, run through audit.run_scenario:

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
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import PVInputs, print_scenario, run_scenario, write_summary  # noqa: E402  (puts src/ on sys.path)
from hlab.bundle import Bundle  # noqa: E402
from hlab.datasets import Dataset  # noqa: E402
from hlab.paths import MANIFESTS, RESULTS  # noqa: E402

OUT = RESULTS / "her-114"
SCREEN = MANIFESTS / "thm_screened_frames.csv"
COLS = ["scenario", "n_healthy", "n_thm", "auroc_files", "auroc_files_ci", "auroc_unique", "auroc_unique_ci",
        "fp_files", "fp_thm", "fp_share_thm", "f1_saved_files", "f1_trivial_files", "f1_trivial_unique",
        "auroc_source", "auroc_source_def", "p1_baseline", "best_single", "logreg16", "leak_adjacent"]


def scenarios(D):
    """(name, keep-mask over the held-out healthy frames) for the five HER-114 scenarios."""
    ho_names = [Path(p).name for p in D.ho]
    is_thm = D.seq == "thm"
    screen = {r["frame"]: r for r in csv.DictReader(SCREEN.open(encoding="utf-8"))}
    flag = np.array([screen[f]["flag"] if f in screen else "n/a" for f in ho_names])
    hot = np.array([(screen[f]["hotspot"] == "yes") if f in screen else False for f in ho_names])
    return [
        ("a_as_is", np.ones(len(ho_names), bool)),
        ("b_thm_removed", ~is_thm),
        ("c_flagged_thm_removed", ~is_thm | (flag != "drop")),
        ("c2_hotspot_thm_removed", ~is_thm | ~hot),
        ("c3_strict_keep_only", ~is_thm | (flag == "keep")),
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pv = Bundle.load("pv")
    D = PVInputs(Dataset("pv"), pv)
    print("Computing global image statistics (no model)...")
    D.features()
    rows = []
    for name, keep in scenarios(D):
        buf = io.StringIO()
        with redirect_stdout(buf):
            print("=" * 70, f"\nHER-114 rerun {name}\n" + "=" * 70)
            S = run_scenario(name, keep, D, pv.require_threshold())
            print_scenario(S)
        (OUT / f"rerun_{name}.txt").write_text(buf.getvalue(), encoding="utf-8")
        print(buf.getvalue())
        rows.append(S)
    print(write_summary(rows, COLS, OUT / "rerun_summary.md", OUT / "rerun_summary.csv"))


if __name__ == "__main__":
    main()
