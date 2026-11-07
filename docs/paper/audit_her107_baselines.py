"""
HER-107: rerun the no-model baselines with the healthy and fault PV images
transformed the same way.

The evaluation-validity audit found that global image statistics separate the
held-out CA-PV healthy frames from the PVMD faults about as well as the
autoencoder does (1st percentile AUROC 0.972, 16-statistic logistic regression
0.999). This script asks how much of that comes from the transform. For each
variant in audit.TRANSFORM_VARIANTS, audit.check_transform_baselines reloads the
same frames from the raw files (892 held-out healthy frames of the seed-42 pv/
split, the 140 pixel-unique PVMD faults) and runs the trivial baselines on the
0-255 model input:

  as_is            scale_by_max + stretch   what pv/ holds (reproduces 0.972)
  crop             scale_by_max + crop      aspect-ratio fix only
  minmax_crop      minmax + crop            one stretch rule for both classes
  equalize         equalize + stretch       intensity fix only
  matched          equalize + crop          what pv_matched/ holds (preprocess_pv_matched.py)

audit.check_stored_baselines repeats the check on the stored pv/ and
pv_matched/ arrays when they are built. No model is trained or scored.

Usage:
    venv\\Scripts\\python.exe docs\\paper\\audit_her107_baselines.py
Writes results/her-107/baselines.md, baselines.csv and baselines_single.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit  # noqa: E402

from hlab.datasets import Dataset, pvmd_unique_groups  # noqa: E402
from hlab.paths import RESULTS  # noqa: E402

OUT = RESULTS / "her-107"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gid = pvmd_unique_groups()[1]
    healthy, faults = audit.pv_frame_paths(Dataset("pv"), gid)
    print(f"{len(healthy)} held-out healthy frames, {len(faults)} unique PVMD faults")
    rows = audit.check_transform_baselines(healthy, faults)
    for r in rows:
        print(f"  {r['variant']}: {r['representation']} + {r['fit']} ...\n"
              f"    p1 median {r['p1_median_healthy']} healthy vs {r['p1_median_fault']} fault, "
              f"AUROC p1 {r['auroc_p1']}, top3 {r['top3']}, logreg {r['logreg']}")
    print()
    print(audit.write_summary(rows, audit.TRANSFORM_COLS, OUT / "baselines.md", OUT / "baselines.csv"))

    single = audit.single_statistic_table(rows)
    (OUT / "baselines_single.md").write_text(single + "\n", encoding="utf-8")
    print("\n" + single)

    for name in ("pv", "pv_matched"):
        b = audit.check_stored_baselines(Dataset(name), gid)
        if b:
            print(f"\nstored {name}/ arrays: AUROC p1 {b['single']['p1']:.3f}, logreg {b['logreg']:.3f}")


if __name__ == "__main__":
    main()
