"""
HER-107: rerun the no-model baselines with the healthy and fault PV images
transformed the same way.

The evaluation-validity audit found that global image statistics separate the
held-out CA-PV healthy frames from the PVMD faults about as well as the
autoencoder does (1st percentile AUROC 0.972, 16-statistic logistic regression
0.999). This script asks how much of that comes from the transform. For each
transform variant it reloads the same frames from the raw files (892 held-out
healthy frames of the seed-42 pv/ split, the 140 pixel-unique PVMD faults),
computes audit.global_features on the 0-255 model input, and runs
audit.check_trivial_baselines:

  as_is            scale_by_max + stretch   what pv/ holds (should reproduce 0.972)
  crop             scale_by_max + crop      aspect-ratio fix only
  minmax_crop      minmax + crop            one stretch rule for both classes
  equalize         equalize + stretch       intensity fix only
  matched          equalize + crop          what pv_matched/ holds (preprocess_pv_matched.py)

No model is trained or scored: the autoencoder's own numbers on pv_matched/
need a retrain on those arrays (train.py).

Usage:
    venv\\Scripts\\python.exe docs\\paper\\audit_her107_baselines.py
Writes results/her-107/baselines.md and results/her-107/baselines.csv
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import FEAT, SEED, check_trivial_baselines, global_features, unique_first, write_summary  # noqa: E402

from hlab.datasets import Dataset, pvmd_unique_groups  # noqa: E402
from hlab.paths import RESULTS, ROOT  # noqa: E402
from hlab.transform import load_gray, to_model_input  # noqa: E402

OUT = RESULTS / "her-107"
VARIANTS = [
    ("as_is", "scale_by_max", "stretch"),
    ("crop", "scale_by_max", "crop"),
    ("minmax_crop", "minmax", "crop"),
    ("equalize", "equalize", "stretch"),
    ("matched", "equalize", "crop"),
]
IDENTITY = {"mean": 0.0, "std": 1.0}      # global_features denormalises; these arrays are already 0-255
COLS = ["variant", "representation", "fit", "p1_median_healthy", "p1_median_fault", "auroc_p1",
        "top3", "logreg"]


def frame_paths():
    """(held-out healthy paths, unique-fault paths) in the pv/ split's order."""
    ds = Dataset("pv")
    rows = ds.manifest()
    healthy = [ROOT / f for f, split, _src in rows if split == "holdout"]
    faults = [ROOT / f for f, split, _src in rows if split == "fault"]
    gid = pvmd_unique_groups()[1]
    if len(gid) != len(faults):
        raise RuntimeError(f"{len(faults)} fault files in the manifest, {len(gid)} in the PVMD folder")
    return healthy, [faults[i] for i in unique_first(gid)]


def features(paths, representation, fit):
    return global_features([to_model_input(load_gray(p), representation, fit) for p in paths], IDENTITY)


def run_variant(name, representation, fit, healthy, faults):
    negF, posF = features(healthy, representation, fit), features(faults, representation, fit)
    b = check_trivial_baselines(negF, posF, seed=SEED)
    j = FEAT.index("p1")
    return dict(variant=name, representation=representation, fit=fit,
                p1_median_healthy=round(float(np.median(negF[:, j])), 1),
                p1_median_fault=round(float(np.median(posF[:, j])), 1),
                auroc_p1=round(b["single"]["p1"], 3),
                top3=", ".join(f"{k} {v:.3f}" for k, v in b["top"]),
                logreg=round(b["logreg"], 3), _single=b["single"])


def stored_check(name):
    """The same baselines on a stored dataset's arrays (pv/ or pv_matched/), if present."""
    ds = Dataset(name)
    if not (ds.has_split("holdout") and ds.has_split("fault")):
        return None
    gid = pvmd_unique_groups()[1]
    negF = global_features(ds.split("holdout").x, ds.stats)
    posF = global_features(ds.split("fault").x, ds.stats)[unique_first(gid)]
    b = check_trivial_baselines(negF, posF, seed=SEED)
    return b["single"]["p1"], b["logreg"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    healthy, faults = frame_paths()
    print(f"{len(healthy)} held-out healthy frames, {len(faults)} unique PVMD faults")
    rows = []
    for name, rep, fit in VARIANTS:
        print(f"  {name}: {rep} + {fit} ...", flush=True)
        r = run_variant(name, rep, fit, healthy, faults)
        print(f"    p1 median {r['p1_median_healthy']} healthy vs {r['p1_median_fault']} fault, "
              f"AUROC p1 {r['auroc_p1']}, top3 {r['top3']}, logreg {r['logreg']}")
        rows.append(r)
    print()
    print(write_summary(rows, COLS, OUT / "baselines.md", OUT / "baselines.csv"))

    single = ["| statistic | " + " | ".join(r["variant"] for r in rows) + " |", "|---|" + "---|" * len(rows)]
    for k in FEAT:
        single.append(f"| {k} | " + " | ".join(f"{r['_single'][k]:.3f}" for r in rows) + " |")
    (OUT / "baselines_single.md").write_text("\n".join(single) + "\n", encoding="utf-8")
    print("\n" + "\n".join(single))

    for name in ("pv", "pv_matched"):
        s = stored_check(name)
        if s:
            print(f"\nstored {name}/ arrays: AUROC p1 {s[0]:.3f}, logreg {s[1]:.3f}")


if __name__ == "__main__":
    main()
