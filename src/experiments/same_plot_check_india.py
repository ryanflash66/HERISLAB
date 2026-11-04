"""
Is the plot split to blame for the low scores? Check on plot 17 ITSELF.

The flights held out during training (hlab.datasets.flight_holdout, seed 42:
17.9, 17.14, 17.11) were never trained on (only used to decide when to stop).
Score their 966 healthy frames and their 139 faulty frames (which were set
aside, unused). Same plot, same days as training. If the split between plots
were the problem, scores here should be much higher than on plots 16 and 18.

Output: results/india/same_plot_check_log.txt (stdout)
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hlab.bundle import Bundle  # noqa: E402
from hlab.datasets import Dataset, flight_holdout  # noqa: E402
from hlab.metrics import auroc, auroc_ci  # noqa: E402
from hlab.panels import peak_scores, resolve_device  # noqa: E402
from hlab.paths import MANIFESTS, ROOT  # noqa: E402
from hlab.rjpg import read_rjpg  # noqa: E402
from hlab.transform import normalize, to_model_input  # noqa: E402


def main(device=None):
    bundle = Bundle.load("india", resolve_device(device))
    ds = Dataset("india")
    xt = np.load(ds.array_path("train"), mmap_mode="r")
    _tr, held, hold = flight_holdout(ds.meta("train"))          # the training run's early-stop flights
    xh = np.stack([xt[i] for i in held])
    rows = [r for r in csv.DictReader(open(MANIFESTS / "india_plot_split.csv", encoding="utf-8"))
            if r["split"] == "train_faulty_unused" and r["flight"] in set(hold)]
    xf = np.stack([normalize(to_model_input(read_rjpg(ROOT / r["rel_path"])[0], "minmax"), ds.stats) for r in rows]).astype(np.float32)
    x = np.concatenate([xh, xf]); y = np.r_[np.zeros(len(xh), int), np.ones(len(xf), int)]
    e = bundle.score_normalized(x, batch_size=64); p = peak_scores(bundle.model, x, bundle.device)
    print(f"SAME-PLOT CHECK, plot 17 held-out flights {sorted(hold)}: {len(xf)} faulty / {len(xh)} healthy")
    for name, v, v16, v18 in [("ae_mean (original)", e, 0.376, 0.373),
                              ("ae_peak3", p["peak3"], 0.640, 0.597),
                              ("input_peak3 (no model)", p["input_peak3 (no model)"], 0.600, 0.585)]:
        a = auroc(v, y); lo, hi_ = auroc_ci(v, y, n_boot=1000, seed=0)
        print(f"  {name:24s} AUROC {a:.3f} (95% CI {lo:.3f} to {hi_:.3f})   [plot 16: {v16:.3f}, plot 18: {v18:.3f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default=None)
    main(ap.parse_args().device)
