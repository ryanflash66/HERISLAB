"""
Option A from the box check: score each frame by how much its sharpest small
error peak stands out from that frame's own typical error.

Declared BEFORE running (Oct 25), so plot 16 is not used to shop for a score:
  primary    peak3   = max of 3x3-averaged error map / median pixel error
  secondary  peak5   = same with a 5x5 window
  secondary  peakmax = max pixel error / mean pixel error
No-model baseline (HER-98): the same 3x3 peak computed on the INPUT image,
  (max of 3x3-averaged input - median input) / input std
so we can tell whether the model adds anything over "find the brightest dot".
The scores themselves are hlab.panels.peak_scores.

Default runs on VAL (plot 16) only. Test (plot 18) needs --test and should be
run once, after the score is frozen.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hlab.bundle import Bundle  # noqa: E402
from hlab.datasets import Dataset  # noqa: E402
from hlab.metrics import auroc, auroc_ci  # noqa: E402
from hlab.panels import peak_scores, resolve_device  # noqa: E402
from hlab.paths import RESULTS  # noqa: E402


def main(split="val", device=None):
    dev = resolve_device(device)
    model = Bundle.load("india", dev).model
    s_ = Dataset("india").split(split)
    x, y = s_.x, s_.labels
    s = peak_scores(model, x, dev)
    print(f"PEAK SCORES on {split.upper()} ({'plot 16' if split == 'val' else 'plot 18'}): "
          f"{int(y.sum())} faulty / {int((y == 0).sum())} healthy   (0.5 = coin flip; old mean score: val 0.376)")
    for k, v in s.items():
        a = auroc(v, y); lo, hi = auroc_ci(v, y, n_boot=1000, seed=0)
        tag = "  <- primary" if k == "peak3" else ""
        print(f"  {k:24s} AUROC {a:.3f}  (95% CI {lo:.3f} to {hi:.3f}){tag}")
    np.save(RESULTS / "india" / f"{split}_peak3.npy", s["peak3"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true"); ap.add_argument("--device", default=None)
    a = ap.parse_args()
    main("test" if a.test else "val", a.device)
