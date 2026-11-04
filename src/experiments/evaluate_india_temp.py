"""
Evaluate the temperature-scale autoencoder (HER-82 retrain, Oct 26).

Default: VAL (plot 16) only. --test runs TEST (plot 18) and CROSS_SITE once.
Everything declared before running:
  model scores   ae_peak3 (primary: max 3x3-averaged error / median error)
                 ae_mean  (the original average-error score, for the record)
  no-model       input_peak3 = brightest 3x3 spot in degrees above the frame's
                 median, in units of the frame's spread (same rule as before, now on C)
  thresholds     best F1 on VAL, frozen, applied to TEST (HER-97)
  comparison     the old model's numbers (per-frame stretched input) printed beside
Outputs: results/india_temp/{val|test}_eval_log.txt (stdout), *_scores.npy
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hlab.bundle import Bundle  # noqa: E402
from hlab.datasets import Dataset  # noqa: E402
from hlab.metrics import auroc, auroc_ci, best_f1_threshold, confusion, paired_delta_ci  # noqa: E402
from hlab.panels import peak_ratio, peak_standout, resolve_device  # noqa: E402
from hlab.paths import RESULTS  # noqa: E402

RES = RESULTS / "india_temp"
OLD = {"val": {"ae_peak3": 0.640, "ae_mean": 0.376, "input_peak3": 0.600},
       "test": {"ae_peak3": 0.597, "ae_mean": 0.373, "input_peak3": 0.585},
       "cross_site": {"ae_peak3": 0.346, "ae_mean": 0.332, "input_peak3": 0.610}}


def load(ds, split):
    """(degrees C, z-scored, meta, labels) of one india_temp split."""
    s = ds.split(split)
    return np.load(ds.array_path(split)), s.x, s.meta, s.labels


@torch.no_grad()
def scores(model, xc, xz, dev):
    out = {"ae_peak3": [], "ae_mean": [], "input_peak3": []}
    for i in range(0, len(xz), 64):
        b = torch.from_numpy(xz[i:i + 64]).unsqueeze(1).to(dev)
        e = (model(b) - b) ** 2
        out["ae_mean"].append(e.flatten(1).mean(1).cpu().numpy())
        out["ae_peak3"].append(peak_ratio(e).cpu().numpy())
        t = torch.from_numpy(xc[i:i + 64]).unsqueeze(1).to(dev)
        out["input_peak3"].append(peak_standout(t).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def report(split, s, y, thr=None):
    print(f"\n======== {split.upper()} ({int(y.sum())} faulty / {int((y == 0).sum())} healthy)   [old model, stretched input, in brackets]")
    if thr is not None:
        fl = confusion(np.ones(len(y)), y, 0); print(f"  flag everything: F1 {fl['f1']:.3f}")
    for k, v in s.items():
        a = auroc(v, y); lo, hi = auroc_ci(v, y, n_boot=1000, seed=0)
        line = f"  {k:12s} AUROC {a:.3f} (95% CI {lo:.3f} to {hi:.3f})  [old {OLD[split][k]:.3f}]"
        if thr is not None:
            r = confusion(v, y, thr[k]); line += f" | precision {r['precision']:.3f} recall {r['recall']:.3f} F1 {r['f1']:.3f}"
        print(line + ("  <- primary" if k == "ae_peak3" else ""))
    d, lo, hi = paired_delta_ci(s["input_peak3"], s["ae_peak3"], y, n_boot=2000, seed=0)
    print(f"  model minus no-model rule: {d:+.3f} (95% CI {lo:+.3f} to {hi:+.3f})")


def main(test=False, device=None):
    dev = resolve_device(device)
    model = Bundle.load("india_temp", dev).model
    ds = Dataset("india_temp")
    xc, xz, meta, y = load(ds, "val"); sv = scores(model, xc, xz, dev)
    report("val", sv, y); np.save(RES / "val_scores.npy", sv, allow_pickle=True)
    if not test:
        return
    thr = {k: best_f1_threshold(v, y) for k, v in sv.items()}
    for split in ("test", "cross_site"):
        xc, xz, meta, y = load(ds, split); s = scores(model, xc, xz, dev)
        report(split, s, y, thr); np.save(RES / f"{split}_scores.npy", s, allow_pickle=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true"); ap.add_argument("--device", default=None)
    a = ap.parse_args()
    main(a.test, a.device)
