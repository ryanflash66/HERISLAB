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

Default runs on VAL (plot 16) only. Test (plot 18) needs --test and should be
run once, after the score is frozen.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_india import WEIGHTS, auroc, boot_ci, load  # noqa: E402
from train_autoencoder import ThermalAutoencoder  # noqa: E402


def scores(model, x, dev):
    out = {"peak3": [], "peak5": [], "peakmax": [], "input_peak3 (no model)": []}
    with torch.no_grad():
        for i in range(0, len(x), 64):
            b = torch.from_numpy(x[i:i + 64]).unsqueeze(1).to(dev)
            e = (model(b) - b) ** 2
            flat = e.flatten(1)
            med = flat.median(1).values.clamp_min(1e-12)
            out["peak3"].append((Fn.avg_pool2d(e, 3, 1, 1).flatten(1).max(1).values / med).cpu().numpy())
            out["peak5"].append((Fn.avg_pool2d(e, 5, 1, 2).flatten(1).max(1).values / med).cpu().numpy())
            out["peakmax"].append((flat.max(1).values / flat.mean(1)).cpu().numpy())
            xf = b.flatten(1)
            ip = Fn.avg_pool2d(b, 3, 1, 1).flatten(1).max(1).values
            out["input_peak3 (no model)"].append(((ip - xf.median(1).values) / xf.std(1)).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true")
    split = "test" if ap.parse_args().test else "val"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThermalAutoencoder().to(dev)
    model.load_state_dict(torch.load(WEIGHTS, map_location=dev)); model.eval()
    x, meta, y = load(split)
    s = scores(model, x, dev)
    print(f"PEAK SCORES on {split.upper()} ({'plot 16' if split == 'val' else 'plot 18'}): "
          f"{int(y.sum())} faulty / {int((y == 0).sum())} healthy   (0.5 = coin flip; old mean score: val 0.376)")
    for k, v in s.items():
        a = auroc(v, y); lo, hi = boot_ci(v, y)
        tag = "  <- primary" if k == "peak3" else ""
        print(f"  {k:24s} AUROC {a:.3f}  (95% CI {lo:.3f} to {hi:.3f}){tag}")
    np.save(Path(__file__).resolve().parent.parent / "results" / "india" / f"{split}_peak3.npy", s["peak3"])


if __name__ == "__main__":
    main()
