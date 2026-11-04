"""
Step 1 after the box check: same peak scores as score_peak_india.py, but only
over panel pixels (the no-model roughness mask, hlab.panels.panel_mask).
Declared before running (Oct 25):
  primary   peak3_masked = max of 3x3-averaged error inside the mask / median error inside the mask
  baseline  input_peak3_masked (no model) = (max of 3x3-averaged input inside mask - median) / std, inside mask
Also reports how much of the annotators' fault boxes survive the mask, so we
know whether the mask is cutting faults out.
VAL (plot 16) only.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hlab.bundle import Bundle  # noqa: E402
from hlab.datasets import Dataset  # noqa: E402
from hlab.metrics import auroc, auroc_ci, paired_delta_ci  # noqa: E402
from hlab.panels import MASK_VERSION, box_mask, panel_mask, peak_ratio, peak_standout, resolve_device  # noqa: E402
from hlab.paths import RESULTS, ROOT  # noqa: E402


def main(device=None):
    dev = resolve_device(device)
    model = Bundle.load("india", dev).model
    s_ = Dataset("india").split("val")
    x, meta, y = s_.x, s_.meta, s_.labels
    # pass bar first, no model and no scores: the mask must keep >= 90% of fault-box pixels
    fi = np.where(y == 1)[0]
    m = panel_mask(torch.from_numpy(x[fi]).unsqueeze(1).to(dev))[:, 0]
    cov = []
    for j, k in enumerate(fi):
        bm = torch.from_numpy(box_mask((ROOT / meta[k]["rel_path"]).with_suffix(".json"), 0)).to(dev)
        cov.append(float(m[j][bm].float().mean()))
    print(f"MASK v{MASK_VERSION} pass bar: fault-box pixels kept {100*np.mean(cov):.1f}% (need >= 90%); "
          f"frames losing every box pixel: {sum(c == 0 for c in cov)}/{len(cov)}")
    if np.mean(cov) < 0.90:
        print("FAILED the pass bar, scores not computed (as agreed: drop masking, go to step 2)"); return
    out = {"peak3_masked": [], "input_peak3_masked (no model)": [], "peak3 (unmasked, earlier)": []}
    kept, box = [], []
    with torch.no_grad():
        for i in range(0, len(x), 64):
            b = torch.from_numpy(x[i:i + 64]).unsqueeze(1).to(dev)
            m = panel_mask(b)
            e = (model(b) - b) ** 2
            out["peak3_masked"].append(peak_ratio(e, m).cpu().numpy())
            out["peak3 (unmasked, earlier)"].append(peak_ratio(e).cpu().numpy())
            out["input_peak3_masked (no model)"].append(peak_standout(b, m).cpu().numpy())
            kept.append(m[:, 0].float().mean((1, 2)).cpu().numpy())
            for j in range(len(b)):
                k = i + j
                if y[k] == 1:
                    bm = torch.from_numpy(box_mask((ROOT / meta[k]["rel_path"]).with_suffix(".json"), 0)).to(dev)
                    mk = m[j, 0]
                    box.append((float(mk[bm].float().mean()), bool(mk[bm].any())))
    s = {k: np.concatenate(v) for k, v in out.items()}
    kept = np.concatenate(kept)
    print(f"MASKED PEAK SCORES on VAL (plot 16): {int(y.sum())} faulty / {int((y == 0).sum())} healthy")
    print(f"  mask keeps {100*np.median(kept):.0f}% of a frame (median)")
    print(f"  fault box pixels kept by the mask: {100*np.mean([b[0] for b in box]):.0f}% on average; "
          f"frames with at least part of a box kept: {sum(b[1] for b in box)}/{len(box)}")
    for k, v in s.items():
        a = auroc(v, y); lo, hi = auroc_ci(v, y, n_boot=1000, seed=0)
        print(f"  {k:32s} AUROC {a:.3f}  (95% CI {lo:.3f} to {hi:.3f})")
    d, lo, hi = paired_delta_ci(s["input_peak3_masked (no model)"], s["peak3_masked"], y, n_boot=2000, seed=0)
    print(f"  model minus no-model (masked): {d:+.3f}  (95% CI {lo:+.3f} to {hi:+.3f})")
    np.save(RESULTS / "india" / f"val_peak3_masked_v{MASK_VERSION}.npy", s["peak3_masked"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default=None)
    main(ap.parse_args().device)
