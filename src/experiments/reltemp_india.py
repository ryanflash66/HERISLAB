"""
Relative-temperature rules on panel pixels (no model). Plot 16 only.
Declared BEFORE running (Oct 25):

  temperatures  real degrees C from each R-JPG (hlab.rjpg), full 640x512
  panel mask    the DINOv2 panel/ground mask (hlab.panels.dino_panel_fit, the
                panel_dino_india.py fit with the same seed), saved to
                results/india/val_panel_mask.npy, scaled up to 640x512
  hot spot      3x3 average of temperature (smooths single noisy pixels)

  rel_local  (primary)   max over panel pixels of  hot spot - local reference,
                         local reference = mean panel temperature in the 31x31
                         window around the pixel (about one module), only where
                         >= 30% of that window is panel  (hlab.panels.local_excess)
  rel_frame  (secondary) max over panel pixels of hot spot - median panel temperature

Compared with (same frames, from val_panel_scores.npy):
  input_panel_peak3 (no model, 8-bit brightness)  0.642
  ae_panel_peak3    (autoencoder)                 0.637
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hlab.datasets import Dataset  # noqa: E402
from hlab.metrics import auroc, auroc_ci, paired_delta_ci  # noqa: E402
from hlab.panels import MIN_PANEL_SHARE, MODULE_WIN, dino_panel_fit, local_excess, resolve_device, smoothed  # noqa: E402
from hlab.paths import RESULTS, ROOT  # noqa: E402
from hlab.rjpg import raw_to_celsius, read_rjpg  # noqa: E402

RES = RESULTS / "india"
SEED, N_CLUSTER_FRAMES, N_BANK_FRAMES = 0, 300, 1500      # the panel_dino_india.py fit


def build_masks(ds, x, dev):
    xt = np.load(ds.array_path("train"), mmap_mode="r")
    pm = dino_panel_fit(xt, ds.stats, seed=SEED, n_bank_frames=N_BANK_FRAMES, n_cluster_frames=N_CLUSTER_FRAMES, device=dev)
    return pm.masks(x)


def main(device=None):
    dev = resolve_device(device)
    ds = Dataset("india")
    s_ = ds.split("val")
    x, meta, y = s_.x, s_.meta, s_.labels
    mpath = RES / "val_panel_mask.npy"
    masks = np.load(mpath) if mpath.exists() else build_masks(ds, x, dev)
    np.save(mpath, masks)
    print(f"panel mask: removes {100*(1-np.median(masks.mean((1, 2)))):.1f}% of a frame (median); "
          f"earlier run removed 29.3%")

    rl, rf, tmed = [], [], []
    for i, r in enumerate(meta):
        raw, calib = read_rjpg(ROOT / r["rel_path"])
        t = torch.from_numpy(raw_to_celsius(raw, calib)).to(dev)[None, None]
        m = Fn.interpolate(torch.from_numpy(masks[i]).float()[None, None].to(dev), size=t.shape[2:], mode="nearest")
        excess, cov = local_excess(t, m, MODULE_WIN)
        ok = (m > 0) & (cov >= MIN_PANEL_SHARE)
        if ok.sum() == 0: ok = m > 0
        if ok.sum() == 0: ok = torch.ones_like(m, dtype=torch.bool)
        rl.append(float(excess[ok].max()))
        t3 = smoothed(t, 3)
        pm = t[m > 0] if (m > 0).any() else t.flatten()
        rf.append(float(t3[m > 0].max() - pm.median()) if (m > 0).any() else float(t3.max() - pm.median()))
        tmed.append(float(pm.median()))
        if i % 400 == 0: print(f"  read {i}/{len(meta)} frames", flush=True)
    s = {"rel_local (primary)": np.array(rl), "rel_frame": np.array(rf)}
    old = np.load(RES / "val_panel_scores.npy", allow_pickle=True).item()
    s["input_panel_peak3 (8-bit, earlier)"] = old["input_panel_peak3 (no model)"]
    s["ae_panel_peak3 (earlier)"] = old["ae_panel_peak3"]

    print(f"RELATIVE TEMPERATURE on VAL (plot 16), panel pixels only: {int(y.sum())} faulty / {int((y == 0).sum())} healthy")
    for name, v in s.items():
        a = auroc(v, y); lo, hi = auroc_ci(v, y, n_boot=1000, seed=0)
        print(f"  {name:36s} AUROC {a:.3f}  (95% CI {lo:.3f} to {hi:.3f})")
    for name in ("rel_local (primary)", "rel_frame"):
        print(f"    {name}: median healthy {np.median(s[name][y == 0]):.1f} K, median faulty {np.median(s[name][y == 1]):.1f} K")
    p = s["rel_local (primary)"]
    for other in ("input_panel_peak3 (8-bit, earlier)", "ae_panel_peak3 (earlier)"):
        d, lo, hi = paired_delta_ci(s[other], p, y, n_boot=2000, seed=0)
        print(f"  rel_local minus {other}: {d:+.3f}  (95% CI {lo:+.3f} to {hi:+.3f})")
    np.save(RES / "val_reltemp_scores.npy", s, allow_pickle=True)
    with open(RES / "val_reltemp_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["rel_path", "label", "rel_local_K", "rel_frame_K", "panel_median_C"])
        for r, a_, b_, c_ in zip(meta, rl, rf, tmed): w.writerow([r["rel_path"], r["label"], f"{a_:.2f}", f"{b_:.2f}", f"{c_:.2f}"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default=None)
    main(ap.parse_args().device)
