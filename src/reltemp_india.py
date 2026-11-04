"""
Relative-temperature rules on panel pixels (no model). Plot 16 only.
Declared BEFORE running (Oct 25):

  temperatures  real degrees C from each R-JPG (src/flir_rjpg.py), full 640x512
  panel mask    the DINOv2 panel/ground mask from panel_dino_india.py, rebuilt
                with the same seed and saved to results/india/val_panel_mask.npy,
                scaled up to 640x512
  hot spot      3x3 average of temperature (smooths single noisy pixels)

  rel_local  (primary)   max over panel pixels of  hot spot - local reference,
                         local reference = mean panel temperature in the 31x31
                         window around the pixel (about one module), only where
                         >= 30% of that window is panel
  rel_frame  (secondary) max over panel pixels of hot spot - median panel temperature

Compared with (same frames, from val_panel_scores.npy):
  input_panel_peak3 (no model, 8-bit brightness)  0.642
  ae_panel_peak3    (autoencoder)                 0.637
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_india import ROOT, auroc, boot_ci, load  # noqa: E402
from flir_rjpg import raw_to_celsius, read_rjpg  # noqa: E402
from panel_dino_india import (DATA, N_BANK_FRAMES, N_CLUSTER_FRAMES, RES, SEED,  # noqa: E402
                              feats, kmeans2, panel_of)

WIN, MIN_PANEL = 31, 0.30


def build_masks(x, dev):
    rng = np.random.default_rng(SEED)
    ns = np.load(DATA / "norm_stats.npy", allow_pickle=True).item(); mean, std = float(ns["mean"]), float(ns["std"])
    net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(dev).eval()
    xt = np.load(DATA / "train.npy", mmap_mode="r")
    pick = np.sort(rng.choice(len(xt), N_BANK_FRAMES, replace=False))       # same draws as panel_dino_india.py
    cl = rng.choice(N_BANK_FRAMES, N_CLUSTER_FRAMES, replace=False)
    f, s = feats(net, xt[pick[cl]], mean, std, dev)
    z = f.reshape(-1, 384).float(); sp = s.reshape(-1)
    cent = kmeans2(z, seed=SEED); a = torch.cdist(z, cent).argmin(1)
    panel_k = int(np.argmin([float(sp[a == k].mean()) for k in range(2)]))
    m = []
    for i in range(0, len(x), 128):
        fv, _ = feats(net, x[i:i + 128], mean, std, dev); m.append(panel_of(fv, cent, panel_k).cpu())
    return torch.cat(m).numpy()


def main():
    dev = torch.device("cuda")
    x, meta, y = load("val")
    mpath = RES / "val_panel_mask.npy"
    masks = np.load(mpath) if mpath.exists() else build_masks(x, dev)
    np.save(mpath, masks)
    print(f"panel mask: removes {100*(1-np.median(masks.mean((1, 2)))):.1f}% of a frame (median); "
          f"earlier run removed 29.3%")

    rl, rf, tmed = [], [], []
    k = torch.ones(1, 1, WIN, WIN, device=dev) / WIN ** 2
    for i, r in enumerate(meta):
        raw, calib = read_rjpg(ROOT / r["rel_path"])
        t = torch.from_numpy(raw_to_celsius(raw, calib)).to(dev)[None, None]
        m = Fn.interpolate(torch.from_numpy(masks[i]).float()[None, None].to(dev), size=t.shape[2:], mode="nearest")
        t3 = Fn.avg_pool2d(Fn.pad(t, (1, 1, 1, 1), mode="replicate"), 3, 1)
        p = WIN // 2
        num = Fn.conv2d(Fn.pad(t * m, (p, p, p, p), mode="replicate"), k)
        den = Fn.conv2d(Fn.pad(m, (p, p, p, p), mode="replicate"), k)
        ok = (m > 0) & (den >= MIN_PANEL)
        if ok.sum() == 0: ok = m > 0
        if ok.sum() == 0: ok = torch.ones_like(m, dtype=torch.bool)
        ref = num / den.clamp_min(1e-6)
        rl.append(float((t3 - ref)[ok].max()))
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
        a = auroc(v, y); lo, hi = boot_ci(v, y)
        print(f"  {name:36s} AUROC {a:.3f}  (95% CI {lo:.3f} to {hi:.3f})")
    for name in ("rel_local (primary)", "rel_frame"):
        print(f"    {name}: median healthy {np.median(s[name][y == 0]):.1f} K, median faulty {np.median(s[name][y == 1]):.1f} K")
    rng = np.random.default_rng(0); i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    p = s["rel_local (primary)"]
    for other in ("input_panel_peak3 (8-bit, earlier)", "ae_panel_peak3 (earlier)"):
        q = s[other]; d = []
        for _ in range(2000):
            idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
            d.append(auroc(p[idx], y[idx]) - auroc(q[idx], y[idx]))
        print(f"  rel_local minus {other}: {auroc(p, y) - auroc(q, y):+.3f}  (95% CI {np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f})")
    np.save(RES / "val_reltemp_scores.npy", s, allow_pickle=True)
    with open(RES / "val_reltemp_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["rel_path", "label", "rel_local_K", "rel_frame_K", "panel_median_C"])
        for r, a_, b_, c_ in zip(meta, rl, rf, tmed): w.writerow([r["rel_path"], r["label"], f"{a_:.2f}", f"{b_:.2f}", f"{c_:.2f}"])


if __name__ == "__main__":
    main()
