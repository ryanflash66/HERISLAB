"""
Cheap panel mask for the India frames (box-check follow-up, step 1).

Idea: panels are smooth, gravel and plants are rough. No model, no labels.
  1. local roughness = std of the input in a 7x7 window
  2. per frame, split rough vs smooth with Otsu's threshold on log(roughness)
     (adapts to each frame's contrast, nothing tuned on fault labels)
  3. fill small holes (cell grid lines, hot spots) with a 11x11 closing
  4. shave 3 px off the panel edges (panel/ground borders are high-error)

Version 2 (Oct 25, the one allowed retry): v1 cut out half the fault-box pixels,
because a hot spot is a sharp bump and looks "rough". v2, declared before running:
  - 5x5 median filter first, which erases bumps a few pixels wide (hot spots)
    but leaves gravel texture rougher than panels
  - roughness window 9, closing 21, no edge shave
Pass bar before any score is looked at: keep >= 90% of fault-box pixels on plot 16.

Run directly to draw the mask on a few frames for a visual check:
  python src/panel_mask.py      -> results/india/panel_mask_check.png
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

SETTINGS = {1: dict(med=0, win=7, close=11, shave=3), 2: dict(med=5, win=9, close=21, shave=0)}
VERSION = 2


def _otsu(v, bins=128):
    h, e = np.histogram(v, bins=bins); c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = ((h * c).sum() - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def _median(b, k):
    p = k // 2; out = []
    for i in range(0, len(b), 8):
        u = Fn.unfold(Fn.pad(b[i:i + 8], (p, p, p, p), mode="reflect"), k)
        out.append(u.median(1).values.view(-1, 1, *b.shape[2:]))
    return torch.cat(out)


def panel_mask(b, version=None):
    """b: (N,1,H,W) float tensor (model input). Returns bool tensor (N,1,H,W), True = panel."""
    cfg = SETTINGS[version or VERSION]
    WIN, CLOSE, SHAVE = cfg["win"], cfg["close"], cfg["shave"]
    src = _median(b, cfg["med"]) if cfg["med"] else b
    p = WIN // 2
    xp = Fn.pad(src, (p, p, p, p), mode="reflect")
    mu = Fn.avg_pool2d(xp, WIN, 1); var = Fn.avg_pool2d(xp ** 2, WIN, 1) - mu ** 2
    lr = torch.log(var.clamp_min(1e-8).sqrt())
    thr = torch.tensor([_otsu(f.flatten().cpu().numpy()) for f in lr], device=b.device).view(-1, 1, 1, 1)
    m = (lr < thr).float()
    c = CLOSE // 2
    m = Fn.max_pool2d(m, CLOSE, 1, c)                     # dilate
    m = -Fn.max_pool2d(-m, CLOSE, 1, c)                   # erode  (= closing)
    if SHAVE:
        m = -Fn.max_pool2d(-m, 2 * SHAVE + 1, 1, SHAVE)   # shave edges
    return m > 0.5


if __name__ == "__main__":
    import csv, json, sys
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from box_check_india import box_mask
    ROOT = Path(__file__).resolve().parent.parent
    D = ROOT / "data" / "CA_Preprocessed" / "india"
    x = np.load(D / "val.npy", mmap_mode="r")
    meta = list(csv.DictReader(open(D / "val_meta.csv", encoding="utf-8")))
    fa = [i for i, m in enumerate(meta) if m["label"] == "faulty"]
    he = [i for i, m in enumerate(meta) if m["label"] == "healthy"]
    rng = np.random.default_rng(1)
    pick = list(rng.choice(he, 4, replace=False)) + list(rng.choice(fa, 4, replace=False))
    b = torch.from_numpy(np.stack([x[i] for i in pick])).unsqueeze(1)
    m = panel_mask(b)[:, 0].numpy()
    fig, ax = plt.subplots(4, 4, figsize=(14, 11))
    for k, i in enumerate(pick):
        a = ax[k // 4 * 2, k % 4]
        a.imshow(x[i], cmap="gray"); a.set_title(f"{meta[i]['label']} {Path(meta[i]['rel_path']).name}", fontsize=8); a.axis("off")
        a2 = ax[k // 4 * 2 + 1, k % 4]
        a2.imshow(x[i], cmap="gray"); a2.imshow(np.ma.masked_where(m[k], np.ones_like(m[k])), cmap="autumn", alpha=.45, vmin=0, vmax=1)
        if meta[i]["label"] == "faulty":
            a2.contour(box_mask((ROOT / meta[i]["rel_path"]).with_suffix(".json"), 2), levels=[.5], colors="cyan", linewidths=1)
        a2.set_title(f"orange = masked out ({100*(1-m[k].mean()):.0f}%)", fontsize=8); a2.axis("off")
    plt.tight_layout(); plt.savefig(ROOT / "results" / "india" / f"panel_mask_check_v{VERSION}.png", dpi=80)
    print(f"saved results/india/panel_mask_check_v{VERSION}.png")
