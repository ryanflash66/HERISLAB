"""
Image-level score v2 ("local standout"), plot 16 (validation plot) only. Oct 27.

The old image-level score took the single highest 3x3 error pixel on the panels,
which landed in a fault box only 8 to 22% of the time. Declared before running:

  v2 score per frame = max over ELIGIBLE pixels of
        (e3 - local mean of e3 over the 31x31 panel neighbourhood) / (frame median of e3 on panels)
  eligible = panel pixels at least 5 px from the panel-mask edge (11x11 erosion)
  e3       = 3x3-averaged squared error of the model (rule: degrees above local panel mean)

Reported for each of the 4 models and the no-model rule:
  image-level AUROC with the OLD score (max/median) and with the v2 score, 95% CI
  on faulty frames: how often the v2 top spot lies in a fault box (pad 2), vs chance
Output: results/india_temp/image_score_v2.txt (stdout)
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from box_check_india import box_mask  # noqa: E402
from compare_fault_level import D8, DT, M, ROOT, ae_maps, blank, rule_maps  # noqa: E402
from evaluate_india import auroc, boot_ci  # noqa: E402

WIN, ERODE = 31, 11


def v2_scores(e, pm, dev):
    """e: (N,H,W) error maps, pm: (N,H,W) bool. Returns (v2 score, old score, argmax index) per frame."""
    v2, old, arg = [], [], []
    k = torch.ones(1, 1, WIN, WIN, device=dev); p = WIN // 2
    for i in range(0, len(e), 64):
        t = torch.from_numpy(e[i:i + 64]).unsqueeze(1).to(dev)
        m = torch.from_numpy(pm[i:i + 64]).unsqueeze(1).float().to(dev)
        num = Fn.conv2d(Fn.pad(t * m, (p, p, p, p), mode="replicate"), k)
        den = Fn.conv2d(Fn.pad(m, (p, p, p, p), mode="replicate"), k).clamp_min(1.0)
        stand = t - num / den
        elig = (-Fn.max_pool2d(-m, ERODE, 1, ERODE // 2)) > 0.5
        for j in range(len(t)):
            mj, ej = m[j, 0] > 0.5, elig[j, 0]
            if ej.sum() < 50: ej = mj if mj.sum() >= 50 else torch.ones_like(mj)
            med = t[j, 0][mj].median() if mj.sum() >= 50 else t[j, 0].median()
            s = torch.where(ej, stand[j, 0], torch.full_like(stand[j, 0], -1e9))
            v2.append(float(s.max() / med.clamp_min(1e-12))); arg.append(int(s.argmax()))
            old.append(float(t[j, 0][mj if mj.sum() >= 50 else torch.ones_like(mj)].max() / med.clamp_min(1e-12)))
    return np.array(v2), np.array(old), np.array(arg)


def main():
    dev = torch.device("cuda")
    meta = list(csv.DictReader(open(DT / "val_meta.csv", encoding="utf-8")))
    y = np.array([1 if m["label"] == "faulty" else 0 for m in meta])
    pm = np.load(DT / "val_panelmask.npy"); xc = np.load(DT / "val.npy")
    ns = np.load(DT / "norm_stats.npy", allow_pickle=True).item()
    xz = ((xc - ns["mean"]) / ns["std"]).astype(np.float32); x8 = np.load(D8 / "val.npy")
    fi = np.where(y == 1)[0]
    boxes2 = {i: box_mask((ROOT / meta[i]["rel_path"]).with_suffix(".json"), 2) for i in fi}
    chance = np.mean([(boxes2[i] & pm[i]).sum() / max(pm[i].sum(), 1) for i in fi])
    models = {
        "A original AE (8-bit)": lambda: ae_maps(M / "autoencoder_india_best.pth", x8, dev),
        "B AE, real temperature": lambda: ae_maps(M / "autoencoder_india_temp_best.pth", xz, dev),
        "C AE, temperature, panels": lambda: ae_maps(M / "autoencoder_india_panel_best.pth", blank(xz, pm), dev),
        "D hidden-patch, panels": lambda: ae_maps(M / "inpaint_india_best.pth", blank(xz, pm), dev, inpaint=True),
        "R rule: deg above local": lambda: rule_maps(xc, pm, dev),
    }
    print(f"IMAGE-LEVEL SCORE v2 (local standout), plot 16 validation plot: {len(fi)} faulty / {len(y) - len(fi)} healthy")
    print(f"{'model':28s} {'old score':>22s} {'v2 score':>22s} {'v2 top spot in box':>20s}  (chance {100*chance:.1f}%)")
    for name, fn in models.items():
        e = fn(); v2, old, arg = v2_scores(e, pm, dev)
        a_old, (lo1, hi1) = auroc(old, y), boot_ci(old, y)
        a_v2, (lo2, hi2) = auroc(v2, y), boot_ci(v2, y)
        hit = np.mean([bool(boxes2[i].flat[arg[i]]) for i in fi])
        print(f"{name:28s} {a_old:.3f} ({lo1:.3f} to {hi1:.3f})   {a_v2:.3f} ({lo2:.3f} to {hi2:.3f})   {100*hit:17.1f}%", flush=True)
        np.save(DT.parent.parent.parent / "results" / "india_temp" / f"val_v2_{name[0]}.npy", v2)


if __name__ == "__main__":
    main()
