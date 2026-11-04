"""
Fault-level and image-level comparison of every India model, plot 16 only.

Frame labels on plot 16 are unreliable (healthy-labelled frames with glowing
modules), so both modes grade each model on what we KNOW: the annotators' boxes
on the 121 faulty frames. Panel pixels only (stage-1 mask).

  models   A original autoencoder (stretched 8-bit input)        bundle india
           B autoencoder, real temperatures                       bundle india_temp
           C autoencoder, real temperatures, panels only          bundle india_panel
           D hidden-patch model, real temperatures, panels only   bundle india_inpaint
           R no-model rule: degrees above the 15x15 local panel mean, clipped at 0

`pixel` (Oct 26 evening, formerly compare_fault_level.py): per faulty frame, score every
panel pixel with the model's 3x3-averaged error map, then
    pixel AUROC   box pixels (pad 1) vs the other panel pixels in that frame; median over frames
    hit           the single highest-scoring panel pixel lies in a box (pad 2); chance = box share
    in/out ratio  median error inside boxes (pad 0) / median outside; median over frames
Output: results/india_temp/fault_level_compare.txt (stdout)

`image` (Oct 27, formerly image_score_v2.py): image-level score v2 ("local standout").
The old image-level score took the single highest 3x3 error pixel on the panels,
which landed in a fault box only 8 to 22% of the time. Declared before running:
    v2 score per frame = max over ELIGIBLE pixels of
        (e3 - local mean of e3 over the 31x31 panel neighbourhood) / (frame median of e3 on panels)
    eligible = panel pixels at least 5 px from the panel-mask edge (11x11 erosion)
    e3       = 3x3-averaged squared error of the model (rule: degrees above local panel mean)
Reported for each model: image-level AUROC with the OLD score (max/median) and with
the v2 score, 95% CI; on faulty frames, how often the v2 top spot lies in a fault box
(pad 2), vs chance. Output: results/india_temp/image_score_v2.txt (stdout)

  python src/experiments/fault_level_india.py pixel
  python src/experiments/fault_level_india.py image
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hlab.bundle import Bundle  # noqa: E402
from hlab.datasets import Dataset  # noqa: E402
from hlab.metrics import auroc, auroc_ci  # noqa: E402
from hlab.panels import blank_ground, box_mask, error_maps, local_excess, local_mean, resolve_device  # noqa: E402
from hlab.paths import RESULTS, ROOT  # noqa: E402

RES = RESULTS / "india_temp"
WIN, ERODE = 31, 11          # v2: local neighbourhood, edge erosion


def rule_maps(xc, pm, dev, win=15):
    """R: degrees above the win x win local panel mean (3x3-smoothed), clipped at 0."""
    t = torch.from_numpy(xc).unsqueeze(1).to(dev); m = torch.from_numpy(pm).unsqueeze(1).float().to(dev)
    d, _ = local_excess(t, m, win)
    return d.clamp_min(0)[:, 0].cpu().numpy()


def inputs(dev, faulty_only):
    """(meta, labels, panel masks, degrees C, z-scored, 8-bit frames) of VAL, the model table, chance of a box hit."""
    dt, d8 = Dataset("india_temp"), Dataset("india")
    meta = dt.meta("val")
    assert [m["rel_path"] for m in meta] == [m["rel_path"] for m in d8.meta("val")]
    y = np.array([1 if m["label"] == "faulty" else 0 for m in meta])
    sel = np.where(y == 1)[0] if faulty_only else np.arange(len(y))
    pm = dt.masks("val")[sel]
    xc = np.load(dt.array_path("val"))[sel]
    xz = dt.split("val").x[sel]
    x8 = d8.split("val").x[sel]
    models = {
        "A original AE (8-bit)": lambda: error_maps(Bundle.load("india", dev).model, x8, dev),
        "B AE, real temperature": lambda: error_maps(Bundle.load("india_temp", dev).model, xz, dev),
        "C AE, temperature, panels": lambda: error_maps(Bundle.load("india_panel", dev).model, blank_ground(xz, pm), dev),
        "D hidden-patch, panels": lambda: error_maps(Bundle.load("india_inpaint", dev).model, blank_ground(xz, pm), dev, inpaint=True),
        "R rule: deg above local": lambda: rule_maps(xc, pm, dev),
    }
    return meta, y, sel, pm, models


def pixel(dev):
    meta, y, fi, pm, models = inputs(dev, faulty_only=True)
    boxes0 = [box_mask((ROOT / meta[i]["rel_path"]).with_suffix(".json"), 0) for i in fi]
    boxes1 = [box_mask((ROOT / meta[i]["rel_path"]).with_suffix(".json"), 1) for i in fi]
    boxes2 = [box_mask((ROOT / meta[i]["rel_path"]).with_suffix(".json"), 2) for i in fi]
    chance = np.mean([(b2 & p).sum() / max(p.sum(), 1) for b2, p in zip(boxes2, pm)])
    print(f"FAULT-LEVEL COMPARISON, plot 16, {len(fi)} faulty frames, panel pixels only")
    print(f"{'model':28s} {'pixel AUROC':>12s} {'hit rate':>10s} {'in/out':>8s}   (hit chance {100*chance:.1f}%)")
    for name, fn in models.items():
        e = fn()
        au, hit, ratio = [], 0, []
        for j in range(len(fi)):
            p = pm[j] if pm[j].sum() > 50 else np.ones_like(pm[j])
            s, b1, b0, b2 = e[j][p], boxes1[j][p], boxes0[j][p], boxes2[j][p]
            if b1.any() and (~b1).any(): au.append(auroc(s, b1.astype(int)))
            hit += bool(b2[s.argmax()])
            if b0.any() and (~b0).any(): ratio.append(np.median(s[b0]) / max(np.median(s[~b0]), 1e-12))
        print(f"{name:28s} {np.median(au):12.3f} {100*hit/len(fi):9.1f}% {np.median(ratio):8.2f}")
    print("\npixel AUROC: 0.5 = the model's error is no higher at faults than elsewhere on the panel; 1.0 = faults are exactly its hottest error")


def v2_scores(e, pm, dev):
    """e: (N,H,W) error maps, pm: (N,H,W) bool. Returns (v2 score, old score, argmax index) per frame."""
    v2, old, arg = [], [], []
    for i in range(0, len(e), 64):
        t = torch.from_numpy(e[i:i + 64]).unsqueeze(1).to(dev)
        m = torch.from_numpy(pm[i:i + 64]).unsqueeze(1).float().to(dev)
        ref, _ = local_mean(t, m, WIN, eps=1.0, count=True)
        stand = t - ref
        elig = (-Fn.max_pool2d(-m, ERODE, 1, ERODE // 2)) > 0.5
        for j in range(len(t)):
            mj, ej = m[j, 0] > 0.5, elig[j, 0]
            if ej.sum() < 50: ej = mj if mj.sum() >= 50 else torch.ones_like(mj)
            med = t[j, 0][mj].median() if mj.sum() >= 50 else t[j, 0].median()
            s = torch.where(ej, stand[j, 0], torch.full_like(stand[j, 0], -1e9))
            v2.append(float(s.max() / med.clamp_min(1e-12))); arg.append(int(s.argmax()))
            old.append(float(t[j, 0][mj if mj.sum() >= 50 else torch.ones_like(mj)].max() / med.clamp_min(1e-12)))
    return np.array(v2), np.array(old), np.array(arg)


def image(dev):
    meta, y, _all, pm, models = inputs(dev, faulty_only=False)
    fi = np.where(y == 1)[0]
    boxes2 = {i: box_mask((ROOT / meta[i]["rel_path"]).with_suffix(".json"), 2) for i in fi}
    chance = np.mean([(boxes2[i] & pm[i]).sum() / max(pm[i].sum(), 1) for i in fi])
    print(f"IMAGE-LEVEL SCORE v2 (local standout), plot 16 validation plot: {len(fi)} faulty / {len(y) - len(fi)} healthy")
    print(f"{'model':28s} {'old score':>22s} {'v2 score':>22s} {'v2 top spot in box':>20s}  (chance {100*chance:.1f}%)")
    for name, fn in models.items():
        e = fn(); v2, old, arg = v2_scores(e, pm, dev)
        a_old, (lo1, hi1) = auroc(old, y), auroc_ci(old, y, n_boot=1000, seed=0)
        a_v2, (lo2, hi2) = auroc(v2, y), auroc_ci(v2, y, n_boot=1000, seed=0)
        hit = np.mean([bool(boxes2[i].flat[arg[i]]) for i in fi])
        print(f"{name:28s} {a_old:.3f} ({lo1:.3f} to {hi1:.3f})   {a_v2:.3f} ({lo2:.3f} to {hi2:.3f})   {100*hit:17.1f}%", flush=True)
        np.save(RES / f"val_v2_{name[0]}.npy", v2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["pixel", "image"]); ap.add_argument("--device", default=None)
    a = ap.parse_args(); dev = resolve_device(a.device)
    pixel(dev) if a.mode == "pixel" else image(dev)
