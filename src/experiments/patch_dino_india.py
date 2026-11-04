"""
Step 2 after the box check: a detector that does not redraw the image.

Patch matching with a frozen, pretrained vision transformer (DINOv2 ViT-S/14),
in the style of PatchCore (Roth et al. 2022) and AnomalyDINO (Damm et al. 2024):
  1. cut every healthy TRAIN frame (plot 17) into patches and store each patch's
     DINOv2 feature in a memory bank (nothing is trained)
  2. for a new frame, each patch's anomaly = 1 - cosine similarity to its
     nearest healthy patch in the bank
  3. frame score = the worst patch

Declared BEFORE running (Oct 25), plot 16 only:
  input      preprocessed 240x320 frame, z-score undone to 0..1, copied to 3
             channels, ImageNet normalisation, resized to 476x630 (34x45 patches,
             one patch ~ 7x7 px of the 240x320 frame)  [hlab.panels.feats]
  bank       1,500 random train frames, then 500,000 random patches of those
  primary    dino_max   = worst patch anomaly in the frame
  secondary  dino_top1  = mean of the worst 1% of patches
  baseline   input_peak3 (no model), same as score_peak_india.py
  extra      does the worst patch land on an annotator fault box? (vs chance)

Outputs: results/india/dino_val_log.txt (stdout), results/india/val_dino_max.npy,
         results/india/dino_val_maps.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hlab.datasets import Dataset  # noqa: E402
from hlab.metrics import auroc, auroc_ci, paired_delta_ci  # noqa: E402
from hlab.panels import FEAT_DIM, GRID, box_mask, feats, load_dino, patch_anomaly, peak_standout, resolve_device  # noqa: E402
from hlab.paths import RESULTS, ROOT  # noqa: E402

RES = RESULTS / "india"
N_BANK_FRAMES, N_BANK_PATCHES, SEED = 1500, 500_000, 0


def main(device=None):
    dev = resolve_device(device)
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    ds = Dataset("india")
    stats = ds.stats
    net = load_dino(dev)

    xt = np.load(ds.array_path("train"), mmap_mode="r")
    pick = np.sort(rng.choice(len(xt), N_BANK_FRAMES, replace=False))
    ft = feats(net, xt[pick], stats, dev)[0].reshape(-1, FEAT_DIM)
    bank = ft[torch.from_numpy(rng.choice(len(ft), N_BANK_PATCHES, replace=False)).to(dev)]
    del ft
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    print(f"bank: {N_BANK_PATCHES:,} patches from {N_BANK_FRAMES} healthy train frames (plot 17)")

    s_ = ds.split("val")
    x, meta, y = s_.x, s_.meta, s_.labels
    maps = []
    for i in range(0, len(x), 128):
        maps.append(patch_anomaly(feats(net, x[i:i + 128], stats, dev)[0], bank).cpu())
    maps = torch.cat(maps).numpy()                         # (N, 1530)
    s = {"dino_max": maps.max(1),
         "dino_top1": np.sort(maps, 1)[:, -int(0.01 * maps.shape[1]):].mean(1)}
    s["input_peak3 (no model)"] = peak_standout(torch.from_numpy(x).unsqueeze(1)).numpy()

    print(f"DINOv2 PATCH MATCHING on VAL (plot 16): {int(y.sum())} faulty / {int((y == 0).sum())} healthy"
          f"   (0.5 = coin flip; autoencoder peak score was 0.640)")
    for k, v in s.items():
        a = auroc(v, y); lo, hi = auroc_ci(v, y, n_boot=1000, seed=0)
        print(f"  {k:24s} AUROC {a:.3f}  (95% CI {lo:.3f} to {hi:.3f}){'  <- primary' if k == 'dino_max' else ''}")
    d, lo, hi = paired_delta_ci(s["input_peak3 (no model)"], s["dino_max"], y, n_boot=2000, seed=SEED)
    print(f"  DINOv2 minus no-model rule: {d:+.3f}  (95% CI {lo:+.3f} to {hi:+.3f})")

    # does the worst patch land on a fault box?
    i1 = np.where(y == 1)[0]
    hits, chance, keep = 0, [], []
    for k in i1:
        bm = torch.from_numpy(box_mask((ROOT / meta[k]["rel_path"]).with_suffix(".json"), 0)).float()[None, None]
        pmk = (Fn.adaptive_max_pool2d(bm, GRID)[0, 0].flatten() > 0).numpy()
        hits += bool(pmk[maps[k].argmax()]); chance.append(pmk.mean())
        if len(keep) < 6: keep.append((k, pmk))
    print(f"  worst patch lands on a fault box: {hits}/{len(i1)} = {100*hits/len(i1):.1f}%  (chance {100*np.mean(chance):.1f}%)")
    np.save(RES / "val_dino_max.npy", s["dino_max"])

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(keep), 2, figsize=(9, 3.3 * len(keep)))
    for r, (k, pmk) in enumerate(keep):
        ax[r, 0].imshow(x[k], cmap="gray"); ax[r, 0].set_title("input " + Path(meta[k]["rel_path"]).name, fontsize=9)
        ax[r, 1].imshow(maps[k].reshape(GRID), cmap="inferno"); ax[r, 1].set_title("DINOv2 patch anomaly", fontsize=9)
        ax[r, 1].contour(pmk.reshape(GRID), levels=[.5], colors="cyan", linewidths=1)
        ax[r, 0].contour(box_mask((ROOT / meta[k]["rel_path"]).with_suffix(".json"), 1), levels=[.5], colors="cyan", linewidths=1)
        for a in ax[r]: a.axis("off")
    plt.tight_layout(); plt.savefig(RES / "dino_val_maps.png", dpi=85)
    print("figure: results/india/dino_val_maps.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default=None)
    main(ap.parse_args().device)
