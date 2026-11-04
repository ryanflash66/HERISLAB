"""
Stage 1 of the pipeline: panel masks for every India frame (Oct 26).

Uses the DINOv2 panel/ground split from panel_dino_india.py (k-means on healthy
plot 17 patches, no fault labels), computed from the stretched 8-bit input in
data/CA_Preprocessed/india (same frame order as india_temp; cross_site is
matched by rel_path because india_temp drops the 17 FLIR Vue Pro frames).

Output: data/CA_Preprocessed/india_temp/{split}_panelmask.npy  bool (N, 240, 320)
        + a one-line summary per split
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_dino_india import (DATA, N_BANK_FRAMES, N_CLUSTER_FRAMES, SEED,  # noqa: E402
                              feats, kmeans2, panel_of)

ROOT = Path(__file__).resolve().parent.parent
TEMP = ROOT / "data" / "CA_Preprocessed" / "india_temp"


def main():
    dev = torch.device("cuda"); rng = np.random.default_rng(SEED)
    ns = np.load(DATA / "norm_stats.npy", allow_pickle=True).item(); mean, std = float(ns["mean"]), float(ns["std"])
    net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(dev).eval()
    xt = np.load(DATA / "train.npy", mmap_mode="r")
    pick = np.sort(rng.choice(len(xt), N_BANK_FRAMES, replace=False))
    cl = rng.choice(N_BANK_FRAMES, N_CLUSTER_FRAMES, replace=False)
    f, s = feats(net, xt[pick[cl]], mean, std, dev)
    z = f.reshape(-1, 384).float(); sp = s.reshape(-1)
    cent = kmeans2(z, seed=SEED); a = torch.cdist(z, cent).argmin(1)
    panel_k = int(np.argmin([float(sp[a == k].mean()) for k in range(2)]))
    del z, sp, a, f, s

    for split in ("train", "val", "test", "cross_site"):
        x = np.load(DATA / f"{split}.npy", mmap_mode="r")
        meta = list(csv.DictReader(open(DATA / f"{split}_meta.csv", encoding="utf-8")))
        tmeta = list(csv.DictReader(open(TEMP / f"{split}_meta.csv", encoding="utf-8")))
        idx = {m["rel_path"]: i for i, m in enumerate(meta)}
        order = [idx[m["rel_path"]] for m in tmeta]                 # align with india_temp
        masks = []
        for i in range(0, len(order), 128):
            b = np.stack([x[j] for j in order[i:i + 128]])
            fv, _ = feats(net, b, mean, std, dev)
            m = panel_of(fv, cent, panel_k).float()[:, None]
            masks.append((Fn.interpolate(m, size=(240, 320), mode="nearest") > 0.5)[:, 0].cpu().numpy())
        masks = np.concatenate(masks)
        np.save(TEMP / f"{split}_panelmask.npy", masks)
        print(f"{split}: {masks.shape}, panel share median {100*np.median(masks.mean((1, 2))):.0f}%, "
              f"frames with <10% panel: {int((masks.mean((1, 2)) < 0.10).sum())}", flush=True)


if __name__ == "__main__":
    main()
