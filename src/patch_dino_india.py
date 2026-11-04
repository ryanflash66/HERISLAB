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
             one patch ~ 7x7 px of the 240x320 frame)
  bank       1,500 random train frames, then 500,000 random patches of those
  primary    dino_max   = worst patch anomaly in the frame
  secondary  dino_top1  = mean of the worst 1% of patches
  baseline   input_peak3 (no model), same as score_peak_india.py
  extra      does the worst patch land on an annotator fault box? (vs chance)

Outputs: results/india/dino_val_log.txt (stdout), results/india/val_dino_max.npy,
         results/india/dino_val_maps.png
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from box_check_india import box_mask  # noqa: E402
from evaluate_india import ROOT, auroc, boot_ci, load  # noqa: E402

DATA = ROOT / "data" / "CA_Preprocessed" / "india"
RES = ROOT / "results" / "india"
SIZE, GRID = (476, 630), (34, 45)
N_BANK_FRAMES, N_BANK_PATCHES, SEED = 1500, 500_000, 0


def prep(x, mean, std, dev):
    b = torch.from_numpy(np.ascontiguousarray(x)).to(dev).float() * std + mean
    b = (b / 255.0).clamp(0, 1).unsqueeze(1).repeat(1, 3, 1, 1)
    b = Fn.interpolate(b, size=SIZE, mode="bilinear", align_corners=False)
    m = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    s = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    return (b - m) / s


@torch.no_grad()
def feats(net, x, mean, std, dev, bs=16):
    out = []
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            f = net.forward_features(prep(x[i:i + bs], mean, std, dev))["x_norm_patchtokens"]
        out.append(Fn.normalize(f.float(), dim=-1).half())
    return torch.cat(out)                                  # (N, 1530, 384)


@torch.no_grad()
def patch_scores(q, bank, chunk=2048):
    """q: (N,P,D) normalised; returns (N,P) = 1 - max cosine sim to bank."""
    N, P, D = q.shape; flat = q.reshape(-1, D); out = torch.empty(len(flat), device=q.device)
    for i in range(0, len(flat), chunk):
        out[i:i + chunk] = 1 - (flat[i:i + chunk] @ bank.T).max(1).values.float()
    return out.view(N, P)


def main():
    dev = torch.device("cuda")
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    ns = np.load(DATA / "norm_stats.npy", allow_pickle=True).item(); mean, std = float(ns["mean"]), float(ns["std"])
    net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(dev).eval()

    xt = np.load(DATA / "train.npy", mmap_mode="r")
    pick = np.sort(rng.choice(len(xt), N_BANK_FRAMES, replace=False))
    ft = feats(net, xt[pick], mean, std, dev).reshape(-1, 384)
    bank = ft[torch.from_numpy(rng.choice(len(ft), N_BANK_PATCHES, replace=False)).to(dev)]
    del ft; torch.cuda.empty_cache()
    print(f"bank: {N_BANK_PATCHES:,} patches from {N_BANK_FRAMES} healthy train frames (plot 17)")

    x, meta, y = load("val")
    maps = []
    for i in range(0, len(x), 128):
        maps.append(patch_scores(feats(net, x[i:i + 128], mean, std, dev), bank).cpu())
    maps = torch.cat(maps).numpy()                         # (N, 1530)
    s = {"dino_max": maps.max(1),
         "dino_top1": np.sort(maps, 1)[:, -int(0.01 * maps.shape[1]):].mean(1)}
    b = torch.from_numpy(x).unsqueeze(1); xf = b.flatten(1)
    s["input_peak3 (no model)"] = ((Fn.avg_pool2d(b, 3, 1, 1).flatten(1).max(1).values - xf.median(1).values) / xf.std(1)).numpy()

    print(f"DINOv2 PATCH MATCHING on VAL (plot 16): {int(y.sum())} faulty / {int((y == 0).sum())} healthy"
          f"   (0.5 = coin flip; autoencoder peak score was 0.640)")
    for k, v in s.items():
        a = auroc(v, y); lo, hi = boot_ci(v, y)
        print(f"  {k:24s} AUROC {a:.3f}  (95% CI {lo:.3f} to {hi:.3f}){'  <- primary' if k == 'dino_max' else ''}")
    p, q = s["dino_max"], s["input_peak3 (no model)"]
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; d = []
    for _ in range(2000):
        idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        d.append(auroc(p[idx], y[idx]) - auroc(q[idx], y[idx]))
    print(f"  DINOv2 minus no-model rule: {auroc(p, y) - auroc(q, y):+.3f}  (95% CI {np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f})")

    # does the worst patch land on a fault box?
    hits, chance, keep = 0, [], []
    for k in i1:
        bm = torch.from_numpy(box_mask((ROOT / meta[k]["rel_path"]).with_suffix(".json"), 0)).float()[None, None]
        pm = (Fn.adaptive_max_pool2d(bm, GRID)[0, 0].flatten() > 0).numpy()
        hits += bool(pm[maps[k].argmax()]); chance.append(pm.mean())
        if len(keep) < 6: keep.append((k, pm))
    print(f"  worst patch lands on a fault box: {hits}/{len(i1)} = {100*hits/len(i1):.1f}%  (chance {100*np.mean(chance):.1f}%)")
    np.save(RES / "val_dino_max.npy", p)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(keep), 2, figsize=(9, 3.3 * len(keep)))
    for r, (k, pm) in enumerate(keep):
        ax[r, 0].imshow(x[k], cmap="gray"); ax[r, 0].set_title("input " + Path(meta[k]["rel_path"]).name, fontsize=9)
        ax[r, 1].imshow(maps[k].reshape(GRID), cmap="inferno"); ax[r, 1].set_title("DINOv2 patch anomaly", fontsize=9)
        ax[r, 1].contour(pm.reshape(GRID), levels=[.5], colors="cyan", linewidths=1)
        ax[r, 0].contour(box_mask((ROOT / meta[k]["rel_path"]).with_suffix(".json"), 1), levels=[.5], colors="cyan", linewidths=1)
        for a in ax[r]: a.axis("off")
    plt.tight_layout(); plt.savefig(RES / "dino_val_maps.png", dpi=85)
    print("figure: results/india/dino_val_maps.png")


if __name__ == "__main__":
    main()
