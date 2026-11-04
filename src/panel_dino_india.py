"""
Panels vs ground from DINOv2 features, then score only panel patches.
Plot 16 only. Everything below was declared BEFORE running (Oct 25):

Mask (no fault labels used to build it)
  - DINOv2 ViT-S/14 patch features (same input as patch_dino_india.py)
  - k-means, k=2, on all patches of 300 random healthy TRAIN frames (plot 17)
  - the PANEL cluster is the one whose patches have the lower average pixel
    spread inside the patch (panels are smooth, gravel and plants are rough)
  - hole fill: a patch also counts as panel if >= 5 of its 8 neighbours are panel
    (so a hot spot that looks unlike normal panel is not thrown out)
Pass bar, both required, checked BEFORE any score is computed
  - keeps >= 90% of fault-box patches (mean over the 121 faulty frames)
  - removes >= 25% of the frame (median over all plot 16 frames)
Scores (only if the bar is passed)
  - dino_panel_max   worst panel patch vs a bank of 500,000 PANEL patches from plot 17
  - ae_panel_peak3   autoencoder peak score, panel pixels only
  - input_panel_peak3 (no model), panel pixels only  <- the baseline both must beat
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from box_check_india import box_mask  # noqa: E402
from evaluate_india import ROOT, WEIGHTS, auroc, boot_ci, load  # noqa: E402
from train_autoencoder import ThermalAutoencoder  # noqa: E402

DATA, RES = ROOT / "data" / "CA_Preprocessed" / "india", ROOT / "results" / "india"
SIZE, GRID, PS = (476, 630), (34, 45), 14
SEED, N_CLUSTER_FRAMES, N_BANK_FRAMES, N_BANK = 0, 300, 1500, 500_000
NEG = -1e9


def to01(x, mean, std, dev):
    b = torch.from_numpy(np.ascontiguousarray(x)).to(dev).float() * std + mean
    return Fn.interpolate((b / 255).clamp(0, 1).unsqueeze(1), size=SIZE, mode="bilinear", align_corners=False)


@torch.no_grad()
def feats(net, x, mean, std, dev, bs=16):
    """Returns normalised features (N,P,384) fp16 and per-patch pixel spread (N,P)."""
    F_, S_ = [], []
    m = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    s = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    for i in range(0, len(x), bs):
        g = to01(x[i:i + bs], mean, std, dev)
        S_.append(Fn.unfold(g, PS, stride=PS).std(1))
        with torch.autocast("cuda", dtype=torch.float16):
            f = net.forward_features((g.repeat(1, 3, 1, 1) - m) / s)["x_norm_patchtokens"]
        F_.append(Fn.normalize(f.float(), dim=-1).half())
    return torch.cat(F_), torch.cat(S_)


def kmeans2(z, iters=50, restarts=5, seed=0):
    g = torch.Generator(device=z.device).manual_seed(seed); best = None
    for _ in range(restarts):
        c = z[torch.randint(len(z), (2,), generator=g, device=z.device)].clone()
        for _ in range(iters):
            a = torch.cdist(z, c).argmin(1)
            c = torch.stack([z[a == k].mean(0) if (a == k).any() else c[k] for k in range(2)])
        inert = torch.cdist(z, c).min(1).values.pow(2).sum()
        if best is None or inert < best[0]: best = (inert, c)
    return best[1]


def panel_of(f, cent, panel_k):
    """f: (N,P,D) -> bool (N,34,45) panel mask with hole fill."""
    a = torch.cdist(f.float().reshape(-1, f.shape[-1]), cent).argmin(1).view(len(f), *GRID)
    m = (a == panel_k).float().unsqueeze(1)
    nb = Fn.conv2d(Fn.pad(m, (1, 1, 1, 1), mode="replicate"), torch.ones(1, 1, 3, 3, device=m.device)) - m
    return ((m > 0) | (nb >= 5))[:, 0]


def peak3(v, m):
    """v: (N,1,H,W) map, m: bool (N,1,H,W). (max 3x3-avg inside m - median inside m) / std inside m."""
    v3 = Fn.avg_pool2d(v, 3, 1, 1); out = []
    for vi, v3i, mi in zip(v, v3, m):
        s = vi[mi] if mi.sum() > 50 else vi.flatten()
        out.append(((v3i[mi].max() if mi.any() else v3i.max()) - s.median()) / s.std().clamp_min(1e-9))
    return torch.stack(out)


def ratio3(v, m):
    """AE version, same as before: max 3x3-avg error inside m / median error inside m."""
    v3 = Fn.avg_pool2d(v, 3, 1, 1); out = []
    for vi, v3i, mi in zip(v, v3, m):
        s = vi[mi] if mi.sum() > 50 else vi.flatten()
        out.append((v3i[mi].max() if mi.any() else v3i.max()) / s.median().clamp_min(1e-12))
    return torch.stack(out)


def main():
    dev = torch.device("cuda"); rng = np.random.default_rng(SEED)
    ns = np.load(DATA / "norm_stats.npy", allow_pickle=True).item(); mean, std = float(ns["mean"]), float(ns["std"])
    net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(dev).eval()

    xt = np.load(DATA / "train.npy", mmap_mode="r")
    pick = np.sort(rng.choice(len(xt), N_BANK_FRAMES, replace=False))
    ft, st = feats(net, xt[pick], mean, std, dev)
    cl = rng.choice(N_BANK_FRAMES, N_CLUSTER_FRAMES, replace=False)
    z = ft[cl].reshape(-1, 384).float(); sp = st[cl].reshape(-1)
    cent = kmeans2(z, seed=SEED)
    a = torch.cdist(z, cent).argmin(1)
    spread = [float(sp[a == k].mean()) for k in range(2)]
    panel_k = int(np.argmin(spread))
    print(f"k-means on {len(z):,} healthy patches: cluster sizes {[(a == k).float().mean().item() for k in range(2)]}, "
          f"pixel spread {spread} -> panel cluster = {panel_k}")
    del z, sp, a

    x, meta, y = load("val")
    vf, vm = [], []
    for i in range(0, len(x), 128):
        f, _ = feats(net, x[i:i + 128], mean, std, dev)
        vm.append(panel_of(f, cent, panel_k).cpu()); vf.append(f.cpu())
    vm = torch.cat(vm)                                       # (N,34,45) bool
    fi = np.where(y == 1)[0]; cov, chance = [], []
    for k in fi:
        bm = torch.from_numpy(box_mask((ROOT / meta[k]["rel_path"]).with_suffix(".json"), 0)).float()[None, None]
        pm = Fn.adaptive_max_pool2d(bm, GRID)[0, 0] > 0
        cov.append(float(vm[k][pm].float().mean()))
    keep_frac = vm.float().mean((1, 2)).numpy()
    removed = 1 - float(np.median(keep_frac))
    ok = np.mean(cov) >= 0.90 and removed >= 0.25
    print(f"PASS BAR: fault-box patches kept {100*np.mean(cov):.1f}% (need >= 90%), "
          f"frame removed {100*removed:.1f}% median (need >= 25%), frames losing every box patch {sum(c == 0 for c in cov)}/{len(cov)} "
          f"-> {'PASSED' if ok else 'FAILED'}")

    # picture first, pass or fail
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    r2 = np.random.default_rng(1)
    he = np.where(y == 0)[0]; show = list(r2.choice(he, 4, replace=False)) + list(r2.choice(fi, 4, replace=False))
    fig, ax = plt.subplots(2, 4, figsize=(14, 6))
    for n, k in enumerate(show):
        A = ax[n // 4, n % 4]; A.imshow(x[k], cmap="gray")
        mm = Fn.interpolate(vm[k][None, None].float(), size=x[k].shape, mode="nearest")[0, 0].numpy()
        A.imshow(np.ma.masked_where(mm > 0, np.ones_like(mm)), cmap="autumn", alpha=.45, vmin=0, vmax=1)
        if y[k]: A.contour(box_mask((ROOT / meta[k]["rel_path"]).with_suffix(".json"), 1), levels=[.5], colors="cyan", linewidths=1)
        A.set_title(f"{'faulty' if y[k] else 'healthy'} {Path(meta[k]['rel_path']).name}  removed {100*(1-mm.mean()):.0f}%", fontsize=8); A.axis("off")
    plt.tight_layout(); plt.savefig(RES / "panel_dino_mask_check.png", dpi=80)
    print("figure: results/india/panel_dino_mask_check.png")
    if not ok:
        print("FAILED the pass bar: scores not computed."); return

    # bank of PANEL patches only
    tm = panel_of(ft, cent, panel_k).reshape(-1)
    pool = ft.reshape(-1, 384)[tm.to(dev)]
    bank = pool[torch.from_numpy(rng.choice(len(pool), min(N_BANK, len(pool)), replace=False)).to(dev)]
    del ft, pool; torch.cuda.empty_cache()

    ae = ThermalAutoencoder().to(dev); ae.load_state_dict(torch.load(WEIGHTS, map_location=dev)); ae.eval()
    s = {"dino_panel_max": [], "ae_panel_peak3": [], "input_panel_peak3 (no model)": []}
    with torch.no_grad():
        for c, i in enumerate(range(0, len(x), 128)):
            f = vf[c].to(dev); m = vm[i:i + 128].to(dev)
            d = torch.empty(f.shape[:2], device=dev); flat = f.reshape(-1, 384)
            for j in range(0, len(flat), 2048):
                d.view(-1)[j:j + 2048] = 1 - (flat[j:j + 2048] @ bank.T).max(1).values.float()
            s["dino_panel_max"].append(torch.where(m.view(len(f), -1), d, torch.full_like(d, NEG)).max(1).values.cpu())
            b = torch.from_numpy(x[i:i + 128]).unsqueeze(1).to(dev)
            mp = Fn.interpolate(m[:, None].float(), size=b.shape[2:], mode="nearest") > 0
            s["ae_panel_peak3"].append(ratio3((ae(b) - b) ** 2, mp).cpu())
            s["input_panel_peak3 (no model)"].append(peak3(b, mp).cpu())
    s = {k: torch.cat(v).numpy() for k, v in s.items()}
    print(f"PANEL-ONLY SCORES on VAL (plot 16): {int(y.sum())} faulty / {int((y == 0).sum())} healthy")
    for k, v in s.items():
        au = auroc(v, y); lo, hi = boot_ci(v, y)
        print(f"  {k:30s} AUROC {au:.3f}  (95% CI {lo:.3f} to {hi:.3f})")
    q = s["input_panel_peak3 (no model)"]; i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    for k in ("dino_panel_max", "ae_panel_peak3"):
        p = s[k]; dd = []
        for _ in range(2000):
            idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
            dd.append(auroc(p[idx], y[idx]) - auroc(q[idx], y[idx]))
        print(f"  {k} minus no-model rule: {auroc(p, y) - auroc(q, y):+.3f}  (95% CI {np.percentile(dd, 2.5):+.3f} to {np.percentile(dd, 97.5):+.3f})")
    np.save(RES / "val_panel_scores.npy", s, allow_pickle=True)


if __name__ == "__main__":
    main()
