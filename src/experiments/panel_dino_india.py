"""
Panels vs ground from DINOv2 features, then score only panel patches.
Plot 16 only. Everything below was declared BEFORE running (Oct 25):

Mask (no fault labels used to build it; the fit is hlab.panels.dino_panel_fit)
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
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hlab.bundle import Bundle  # noqa: E402
from hlab.datasets import Dataset  # noqa: E402
from hlab.metrics import auroc, auroc_ci, paired_delta_ci  # noqa: E402
from hlab.panels import GRID, box_mask, dino_panel_fit, patch_anomaly, peak_ratio, peak_standout, resolve_device  # noqa: E402
from hlab.paths import RESULTS, ROOT  # noqa: E402

RES = RESULTS / "india"
SEED, N_CLUSTER_FRAMES, N_BANK_FRAMES, N_BANK = 0, 300, 1500, 500_000
NEG = -1e9


def main(device=None):
    dev = resolve_device(device)
    ds = Dataset("india")
    xt = np.load(ds.array_path("train"), mmap_mode="r")
    pm = dino_panel_fit(xt, ds.stats, seed=SEED, n_bank_frames=N_BANK_FRAMES, n_cluster_frames=N_CLUSTER_FRAMES,
                        device=dev, bank_patches=N_BANK)
    print(f"k-means on {pm.info['n_patches']:,} healthy patches: cluster sizes {pm.info['sizes']}, "
          f"pixel spread {pm.info['spread']} -> panel cluster = {pm.panel_k}")

    s_ = ds.split("val")
    x, meta, y = s_.x, s_.meta, s_.labels
    vf, vm = [], []
    for i in range(0, len(x), 128):
        f, _ = pm.feats(x[i:i + 128])
        vm.append(pm.panel_of(f).cpu()); vf.append(f.cpu())
    vm = torch.cat(vm)                                       # (N,34,45) bool
    fi = np.where(y == 1)[0]; cov = []
    for k in fi:
        bm = torch.from_numpy(box_mask((ROOT / meta[k]["rel_path"]).with_suffix(".json"), 0)).float()[None, None]
        pmk = Fn.adaptive_max_pool2d(bm, GRID)[0, 0] > 0
        cov.append(float(vm[k][pmk].float().mean()))
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

    ae = Bundle.load("india", dev).model
    s = {"dino_panel_max": [], "ae_panel_peak3": [], "input_panel_peak3 (no model)": []}
    with torch.no_grad():
        for c, i in enumerate(range(0, len(x), 128)):
            f = vf[c].to(dev); m = vm[i:i + 128].to(dev)
            d = patch_anomaly(f, pm.bank)
            s["dino_panel_max"].append(torch.where(m.view(len(f), -1), d, torch.full_like(d, NEG)).max(1).values.cpu())
            b = torch.from_numpy(x[i:i + 128]).unsqueeze(1).to(dev)
            mp = Fn.interpolate(m[:, None].float(), size=b.shape[2:], mode="nearest") > 0
            s["ae_panel_peak3"].append(peak_ratio((ae(b) - b) ** 2, mp).cpu())
            s["input_panel_peak3 (no model)"].append(peak_standout(b, mp, eps=1e-9).cpu())
    s = {k: torch.cat(v).numpy() for k, v in s.items()}
    print(f"PANEL-ONLY SCORES on VAL (plot 16): {int(y.sum())} faulty / {int((y == 0).sum())} healthy")
    for k, v in s.items():
        au = auroc(v, y); lo, hi = auroc_ci(v, y, n_boot=1000, seed=0)
        print(f"  {k:30s} AUROC {au:.3f}  (95% CI {lo:.3f} to {hi:.3f})")
    q = s["input_panel_peak3 (no model)"]
    for k in ("dino_panel_max", "ae_panel_peak3"):
        d, lo, hi = paired_delta_ci(q, s[k], y, n_boot=2000, seed=SEED)
        print(f"  {k} minus no-model rule: {d:+.3f}  (95% CI {lo:+.3f} to {hi:+.3f})")
    np.save(RES / "val_panel_scores.npy", s, allow_pickle=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default=None)
    main(ap.parse_args().device)
