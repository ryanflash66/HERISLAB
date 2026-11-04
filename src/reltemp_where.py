"""
Diagnostic for reltemp_india.py (no new score): where does the hottest relative
spot sit? Healthy frames show ~11 K spots, so check whether they are at the
panel/ground border (mask leak) or inside panels.
Outputs: stdout, results/india/reltemp_where.png
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from box_check_india import box_mask  # noqa: E402
from evaluate_india import ROOT, load  # noqa: E402
from flir_rjpg import raw_to_celsius, read_rjpg  # noqa: E402
from reltemp_india import MIN_PANEL, RES, WIN  # noqa: E402


def spot(i, meta, masks, dev):
    raw, calib = read_rjpg(ROOT / meta[i]["rel_path"])
    t = torch.from_numpy(raw_to_celsius(raw, calib)).to(dev)[None, None]
    m = Fn.interpolate(torch.from_numpy(masks[i]).float()[None, None].to(dev), size=t.shape[2:], mode="nearest")
    t3 = Fn.avg_pool2d(Fn.pad(t, (1, 1, 1, 1), mode="replicate"), 3, 1)
    p = WIN // 2; k = torch.ones(1, 1, WIN, WIN, device=dev) / WIN ** 2
    num = Fn.conv2d(Fn.pad(t * m, (p, p, p, p), mode="replicate"), k)
    den = Fn.conv2d(Fn.pad(m, (p, p, p, p), mode="replicate"), k)
    ok = (m > 0) & (den >= MIN_PANEL)
    d = torch.where(ok, t3 - num / den.clamp_min(1e-6), torch.full_like(t, -1e9))
    j = int(d.flatten().argmax()); r, c = divmod(j, t.shape[3])
    # distance (px) from the spot to the nearest non-panel pixel
    nm = (m[0, 0] == 0).nonzero().float()
    dist = float(((nm - torch.tensor([r, c], device=dev).float()) ** 2).sum(1).min().sqrt()) if len(nm) else 999.0
    return t[0, 0].cpu().numpy(), m[0, 0].cpu().numpy(), r, c, float(d.max()), dist


def main():
    dev = torch.device("cuda")
    x, meta, y = load("val")
    masks = np.load(RES / "val_panel_mask.npy")
    sc = {r["rel_path"]: float(r["rel_local_K"]) for r in csv.DictReader(open(RES / "val_reltemp_scores.csv", encoding="utf-8"))}
    v = np.array([sc[m["rel_path"]] for m in meta])
    he = np.where(y == 0)[0]; fa = np.where(y == 1)[0]
    rng = np.random.default_rng(2)
    samp = {"healthy": rng.choice(he, 200, replace=False), "faulty": fa}
    for name, idx in samp.items():
        dd = [spot(i, meta, masks, dev)[5] for i in idx]
        dd = np.array(dd)
        print(f"{name}: hottest relative spot within 14 px (one patch) of ground: {100*np.mean(dd <= 14):.0f}% of {len(dd)} frames; median distance {np.median(dd):.0f} px")
    hits = 0
    for i in fa:
        _, _, r, c, _, _ = spot(i, meta, masks, dev)
        bm = box_mask((ROOT / meta[i]["rel_path"]).with_suffix(".json"), 0)
        hits += bool(bm[min(239, r * 240 // 512), min(319, c * 320 // 640)])
    print(f"faulty: hottest relative spot lands in a fault box {hits}/{len(fa)}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    top_h = he[np.argsort(-v[he])[:4]]; top_f = fa[np.argsort(-v[fa])[:4]]
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.5))
    for n, i in enumerate(list(top_h) + list(top_f)):
        t, m, r, c, dk, dist = spot(i, meta, masks, dev)
        A = ax[n // 4, n % 4]; A.imshow(t, cmap="inferno")
        A.contour(m, levels=[.5], colors="lime", linewidths=.7)
        A.plot(c, r, "o", mfc="none", mec="cyan", ms=16, mew=2)
        A.set_title(f"{meta[i]['label']} {Path(meta[i]['rel_path']).name}  +{dk:.1f} K, {dist:.0f}px from ground", fontsize=8); A.axis("off")
    plt.tight_layout(); plt.savefig(RES / "reltemp_where.png", dpi=80)
    print("figure: results/india/reltemp_where.png  (top row: 4 healthy frames with the biggest spots; bottom: 4 faulty)")


if __name__ == "__main__":
    main()
