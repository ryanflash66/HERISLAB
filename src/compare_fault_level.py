"""
Fault-level comparison of every model, plot 16 only (Oct 26 evening).

Frame labels on plot 16 are unreliable (healthy-labelled frames with glowing
modules), so this grades each model only on what we KNOW: the annotators' boxes
on the 121 faulty frames. Panel pixels only (stage-1 mask). Declared before running:

  per faulty frame, score every panel pixel with the model's 3x3-averaged error map, then
    pixel AUROC   box pixels (pad 1) vs the other panel pixels in that frame; median over frames
    hit           the single highest-scoring panel pixel lies in a box (pad 2); chance = box share
    in/out ratio  median error inside boxes (pad 0) / median outside; median over frames

  models   A original autoencoder (stretched 8-bit input)        models/autoencoder_india_best.pth
           B autoencoder, real temperatures                       models/autoencoder_india_temp_best.pth
           C autoencoder, real temperatures, panels only          models/autoencoder_india_panel_best.pth
           D hidden-patch model, real temperatures, panels only   models/inpaint_india_best.pth
           R no-model rule: degrees above the 15x15 local panel mean, clipped at 0
Output: results/india_temp/fault_level_compare.txt (stdout)
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from box_check_india import box_mask  # noqa: E402
from evaluate_india import auroc  # noqa: E402
from inpaint_india import predict  # noqa: E402
from train_autoencoder import ThermalAutoencoder  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D8 = ROOT / "data" / "CA_Preprocessed" / "india"
DT = ROOT / "data" / "CA_Preprocessed" / "india_temp"
M = ROOT / "models"


def blank(xz, pm):
    xz = xz.copy()
    for i in range(len(xz)):
        if pm[i].any(): xz[i][~pm[i]] = np.median(xz[i][pm[i]])
    return xz


def ae_maps(weights, x, dev, inpaint=False):
    net = ThermalAutoencoder().to(dev); net.load_state_dict(torch.load(weights, map_location=dev)); net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 32):
            b = torch.from_numpy(x[i:i + 32]).unsqueeze(1).to(dev)
            r = predict(net, b, dev) if inpaint else net(b)
            out.append(Fn.avg_pool2d((r - b) ** 2, 3, 1, 1)[:, 0].cpu().numpy())
    return np.concatenate(out)


def rule_maps(xc, pm, dev, win=15):
    t = torch.from_numpy(xc).unsqueeze(1).to(dev); m = torch.from_numpy(pm).unsqueeze(1).float().to(dev)
    k = torch.ones(1, 1, win, win, device=dev) / win ** 2; p = win // 2
    num = Fn.conv2d(Fn.pad(t * m, (p, p, p, p), mode="replicate"), k)
    den = Fn.conv2d(Fn.pad(m, (p, p, p, p), mode="replicate"), k).clamp_min(1e-6)
    d = (Fn.avg_pool2d(Fn.pad(t, (1, 1, 1, 1), mode="replicate"), 3, 1) - num / den).clamp_min(0)
    return d[:, 0].cpu().numpy()


def main():
    dev = torch.device("cuda")
    meta = list(csv.DictReader(open(DT / "val_meta.csv", encoding="utf-8")))
    meta8 = list(csv.DictReader(open(D8 / "val_meta.csv", encoding="utf-8")))
    assert [m["rel_path"] for m in meta] == [m["rel_path"] for m in meta8]
    fi = np.array([i for i, m in enumerate(meta) if m["label"] == "faulty"])
    pm = np.load(DT / "val_panelmask.npy")[fi]
    xc = np.load(DT / "val.npy")[fi]
    ns = np.load(DT / "norm_stats.npy", allow_pickle=True).item()
    xz = ((xc - ns["mean"]) / ns["std"]).astype(np.float32)
    x8 = np.load(D8 / "val.npy")[fi]
    boxes0 = [box_mask((ROOT / meta[i]["rel_path"]).with_suffix(".json"), 0) for i in fi]
    boxes1 = [box_mask((ROOT / meta[i]["rel_path"]).with_suffix(".json"), 1) for i in fi]
    boxes2 = [box_mask((ROOT / meta[i]["rel_path"]).with_suffix(".json"), 2) for i in fi]

    maps = {
        "A original AE (8-bit)": ae_maps(M / "autoencoder_india_best.pth", x8, dev),
        "B AE, real temperature": ae_maps(M / "autoencoder_india_temp_best.pth", xz, dev),
        "C AE, temperature, panels": ae_maps(M / "autoencoder_india_panel_best.pth", blank(xz, pm), dev),
        "D hidden-patch, panels": ae_maps(M / "inpaint_india_best.pth", blank(xz, pm), dev, inpaint=True),
        "R rule: deg above local": rule_maps(xc, pm, dev),
    }
    chance = np.mean([(b2 & p).sum() / max(p.sum(), 1) for b2, p in zip(boxes2, pm)])
    print(f"FAULT-LEVEL COMPARISON, plot 16, {len(fi)} faulty frames, panel pixels only")
    print(f"{'model':28s} {'pixel AUROC':>12s} {'hit rate':>10s} {'in/out':>8s}   (hit chance {100*chance:.1f}%)")
    for name, e in maps.items():
        au, hit, ratio = [], 0, []
        for j in range(len(fi)):
            p = pm[j] if pm[j].sum() > 50 else np.ones_like(pm[j])
            s, b1, b0, b2 = e[j][p], boxes1[j][p], boxes0[j][p], boxes2[j][p]
            if b1.any() and (~b1).any(): au.append(auroc(s, b1.astype(int)))
            hit += bool(b2[s.argmax()])
            if b0.any() and (~b0).any(): ratio.append(np.median(s[b0]) / max(np.median(s[~b0]), 1e-12))
        print(f"{name:28s} {np.median(au):12.3f} {100*hit/len(fi):9.1f}% {np.median(ratio):8.2f}")
    print("\npixel AUROC: 0.5 = the model's error is no higher at faults than elsewhere on the panel; 1.0 = faults are exactly its hottest error")


if __name__ == "__main__":
    main()
