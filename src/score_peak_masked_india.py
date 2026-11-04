"""
Step 1 after the box check: same peak scores as score_peak_india.py, but only
over panel pixels (src/panel_mask.py). Declared before running (Oct 25):
  primary   peak3_masked = max of 3x3-averaged error inside the mask / median error inside the mask
  baseline  input_peak3_masked (no model) = (max of 3x3-averaged input inside mask - median) / std, inside mask
Also reports how much of the annotators' fault boxes survive the mask, so we
know whether the mask is cutting faults out.
VAL (plot 16) only.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from box_check_india import box_mask  # noqa: E402
from evaluate_india import ROOT, WEIGHTS, auroc, boot_ci, load  # noqa: E402
from panel_mask import VERSION as MASK_VERSION, panel_mask  # noqa: E402
from train_autoencoder import ThermalAutoencoder  # noqa: E402

NEG = -1e9


def masked_stats(v, m):
    """v, m: (B,1,H,W). Returns per-frame (median, std) of v over m."""
    med, sd = [], []
    for vi, mi in zip(v, m):
        s = vi[mi]; s = s if s.numel() > 50 else vi.flatten()
        med.append(s.median()); sd.append(s.std())
    return torch.stack(med), torch.stack(sd)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThermalAutoencoder().to(dev)
    model.load_state_dict(torch.load(WEIGHTS, map_location=dev)); model.eval()
    x, meta, y = load("val")
    # pass bar first, no model and no scores: the mask must keep >= 90% of fault-box pixels
    fi = np.where(y == 1)[0]
    m = panel_mask(torch.from_numpy(x[fi]).unsqueeze(1).to(dev))[:, 0]
    cov = []
    for j, k in enumerate(fi):
        bm = torch.from_numpy(box_mask((ROOT / meta[k]["rel_path"]).with_suffix(".json"), 0)).to(dev)
        cov.append(float(m[j][bm].float().mean()))
    print(f"MASK v{MASK_VERSION} pass bar: fault-box pixels kept {100*np.mean(cov):.1f}% (need >= 90%); "
          f"frames losing every box pixel: {sum(c == 0 for c in cov)}/{len(cov)}")
    if np.mean(cov) < 0.90:
        print("FAILED the pass bar, scores not computed (as agreed: drop masking, go to step 2)"); return
    out = {"peak3_masked": [], "input_peak3_masked (no model)": [], "peak3 (unmasked, earlier)": []}
    kept = []
    with torch.no_grad():
        for i in range(0, len(x), 64):
            b = torch.from_numpy(x[i:i + 64]).unsqueeze(1).to(dev)
            m = panel_mask(b)
            e = (model(b) - b) ** 2
            e3 = Fn.avg_pool2d(e, 3, 1, 1)
            emed, _ = masked_stats(e, m)
            out["peak3_masked"].append((torch.where(m, e3, NEG).flatten(1).max(1).values / emed.clamp_min(1e-12)).cpu().numpy())
            out["peak3 (unmasked, earlier)"].append((e3.flatten(1).max(1).values / e.flatten(1).median(1).values.clamp_min(1e-12)).cpu().numpy())
            b3 = Fn.avg_pool2d(b, 3, 1, 1)
            bmed, bsd = masked_stats(b, m)
            out["input_peak3_masked (no model)"].append(((torch.where(m, b3, NEG).flatten(1).max(1).values - bmed) / bsd.clamp_min(1e-6)).cpu().numpy())
            kept.append(m[:, 0].float().mean((1, 2)).cpu().numpy())
            for j in range(len(b)):
                k = i + j
                if y[k] == 1:
                    bm = torch.from_numpy(box_mask((ROOT / meta[k]["rel_path"]).with_suffix(".json"), 0)).to(dev)
                    mk = m[j, 0]
                    out.setdefault("_box", []).append((float(mk[bm].float().mean()), bool(mk[bm].any())))
    box = out.pop("_box")
    s = {k: np.concatenate(v) for k, v in out.items()}
    kept = np.concatenate(kept)
    print(f"MASKED PEAK SCORES on VAL (plot 16): {int(y.sum())} faulty / {int((y == 0).sum())} healthy")
    print(f"  mask keeps {100*np.median(kept):.0f}% of a frame (median)")
    print(f"  fault box pixels kept by the mask: {100*np.mean([b[0] for b in box]):.0f}% on average; "
          f"frames with at least part of a box kept: {sum(b[1] for b in box)}/{len(box)}")
    for k, v in s.items():
        a = auroc(v, y); lo, hi = boot_ci(v, y)
        print(f"  {k:32s} AUROC {a:.3f}  (95% CI {lo:.3f} to {hi:.3f})")
    p, q = s["peak3_masked"], s["input_peak3_masked (no model)"]
    rng = np.random.default_rng(0); i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; d = []
    for _ in range(2000):
        idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        d.append(auroc(p[idx], y[idx]) - auroc(q[idx], y[idx]))
    print(f"  model minus no-model (masked): {auroc(p, y) - auroc(q, y):+.3f}  (95% CI {np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f})")
    np.save(ROOT / "results" / "india" / f"val_peak3_masked_v{MASK_VERSION}.npy", p)


if __name__ == "__main__":
    main()
