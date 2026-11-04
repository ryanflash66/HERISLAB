"""
Predict-the-hidden-patch model (inpainting anomaly detection), Oct 26.

Same network as before (ThermalAutoencoder), same healthy-only training data
(plot 17, real degrees C, ground blanked with the panel median from the stage-1
panel mask). Only the TASK changes: patches of the frame are hidden and the model
must predict them from their surroundings. It never sees the pixels it is scored
on, so a hot spot cannot be copied. Declared before running:

  patches     8x8 pixel blocks on the 240x320 frame (a fault spot is ~3x3 px, so a
              block fully hides it); hidden blocks are set to 0 (the train mean)
  training    each step hides a random 25% of blocks; loss = MSE on hidden pixels
              that are on panels; 50 epochs, same optimiser and early-stop as before
  inference   4 fixed disjoint block patterns, (row + col) % 4, so every block is
              hidden exactly once and never together with an edge neighbour; the
              4 predictions are stitched into one "predicted from surroundings"
              frame; error map = (prediction - input)^2
  scores      peak3 and mean inside panels (as evaluate_india_panel.py), beside the
              same no-model rule; VAL only unless --test

  python src/inpaint_india.py train
  python src/inpaint_india.py eval [--test]
"""
import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_india import at_threshold, auroc, best_f1_threshold, boot_ci  # noqa: E402
from evaluate_india_panel import OLD, masked_peak  # noqa: E402
from train_autoencoder import ThermalAutoencoder  # noqa: E402
from train_autoencoder_india import flight_holdout  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "CA_Preprocessed" / "india_temp"
RES = ROOT / "results" / "india_temp"
MODELS = ROOT / "models"
WEIGHTS = MODELS / "inpaint_india_best.pth"
SEED, BATCH, EPOCHS, PATIENCE, LR, CLIP = 42, 32, 50, 10, 3e-4, 1.0
B, P_HIDE, H, W = 8, 0.25, 240, 320
GH, GW = H // B, W // B


def load(split, ns):
    x = np.load(DATA / f"{split}.npy")
    pm = np.load(DATA / f"{split}_panelmask.npy")
    meta = list(csv.DictReader(open(DATA / f"{split}_meta.csv", encoding="utf-8")))
    y = np.array([1 if m["label"] == "faulty" else 0 for m in meta])
    xz = ((x - ns["mean"]) / ns["std"]).astype(np.float32)
    for i in range(len(xz)):
        if pm[i].any():
            xz[i][~pm[i]] = np.median(xz[i][pm[i]])
    return x, xz, pm, meta, y


def block_mask(grid):
    """grid: (N,GH,GW) bool -> (N,1,H,W) float, 1 = hidden."""
    return grid.float().view(-1, 1, GH, 1, GW, 1).expand(-1, 1, GH, B, GW, B).reshape(-1, 1, H, W)


def train(dev):
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    ns = np.load(DATA / "norm_stats.npy", allow_pickle=True).item()
    _, x, pm, meta, _ = load("train", ns)
    tr, ho, flights = flight_holdout(meta)
    print(f"device {dev} | train {len(tr)} | hold-out {len(ho)} from flights {flights} | hide {B}x{B} blocks, {P_HIDE:.0%} per step", flush=True)
    t = torch.from_numpy(x).unsqueeze(1); pmt = torch.from_numpy(pm).unsqueeze(1)
    dl_tr = DataLoader(TensorDataset(t[tr], pmt[tr]), batch_size=BATCH, shuffle=True, pin_memory=True)
    dl_ho = DataLoader(TensorDataset(t[ho], pmt[ho]), batch_size=BATCH, shuffle=False, pin_memory=True)
    model = ThermalAutoencoder().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-6)
    g = torch.Generator(device=dev).manual_seed(SEED)

    def step(b, m, train_mode):
        b = b.to(dev, non_blocking=True); m = m.to(dev, non_blocking=True).float()
        hid = block_mask((torch.rand(len(b), GH, GW, generator=g, device=dev) < P_HIDE))
        out = model(b * (1 - hid))
        w = hid * m
        return (((out - b) ** 2) * w).sum() / w.sum().clamp_min(1.0)

    best, bad, t0 = float("inf"), 0, time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); tl = 0.0
        for b, m in dl_tr:
            opt.zero_grad(); loss = step(b, m, True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP); opt.step(); tl += loss.item() * len(b)
        model.eval(); vl = 0.0
        with torch.no_grad():
            for b, m in dl_ho:
                vl += step(b, m, False).item() * len(b)
        tl /= len(tr); vl /= len(ho); sched.step(vl)
        mark = ""
        if vl < best:
            best, bad, mark = vl, 0, "  *best"; torch.save(model.state_dict(), WEIGHTS)
        else:
            bad += 1
        print(f"Epoch {ep:3d}/{EPOCHS} train {tl:.6f}  holdout {vl:.6f}  lr {opt.param_groups[0]['lr']:.1e}  {time.time()-t0:.0f}s{mark}", flush=True)
        if bad >= PATIENCE:
            print(f"Early stop at epoch {ep}", flush=True); break
    torch.save(model.state_dict(), MODELS / "inpaint_india_final.pth")
    print(f"DONE best holdout {best:.6f} in {time.time()-t0:.0f}s", flush=True)


@torch.no_grad()
def predict(model, b, dev):
    """Stitch 4 disjoint-pattern predictions into one frame predicted from surroundings."""
    bi, bj = torch.meshgrid(torch.arange(GH), torch.arange(GW), indexing="ij")
    pat = ((bi + bj) % 4).to(dev)
    out = torch.zeros_like(b)
    for k in range(4):
        hid = block_mask((pat == k)[None].expand(len(b), -1, -1))
        out += model(b * (1 - hid)) * hid
    return out


@torch.no_grad()
def scores(model, xc, xz, pm, dev):
    out = {"inpaint_peak3": [], "inpaint_mean": [], "input_peak3": []}
    for i in range(0, len(xz), 64):
        b = torch.from_numpy(xz[i:i + 64]).unsqueeze(1).to(dev)
        m = torch.from_numpy(pm[i:i + 64]).unsqueeze(1).to(dev)
        e = (predict(model, b, dev) - b) ** 2
        mx, md, _ = masked_peak(e, m)
        out["inpaint_peak3"].append((mx / md.clamp_min(1e-12)).cpu().numpy())
        out["inpaint_mean"].append(torch.stack([ei[mi].mean() if mi.any() else ei.mean() for ei, mi in zip(e, m)]).cpu().numpy())
        t = torch.from_numpy(xc[i:i + 64]).unsqueeze(1).to(dev)
        mx, md, sd = masked_peak(t, m)
        out["input_peak3"].append(((mx - md) / sd.clamp_min(1e-6)).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def report(split, s, y, thr=None):
    print(f"\n======== {split.upper()} ({int(y.sum())} faulty / {int((y == 0).sum())} healthy)")
    for k, v in s.items():
        a = auroc(v, y); lo, hi = boot_ci(v, y)
        line = f"  {k:14s} AUROC {a:.3f} (95% CI {lo:.3f} to {hi:.3f})"
        if thr is not None:
            r = at_threshold(v, y, thr[k]); line += f" | precision {r['precision']:.3f} recall {r['recall']:.3f} F1 {r['f1']:.3f}"
        print(line + ("  <- primary" if k == "inpaint_peak3" else ""))
    p, q = s["inpaint_peak3"], s["input_peak3"]; rng = np.random.default_rng(0)
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; d = []
    for _ in range(2000):
        idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]); d.append(auroc(p[idx], y[idx]) - auroc(q[idx], y[idx]))
    print(f"  model minus no-model rule: {auroc(p, y) - auroc(q, y):+.3f} (95% CI {np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f})")
    print(f"  earlier models on this split (peak, panels only): old 0.637 / temp 0.601 / panels-only 0.543; rule 0.639" if split == "val" else "")


def evaluate(dev, test):
    ns = np.load(DATA / "norm_stats.npy", allow_pickle=True).item()
    model = ThermalAutoencoder().to(dev); model.load_state_dict(torch.load(WEIGHTS, map_location=dev)); model.eval()
    xc, xz, pm, meta, y = load("val", ns); sv = scores(model, xc, xz, pm, dev)
    report("val", sv, y); np.save(RES / "val_inpaint_scores.npy", sv, allow_pickle=True)
    if not test:
        return
    thr = {k: best_f1_threshold(v, y) for k, v in sv.items()}
    for split in ("test", "cross_site"):
        xc, xz, pm, meta, y = load(split, ns); s = scores(model, xc, xz, pm, dev)
        report(split, s, y, thr); np.save(RES / f"{split}_inpaint_scores.npy", s, allow_pickle=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["train", "eval"]); ap.add_argument("--test", action="store_true")
    a = ap.parse_args(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(dev) if a.mode == "train" else evaluate(dev, a.test)
