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
              frame (hlab.panels.hidden_block_predict); error map = (prediction - input)^2
  scores      peak3 and mean inside panels (as evaluate_india_panel.py), beside the
              same no-model rule; VAL only unless --test

  python src/experiments/inpaint_india.py train
  python src/experiments/inpaint_india.py eval [--test]
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hlab.bundle import Bundle  # noqa: E402
from hlab.datasets import Dataset, flight_holdout  # noqa: E402
from hlab.metrics import auroc, auroc_ci, best_f1_threshold, confusion, paired_delta_ci  # noqa: E402
from hlab.model import ThermalAutoencoder  # noqa: E402
from hlab.panels import blank_ground, block_mask, hidden_block_predict, peak_ratio, peak_standout, resolve_device  # noqa: E402
from hlab.paths import MODELS, RESULTS  # noqa: E402

RES = RESULTS / "india_temp"
WEIGHTS = MODELS / "inpaint_india_best.pth"
SEED, BATCH, EPOCHS, PATIENCE, LR, CLIP = 42, 32, 50, 10, 3e-4, 1.0
B, P_HIDE, H, W = 8, 0.25, 240, 320
GH, GW = H // B, W // B


def load(ds, split):
    """(degrees C, z-scored with the ground blanked, panel masks, meta, labels)."""
    s = ds.split(split)
    pm = ds.masks(split)
    return np.load(ds.array_path(split)), blank_ground(s.x, pm), pm, s.meta, s.labels


def train(dev):
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    ds = Dataset("india_temp")
    _, x, pm, meta, _ = load(ds, "train")
    tr, ho, flights = flight_holdout(meta)
    print(f"device {dev} | train {len(tr)} | hold-out {len(ho)} from flights {flights} | hide {B}x{B} blocks, {P_HIDE:.0%} per step", flush=True)
    t = torch.from_numpy(x).unsqueeze(1); pmt = torch.from_numpy(pm).unsqueeze(1)
    dl_tr = DataLoader(TensorDataset(t[tr], pmt[tr]), batch_size=BATCH, shuffle=True, pin_memory=True)
    dl_ho = DataLoader(TensorDataset(t[ho], pmt[ho]), batch_size=BATCH, shuffle=False, pin_memory=True)
    model = ThermalAutoencoder().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-6)
    g = torch.Generator(device=dev).manual_seed(SEED)

    def step(b, m):
        b = b.to(dev, non_blocking=True); m = m.to(dev, non_blocking=True).float()
        hid = block_mask((torch.rand(len(b), GH, GW, generator=g, device=dev) < P_HIDE), B)
        out = model(b * (1 - hid))
        w = hid * m
        return (((out - b) ** 2) * w).sum() / w.sum().clamp_min(1.0)

    best, bad, t0 = float("inf"), 0, time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); tl = 0.0
        for b, m in dl_tr:
            opt.zero_grad(); loss = step(b, m); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP); opt.step(); tl += loss.item() * len(b)
        model.eval(); vl = 0.0
        with torch.no_grad():
            for b, m in dl_ho:
                vl += step(b, m).item() * len(b)
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
def scores(model, xc, xz, pm, dev):
    out = {"inpaint_peak3": [], "inpaint_mean": [], "input_peak3": []}
    for i in range(0, len(xz), 64):
        b = torch.from_numpy(xz[i:i + 64]).unsqueeze(1).to(dev)
        m = torch.from_numpy(pm[i:i + 64]).unsqueeze(1).to(dev)
        e = (hidden_block_predict(model, b, B) - b) ** 2
        out["inpaint_peak3"].append(peak_ratio(e, m).cpu().numpy())
        out["inpaint_mean"].append(torch.stack([ei[mi].mean() if mi.any() else ei.mean() for ei, mi in zip(e, m)]).cpu().numpy())
        t = torch.from_numpy(xc[i:i + 64]).unsqueeze(1).to(dev)
        out["input_peak3"].append(peak_standout(t, m).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def report(split, s, y, thr=None):
    print(f"\n======== {split.upper()} ({int(y.sum())} faulty / {int((y == 0).sum())} healthy)")
    for k, v in s.items():
        a = auroc(v, y); lo, hi = auroc_ci(v, y, n_boot=1000, seed=0)
        line = f"  {k:14s} AUROC {a:.3f} (95% CI {lo:.3f} to {hi:.3f})"
        if thr is not None:
            r = confusion(v, y, thr[k]); line += f" | precision {r['precision']:.3f} recall {r['recall']:.3f} F1 {r['f1']:.3f}"
        print(line + ("  <- primary" if k == "inpaint_peak3" else ""))
    d, lo, hi = paired_delta_ci(s["input_peak3"], s["inpaint_peak3"], y, n_boot=2000, seed=0)
    print(f"  model minus no-model rule: {d:+.3f} (95% CI {lo:+.3f} to {hi:+.3f})")
    print(f"  earlier models on this split (peak, panels only): old 0.637 / temp 0.601 / panels-only 0.543; rule 0.639" if split == "val" else "")


def evaluate(dev, test):
    ds = Dataset("india_temp")
    model = Bundle.load("india_inpaint", dev).model
    xc, xz, pm, meta, y = load(ds, "val"); sv = scores(model, xc, xz, pm, dev)
    report("val", sv, y); np.save(RES / "val_inpaint_scores.npy", sv, allow_pickle=True)
    if not test:
        return
    thr = {k: best_f1_threshold(v, y) for k, v in sv.items()}
    for split in ("test", "cross_site"):
        xc, xz, pm, meta, y = load(ds, split); s = scores(model, xc, xz, pm, dev)
        report(split, s, y, thr); np.save(RES / f"{split}_inpaint_scores.npy", s, allow_pickle=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["train", "eval"]); ap.add_argument("--test", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(); dev = resolve_device(a.device)
    train(dev) if a.mode == "train" else evaluate(dev, a.test)
