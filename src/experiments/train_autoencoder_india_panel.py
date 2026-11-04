"""
PANELS-ONLY retrain (stage 2 of the pipeline, Oct 26): same model and recipe as
train_autoencoder_india_temp.py, input = degrees C on one fixed scale, with
ground pixels (from the stage-1 panel mask, {split}_panelmask.npy) replaced by
the frame's panel median (hlab.panels.blank_ground), and the loss computed on
panel pixels only. Weights: models/autoencoder_india_panel_{best,final}.pth.

Recipe: whole-flight early-stop hold-out (hlab.datasets.flight_holdout, seed 42),
lr 3e-4, gradient clipping at norm 1.0, batch 32, patience 10, 50 epochs.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hlab.datasets import Dataset, flight_holdout  # noqa: E402
from hlab.model import ThermalAutoencoder  # noqa: E402
from hlab.panels import blank_ground, resolve_device  # noqa: E402
from hlab.paths import MODELS  # noqa: E402

SEED, BATCH_SIZE, EPOCHS, PATIENCE = 42, 32, 50, 10
LEARNING_RATE, GRAD_CLIP, HOLDOUT_FRAC = 3e-4, 1.0, 0.10


def main(device=None):
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = resolve_device(device)
    ds = Dataset("india_temp")
    s = ds.split("train")
    pm = ds.masks("train")
    assert pm.shape == s.x.shape
    x = blank_ground(s.x, pm)                           # ground = that frame's panel median
    meta = s.meta
    assert len(meta) == len(x) and all(m["label"] == "healthy" for m in meta)
    tr, ho, flights = flight_holdout(meta, HOLDOUT_FRAC, SEED)
    print(f"device {dev} | train frames {len(tr)} | early-stop hold-out {len(ho)} frames from flights {flights}", flush=True)

    t = torch.from_numpy(x).unsqueeze(1); pmt = torch.from_numpy(pm).unsqueeze(1)
    dl_tr = DataLoader(TensorDataset(t[tr], pmt[tr]), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    dl_ho = DataLoader(TensorDataset(t[ho], pmt[ho]), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    def masked_mse(out, b, m):
        m = m.float(); return (((out - b) ** 2) * m).sum() / m.sum().clamp_min(1.0)

    model = ThermalAutoencoder().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-6)
    MODELS.mkdir(exist_ok=True)
    best, bad, t0 = float("inf"), 0, time.time()

    for ep in range(1, EPOCHS + 1):
        model.train(); tl = 0.0
        for b, m in dl_tr:
            b = b.to(dev, non_blocking=True); m = m.to(dev, non_blocking=True)
            opt.zero_grad()
            loss = masked_mse(model(b), b, m)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tl += loss.item() * len(b)
        tl /= len(tr)
        model.eval(); vl = 0.0
        with torch.no_grad():
            for b, m in dl_ho:
                b = b.to(dev, non_blocking=True); m = m.to(dev, non_blocking=True)
                vl += masked_mse(model(b), b, m).item() * len(b)
        vl /= len(ho)
        sched.step(vl)
        lr = opt.param_groups[0]["lr"]
        mark = ""
        if vl < best:
            best, bad, mark = vl, 0, "  *best"
            torch.save(model.state_dict(), MODELS / "autoencoder_india_panel_best.pth")
        else:
            bad += 1
        print(f"Epoch {ep:3d}/{EPOCHS} train {tl:.6f}  holdout {vl:.6f}  lr {lr:.1e}  {time.time()-t0:.0f}s{mark}", flush=True)
        if bad >= PATIENCE:
            print(f"Early stop at epoch {ep} (no improvement for {PATIENCE})", flush=True)
            break
    torch.save(model.state_dict(), MODELS / "autoencoder_india_panel_final.pth")
    print(f"DONE best holdout {best:.6f} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default=None)
    main(ap.parse_args().device)
