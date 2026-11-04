"""
Train the autoencoder on the Thermal-Library India train plot (milestone 06).

Changes vs train_autoencoder_pv.py:
  - data: data/CA_Preprocessed/india/train.npy (plot 17, healthy only; HER-18/82)
  - early-stopping hold-out is whole FLIGHTS of plot 17 (~10% of frames), not
    random frames, so it has no near-copies in training (HER-18)
  - HER-106 fix for the epoch-6 loss spike: lr 3e-4 (was 1e-3) + gradient
    clipping at norm 1.0
Architecture unchanged (ThermalAutoencoder from train_autoencoder.py).

Outputs: models/autoencoder_india_best.pth, _final.pth
         results/india/training_log.txt (via stdout redirect)
"""
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_autoencoder import ThermalAutoencoder  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "CA_Preprocessed" / "india"
MODELS = ROOT / "models"
SEED, BATCH_SIZE, EPOCHS, PATIENCE = 42, 32, 50, 10
LEARNING_RATE, GRAD_CLIP, HOLDOUT_FRAC = 3e-4, 1.0, 0.10


def flight_holdout(meta):
    """Pick whole flights until ~HOLDOUT_FRAC of frames are held out."""
    by_flight = defaultdict(list)
    for i, m in enumerate(meta):
        by_flight[m["flight"]].append(i)
    flights = sorted(by_flight)
    random.Random(SEED).shuffle(flights)
    target, held, chosen = HOLDOUT_FRAC * len(meta), [], []
    for f in flights:
        if len(held) >= target:
            break
        held += by_flight[f]; chosen.append(f)
    held_set = set(held)
    train_idx = [i for i in range(len(meta)) if i not in held_set]
    return np.array(train_idx), np.array(sorted(held)), chosen


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = np.load(DATA / "train.npy")
    meta = list(csv.DictReader(open(DATA / "train_meta.csv", encoding="utf-8")))
    assert len(meta) == len(x) and all(m["label"] == "healthy" for m in meta)
    tr, ho, flights = flight_holdout(meta)
    print(f"device {dev} | train frames {len(tr)} | early-stop hold-out {len(ho)} frames from flights {flights}", flush=True)

    t = torch.from_numpy(x).unsqueeze(1)
    dl_tr = DataLoader(TensorDataset(t[tr]), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    dl_ho = DataLoader(TensorDataset(t[ho]), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    model = ThermalAutoencoder().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-6)
    loss_fn = nn.MSELoss()
    MODELS.mkdir(exist_ok=True)
    best, bad, t0 = float("inf"), 0, time.time()

    for ep in range(1, EPOCHS + 1):
        model.train(); tl = 0.0
        for (b,) in dl_tr:
            b = b.to(dev, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(b), b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tl += loss.item() * len(b)
        tl /= len(tr)
        model.eval(); vl = 0.0
        with torch.no_grad():
            for (b,) in dl_ho:
                b = b.to(dev, non_blocking=True)
                vl += loss_fn(model(b), b).item() * len(b)
        vl /= len(ho)
        sched.step(vl)
        lr = opt.param_groups[0]["lr"]
        mark = ""
        if vl < best:
            best, bad, mark = vl, 0, "  *best"
            torch.save(model.state_dict(), MODELS / "autoencoder_india_best.pth")
        else:
            bad += 1
        print(f"Epoch {ep:3d}/{EPOCHS} train {tl:.6f}  holdout {vl:.6f}  lr {lr:.1e}  {time.time()-t0:.0f}s{mark}", flush=True)
        if bad >= PATIENCE:
            print(f"Early stop at epoch {ep} (no improvement for {PATIENCE})", flush=True)
            break
    torch.save(model.state_dict(), MODELS / "autoencoder_india_final.pth")
    print(f"DONE best holdout {best:.6f} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
