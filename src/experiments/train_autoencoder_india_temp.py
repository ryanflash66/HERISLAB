"""
Temperature-scale retrain (HER-82, Oct 26): same model, same recipe as
train_autoencoder_india.py, but the input is degrees C on ONE fixed scale
(data/CA_Preprocessed/india_temp), z-scored with the train mean/std. No
per-frame stretching. Weights go to models/autoencoder_india_temp_{best,final}.pth.
The same recipe is the "india_temp" entry of config/training.json.

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
from hlab.panels import resolve_device  # noqa: E402
from hlab.paths import MODELS  # noqa: E402

SEED, BATCH_SIZE, EPOCHS, PATIENCE = 42, 32, 50, 10
LEARNING_RATE, GRAD_CLIP, HOLDOUT_FRAC = 3e-4, 1.0, 0.10


def main(device=None):
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = resolve_device(device)
    s = Dataset("india_temp").split("train")           # z-scored with the train mean/std
    x, meta = s.x, s.meta
    assert len(meta) == len(x) and all(m["label"] == "healthy" for m in meta)
    tr, ho, flights = flight_holdout(meta, HOLDOUT_FRAC, SEED)
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
            torch.save(model.state_dict(), MODELS / "autoencoder_india_temp_best.pth")
        else:
            bad += 1
        print(f"Epoch {ep:3d}/{EPOCHS} train {tl:.6f}  holdout {vl:.6f}  lr {lr:.1e}  {time.time()-t0:.0f}s{mark}", flush=True)
        if bad >= PATIENCE:
            print(f"Early stop at epoch {ep} (no improvement for {PATIENCE})", flush=True)
            break
    torch.save(model.state_dict(), MODELS / "autoencoder_india_temp_final.pth")
    print(f"DONE best holdout {best:.6f} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default=None)
    main(ap.parse_args().device)
