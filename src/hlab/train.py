"""
Training (HER-128): one loop, configurations in config/training.json.

    bundle, history = fit(Dataset("pv"), "pv", holdout=RandomFrames(seed=43))
    bundle, history = fit_config("pv")            # the config/training.json entry

fit() is the loop every tracked training script used to carry: Adam,
ReduceLROnPlateau(0.5, patience 5, min_lr 1e-6), early stopping on the
holdout loss, best checkpoint saved as it improves. What varied between the
scripts is now an argument:

  holdout   RandomFrames(seed, fraction)  random frames of the split (v2 seed 42, PV seed 43)
            Flights(seed, fraction)       whole flights, from the split's "flight" metadata (India)
  loss      "mse" | "masked_mse"          masked = MSE over the panel pixels of dataset.masks(split)
  lr, clip  learning rate, gradient-norm clip (None = no clipping)
  seed      torch / numpy / random seed (None = unseeded, as the v2 and PV runs were)
  final     "best" copies the best checkpoint to *_final.pth; "last" saves the last epoch

Weights go to model_dir (models/ by default) under the bundle's file names when
out_name is a bundle name (v2 -> autoencoder_best.pth), otherwise
autoencoder_<out_name>_{best,final}.pth. The returned Bundle wraps the best
weights and the dataset's stats; its threshold is None until an evaluator runs.

Not reproduced: byte-identical weights. CUDA training is non-deterministic, so
retraining a configuration is checked by its holdout loss and the evaluator's
AUROC landing within noise of the recorded run, not by comparing checkpoints.
"""
import json
import random
import sys
import time
from dataclasses import dataclass

import numpy as np

from hlab.bundle import BUNDLES, Bundle, resolve
from hlab.datasets import Dataset, flight_holdout
from hlab.paths import CONFIG, MODELS, RESULTS

TRAINING_CONFIG = CONFIG / "training.json"
LOSSES = ("mse", "masked_mse")
FINALS = ("best", "last")


# ---------------------------------------------------------------- holdouts
@dataclass(frozen=True)
class RandomFrames:
    """Random frames of the split: a permutation seeded once, first `fraction` held out."""
    seed: int = 42
    fraction: float = 0.1

    def indices(self, split):
        n = len(split)
        n_val = int(n * self.fraction)
        idx = np.random.default_rng(self.seed).permutation(n)
        return idx[n_val:], idx[:n_val]


@dataclass(frozen=True)
class Flights:
    """Whole flights of the split (meta "flight") until `fraction` of its frames are held out."""
    seed: int = 42
    fraction: float = 0.1

    def indices(self, split):
        train_idx, held, _flights = flight_holdout(split.meta, fraction=self.fraction, seed=self.seed)
        return train_idx, held


HOLDOUTS = {"random_frames": RandomFrames, "flights": Flights}


def holdout_from_config(d):
    kind = d.get("type", "random_frames")
    if kind not in HOLDOUTS:
        raise ValueError(f"unknown holdout type {kind!r}; known: {', '.join(HOLDOUTS)}")
    return HOLDOUTS[kind](**{k: v for k, v in d.items() if k != "type"})


# ---------------------------------------------------------------- helpers
class Tee:
    """Write to several streams (console + training log)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, msg):
        for s in self.streams:
            s.write(msg)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def weights_paths(out_name, model_dir):
    """(best, final) checkpoint paths; bundle names keep their established file names."""
    best = resolve(out_name)["weights"].name if out_name in BUNDLES else f"autoencoder_{out_name}_best.pth"
    return model_dir / best, model_dir / best.replace("_best.pth", "_final.pth")


def _masked_mse(out, target, mask):
    m = mask.float()
    return (((out - target) ** 2) * m).sum() / m.sum().clamp_min(1.0)


# ---------------------------------------------------------------- the loop
def fit(dataset, out_name, split="train", holdout=None, loss="mse", lr=1e-3, clip=None, seed=None,
        epochs=50, batch=32, patience=10, final="best", device=None, model_dir=None, log=None):
    """Train the autoencoder on `dataset.split(split)`; returns (Bundle of the best weights, history).

    history: train_loss and val_loss per epoch, best_epoch, best_val_loss, epochs_run,
    final_epoch (which epoch's weights *_final.pth holds), best_path, final_path, minutes.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from hlab.model import ThermalAutoencoder

    if loss not in LOSSES:
        raise ValueError(f"unknown loss {loss!r}; known: {', '.join(LOSSES)}")
    if final not in FINALS:
        raise ValueError(f"unknown final {final!r}; known: {', '.join(FINALS)}")
    holdout = holdout or RandomFrames()
    model_dir = model_dir or MODELS
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path, final_path = weights_paths(out_name, model_dir)
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_handle = None
    original_stdout = sys.stdout
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log, "w", encoding="utf-8")
        sys.stdout = Tee(original_stdout, log_handle)
    try:
        start = time.time()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        print(f"Device: {dev}")
        if dev.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        print(f"\nLoading {dataset.name} / {split}...")
        s = dataset.split(split)
        print(f"  Shape: {s.x.shape}, dtype: {s.x.dtype}")
        tr_idx, va_idx = holdout.indices(s)
        x = torch.from_numpy(np.ascontiguousarray(s.x[:, np.newaxis])).float()
        parts_tr, parts_va = [x[tr_idx]], [x[va_idx]]
        if loss == "masked_mse":
            m = torch.from_numpy(np.ascontiguousarray(dataset.masks(split)[:, np.newaxis]))
            parts_tr.append(m[tr_idx])
            parts_va.append(m[va_idx])
        del x
        print(f"  Train: {len(tr_idx)} images")
        print(f"  Val:   {len(va_idx)} images  ({holdout})")

        pin = dev.type == "cuda"
        train_loader = DataLoader(TensorDataset(*parts_tr), batch_size=batch, shuffle=True, pin_memory=pin)
        val_loader = DataLoader(TensorDataset(*parts_va), batch_size=batch, shuffle=False, pin_memory=pin)

        print("\nBuilding autoencoder...")
        model = ThermalAutoencoder().to(dev)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)
        mse = nn.MSELoss()

        def batch_loss(tensors):
            tensors = [t.to(dev, non_blocking=True) for t in tensors]
            out = model(tensors[0])
            if loss == "masked_mse":
                return _masked_mse(out, tensors[0], tensors[1]), len(tensors[0])
            return mse(out, tensors[0]), len(tensors[0])

        history = {"train_loss": [], "val_loss": []}
        best_val, best_epoch, bad = float("inf"), 0, 0
        print(f"\nStarting training -- batch {batch}, max epochs {epochs}, patience {patience}, "
              f"lr {lr:g}, clip {clip}, loss {loss}, seed {seed}")
        epochs_run = 0
        for epoch in range(1, epochs + 1):
            epochs_run = epoch
            model.train()
            total, count = 0.0, 0
            for tensors in train_loader:
                l, n = batch_loss(tensors)
                optimizer.zero_grad()
                l.backward()
                if clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                total += l.item() * n
                count += n
            train_loss = total / max(count, 1)

            model.eval()
            total, count = 0.0, 0
            with torch.no_grad():
                for tensors in val_loader:
                    l, n = batch_loss(tensors)
                    total += l.item() * n
                    count += n
            val_loss = total / max(count, 1)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d}/{epochs} -- train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}  lr: {current_lr:.1e}")
            scheduler.step(val_loss)

            if val_loss < best_val:
                best_val, best_epoch, bad = val_loss, epoch, 0
                torch.save(model.state_dict(), best_path)
                print(f"  >> Saved best model (val_loss: {val_loss:.6f})")
            else:
                bad += 1
                if bad >= patience:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    break

        if final == "best":
            model.load_state_dict(torch.load(best_path, map_location=dev, weights_only=True))
            final_epoch = best_epoch
        else:
            final_epoch = epochs_run
        torch.save(model.state_dict(), final_path)

        minutes = (time.time() - start) / 60
        print(f"\nTraining complete in {minutes:.1f} min")
        print(f"  Best val_loss: {best_val:.6f} at epoch {best_epoch}")
        print(f"  Best weights:  {best_path}")
        print(f"  Final weights: {final_path} (epoch {final_epoch})")
        if log is not None:
            print(f"  Training log:  {log}")
        history.update(best_epoch=best_epoch, best_val_loss=best_val, epochs_run=epochs_run,
                       final_epoch=final_epoch, best_path=best_path, final_path=final_path, minutes=minutes)
    finally:
        if log_handle is not None:
            sys.stdout = original_stdout
            log_handle.close()

    return Bundle(out_name, best_path, dataset.stats, None, device=dev), history


# ---------------------------------------------------------------- configurations
def load_training_config(name, path=None):
    """One entry of config/training.json, by name."""
    path = path or TRAINING_CONFIG
    entries = json.loads(path.read_text(encoding="utf-8"))
    if name not in entries:
        raise ValueError(f"no training config {name!r} in {path}; known: {', '.join(sorted(entries))}")
    return entries[name]


def fit_config(name, path=None, device=None, model_dir=None, log_dir=None, **overrides):
    """fit() with the arguments of a config/training.json entry; overrides win (epochs=2, ...)."""
    cfg = dict(load_training_config(name, path))
    cfg.update(overrides)
    log = (log_dir or RESULTS) / cfg["log"] if cfg.get("log") else None
    return fit(Dataset(cfg["dataset"]), cfg.get("out_name", name), split=cfg.get("split", "train"),
               holdout=holdout_from_config(cfg.get("holdout", {})), loss=cfg.get("loss", "mse"),
               lr=cfg.get("lr", 1e-3), clip=cfg.get("clip"), seed=cfg.get("seed"), epochs=cfg.get("epochs", 50),
               batch=cfg.get("batch", 32), patience=cfg.get("patience", 10), final=cfg.get("final", "best"),
               device=device, model_dir=model_dir, log=log)
