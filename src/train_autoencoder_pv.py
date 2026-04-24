"""
Train the PV-specialist autoencoder (HER-89).

Same architecture as the v2 model (`ThermalAutoencoder` imported from
`train_autoencoder`), same hyperparameters, different data source:

  Input:  data/CA_Preprocessed/pv/train_normal.npy (PV-only, 8019 images)
  Output: models/autoencoder_pv_best.pth
          models/autoencoder_pv_final.pth
          results/training_log_pv.txt

v2's multi-equipment model lives at models/autoencoder_best.pth and is
left untouched.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train_autoencoder import ThermalAutoencoder  # reuse v2's architecture

# --- Config ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "CA_Preprocessed" / "pv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
BATCH_SIZE = 32
EPOCHS = 50
VAL_SPLIT = 0.1
LEARNING_RATE = 1e-3
PATIENCE = 10

BEST_PATH = MODEL_DIR / "autoencoder_pv_best.pth"
FINAL_PATH = MODEL_DIR / "autoencoder_pv_final.pth"
LOG_PATH = RESULTS_DIR / "training_log_pv.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tee:
    """Simple tee: write to stdout + a file handle."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, msg):
        for s in self.streams:
            s.write(msg)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Tee console + training log
    log_handle = LOG_PATH.open("w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_handle)

    try:
        start = time.time()
        print(f"Device: {DEVICE}")
        if DEVICE.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # --- Load PV training data ---
        train_path = DATA_DIR / "train_normal.npy"
        if not train_path.exists():
            print(f"ERROR: {train_path} not found. Run src/preprocess_pv.py first.")
            return
        print(f"\nLoading PV training data from {train_path.name}...")
        train_data = np.load(train_path)
        print(f"  Shape: {train_data.shape}, dtype: {train_data.dtype}")

        # Channel dim: (N, 240, 320) -> (N, 1, 240, 320)
        train_data = train_data[:, np.newaxis, :, :]

        # Shuffle + split into train/val using a separate seed from the preprocess split
        # so the training-internal val set is independent of the holdout test set.
        n_total = train_data.shape[0]
        n_val = int(n_total * VAL_SPLIT)
        rng = np.random.default_rng(43)  # distinct from preprocess seed (42)
        indices = rng.permutation(n_total)

        x_train = torch.from_numpy(train_data[indices[n_val:]]).float()
        x_val = torch.from_numpy(train_data[indices[:n_val]]).float()
        del train_data

        print(f"  Train: {x_train.shape[0]} images")
        print(f"  Val:   {x_val.shape[0]} images  (internal val, separate from holdout test)")

        train_loader = DataLoader(
            TensorDataset(x_train, x_train),
            batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
        )
        val_loader = DataLoader(
            TensorDataset(x_val, x_val),
            batch_size=BATCH_SIZE, shuffle=False, pin_memory=True,
        )

        # --- Build model ---
        print("\nBuilding PV-specialist autoencoder (same arch as v2)...")
        model = ThermalAutoencoder().to(DEVICE)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5, min_lr=1e-6,
        )
        criterion = nn.MSELoss()

        # --- Training loop ---
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        print(f"\nStarting training -- batch {BATCH_SIZE}, max epochs {EPOCHS}, patience {PATIENCE}")
        for epoch in range(EPOCHS):
            # Train
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                output = model(batch_x)
                loss = criterion(output, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
            train_loss /= len(train_loader.dataset)

            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(DEVICE)
                    batch_y = batch_y.to(DEVICE)
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
            val_loss /= len(val_loader.dataset)

            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:3d}/{EPOCHS} -- train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}  lr: {lr:.1e}")

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(model.state_dict(), BEST_PATH)
                print(f"  >> Saved best PV model (val_loss: {val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                    break

        # Load best weights and save as final
        model.load_state_dict(torch.load(BEST_PATH, weights_only=True))
        torch.save(model.state_dict(), FINAL_PATH)

        elapsed_min = (time.time() - start) / 60
        print(f"\nTraining complete in {elapsed_min:.1f} min")
        print(f"  Best val_loss: {best_val_loss:.6f} at epoch {best_epoch}")
        print(f"  Best weights: {BEST_PATH}")
        print(f"  Final weights: {FINAL_PATH}")
        print(f"  Training log:  {LOG_PATH}")
    finally:
        sys.stdout = original_stdout
        log_handle.close()


if __name__ == "__main__":
    main()
