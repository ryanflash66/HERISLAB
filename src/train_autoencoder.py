"""
Train a Convolutional Autoencoder for thermal fault detection (Track A).

Loads preprocessed images from CA_Preprocessed/, trains the autoencoder
on normal images only, and saves the trained model.

Architecture: symmetric encoder-decoder with Conv2D layers.
  Input: (1, 240, 320) grayscale image
  Encoder: 4 conv blocks (32->64->128->256 filters), each with MaxPool2d
  Bottleneck: (256, 15, 20)
  Decoder: 4 conv blocks with Upsample, final 1-filter output
  Loss: MSE (mean squared error between input and reconstruction)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# --- Config ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "CA_Preprocessed"
MODEL_DIR = ROOT / "models"
BATCH_SIZE = 32
EPOCHS = 50
VAL_SPLIT = 0.1
LEARNING_RATE = 1e-3
PATIENCE = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ThermalAutoencoder(nn.Module):
    """Symmetric encoder-decoder with 4 downsampling/upsampling stages.

    240 and 320 are both divisible by 16 (2^4), so spatial dimensions
    halve cleanly at each MaxPool and double back at each Upsample:
      (240, 320) -> (120, 160) -> (60, 80) -> (30, 40) -> (15, 20)
    """

    def __init__(self):
        super().__init__()

        # --- Encoder ---
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # -> (120, 160)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # -> (60, 80)

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(2, 2)  # -> (30, 40)

        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        )
        self.pool4 = nn.MaxPool2d(2, 2)  # -> (15, 20) bottleneck

        # --- Decoder ---
        self.up4 = nn.Upsample(scale_factor=2)  # -> (30, 40)
        self.dec4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
        )

        self.up3 = nn.Upsample(scale_factor=2)  # -> (60, 80)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
        )

        self.up2 = nn.Upsample(scale_factor=2)  # -> (120, 160)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
        )

        self.up1 = nn.Upsample(scale_factor=2)  # -> (240, 320)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )

        # Output: single channel, no activation (z-scored data can be negative)
        self.output_conv = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        bottleneck = self.pool4(e4)

        # Decode
        d4 = self.dec4(self.up4(bottleneck))
        d3 = self.dec3(self.up3(d4))
        d2 = self.dec2(self.up2(d3))
        d1 = self.dec1(self.up1(d2))

        return self.output_conv(d1)


def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load training data
    print("\nLoading training data...")
    train_data = np.load(DATA_DIR / "train_normal.npy")  # (N, 240, 320)
    print(f"  Shape: {train_data.shape}, dtype: {train_data.dtype}")

    # Add channel dimension: (N, 240, 320) -> (N, 1, 240, 320)
    train_data = train_data[:, np.newaxis, :, :]

    # Shuffle and split into train/val
    n_total = train_data.shape[0]
    n_val = int(n_total * VAL_SPLIT)
    rng = np.random.default_rng(42)
    indices = rng.permutation(n_total)

    x_train = torch.from_numpy(train_data[indices[n_val:]]).float()
    x_val = torch.from_numpy(train_data[indices[:n_val]]).float()
    del train_data

    print(f"  Train: {x_train.shape[0]} images")
    print(f"  Val:   {x_val.shape[0]} images")

    train_loader = DataLoader(
        TensorDataset(x_train, x_train),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, x_val),
        batch_size=BATCH_SIZE, shuffle=False, pin_memory=True,
    )

    # Build model
    print("\nBuilding autoencoder...")
    model = ThermalAutoencoder().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, min_lr=1e-6,
    )
    criterion = nn.MSELoss()

    # Training loop
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    print("\nStarting training...")
    for epoch in range(EPOCHS):
        # --- Train ---
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

        # --- Validate ---
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

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_DIR / "autoencoder_best.pth")
            print(f"  >> Saved best model (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                break

    # Load best weights and save final
    model.load_state_dict(torch.load(MODEL_DIR / "autoencoder_best.pth", weights_only=True))
    torch.save(model.state_dict(), MODEL_DIR / "autoencoder_final.pth")

    print(f"\nTraining complete.")
    print(f"  Best val_loss: {best_val_loss:.6f}")
    print(f"  Models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
