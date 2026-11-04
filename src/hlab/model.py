"""
The convolutional autoencoder every bundle in this repo trains and scores with.

Moved from src/train_autoencoder.py (HER-126); that script re-exports it so
the existing importers keep working. Architecture unchanged.
"""
import torch.nn as nn


class ThermalAutoencoder(nn.Module):
    """Symmetric encoder-decoder with 4 downsampling/upsampling stages.

    Input: (1, 240, 320) grayscale image, z-scored.
    Encoder: 4 conv blocks (32->64->128->256 filters), each with MaxPool2d
    Bottleneck: (256, 15, 20)
    Decoder: 4 conv blocks with Upsample, final 1-filter output

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
