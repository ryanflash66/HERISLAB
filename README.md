# Thermal Anomaly Detection in Electrical Systems

A convolutional autoencoder trained exclusively on normal thermal images to detect faults in electrical equipment through reconstruction error analysis.

Developed at **HERIS Lab** -- Holistic Engineering Research for Intelligent Systems.

## Overview

Thermal imaging is widely used for non-destructive inspection of electrical equipment, but manual review is time-consuming and subjective. This project automates fault detection using an unsupervised deep learning approach: a convolutional autoencoder learns the distribution of "normal" thermal patterns, then flags anomalies by measuring how poorly it reconstructs a new image.

**Key idea:** The model only ever sees healthy equipment during training. At inference time, a faulty image produces high reconstruction error because the model has never learned to reproduce fault patterns.

## Results

| Metric | Value |
|--------|-------|
| AUROC | 0.909 |
| F1 Score | 0.975 |
| Precision | 0.952 |
| Recall | 1.000 |

Evaluated on 33 normal and 452 fault test images from electric motor and induction motor thermal datasets.

## Architecture

```
Input (1x240x320) grayscale thermal image
    |
Encoder: 4 conv blocks (32 -> 64 -> 128 -> 256 filters)
    |         each block: Conv3x3 + ReLU + Conv3x3 + ReLU + MaxPool2x2
    |
Bottleneck: 256 x 15 x 20
    |
Decoder: 4 conv blocks (256 -> 128 -> 64 -> 32 filters)
    |         each block: Upsample2x + Conv3x3 + ReLU + Conv3x3 + ReLU
    |
Output (1x240x320) reconstructed image
```

- **Parameters:** 2,352,353
- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam (lr=1e-3) with ReduceLROnPlateau scheduling
- **Early stopping:** Patience of 10 epochs

## Project Structure

```
HERISLAB/
├── src/
│   ├── preprocess.py             # Image loading, normalization, train/test split
│   ├── train_autoencoder.py      # Model definition and training loop
│   └── evaluate_autoencoder.py   # Threshold calibration, AUROC, F1 metrics
├── data/
│   ├── CA_Training_Data/         # Raw thermal images (hosted on HuggingFace)
│   └── CA_Preprocessed/          # Z-score normalized .npy arrays
├── models/                       # Trained model weights (.pth)
├── results/                      # Reconstruction error arrays and metrics
├── docs/                         # Training plan, architecture docs, code walkthrough
├── .gitignore
├── LICENSE
└── README.md
```

## Dataset

Training data is sourced from multiple public thermal imaging datasets:

| Source | Split | Count | Format |
|--------|-------|-------|--------|
| Electric Motor Thermal Fault Diagnosis | Train (normal) | 176 | PNG |
| Thermal Images of Induction Motor (Noload) | Train (normal) | 25 | BMP |
| Photovoltaic System O&M Inspection | Train (normal) | 7,836 | TIFF |
| Photovoltaic System Thermal Inspection | Train (normal) | 1,062 | TIFF |
| Electric Motor Thermal Fault Diagnosis | Test (normal/fault) | 20 / 173 | PNG |
| Thermal Images of Induction Motor | Test (normal/fault) | 13 / 279 | BMP |

**Preprocessing pipeline:**
1. Load as grayscale float32
2. Scale to common 0-255 range (16-bit radiometric TIFFs are min-max scaled; 8-bit images pass through)
3. Resize to 320x240 using Lanczos resampling
4. Z-score normalize using training set global statistics (mean=171.97, std=45.68)

Images and preprocessed arrays are hosted on HuggingFace and excluded from this repository via `.gitignore`.

## Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (tested on RTX 5070)

### Installation

```bash
git clone https://github.com/ryanflash66/HERISLAB.git
cd HERISLAB
python -m venv venv
venv\Scripts\activate        # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy pillow
```

### Usage

```bash
# 1. Preprocess raw images into normalized arrays
python src/preprocess.py

# 2. Train the autoencoder (normal images only)
python src/train_autoencoder.py

# 3. Evaluate on test set
python src/evaluate_autoencoder.py
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
