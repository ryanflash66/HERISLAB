# Code Walkthrough -- HERISLAB Thermal Fault Detection

A simplified guide to all the code and data in this project so far.

---

## What are we building?

A system that looks at thermal camera images of electrical equipment (motors) and detects if something is wrong (a fault). We're using a **Convolutional Autoencoder** -- a type of neural network that learns what "normal" looks like, and flags anything that doesn't match.

The core idea: train only on normal images. When a fault image comes in, the model can't reconstruct it well because it's never seen a fault. That high reconstruction error = "something's wrong."

---

## Project structure

```
HERISLAB/
  src/                        <-- All Python code
    preprocess.py             <-- Preprocessing pipeline (raw images -> normalized .npy arrays)
    train_autoencoder.py      <-- Autoencoder model definition + training loop (PyTorch, GPU)
    evaluate_autoencoder.py   <-- Evaluation: reconstruction errors, threshold, AUROC, F1
  models/                     <-- Saved model weights (gitignored)
    autoencoder_best.pth      <-- Best model from training (epoch 37, val_loss 0.0075)
    autoencoder_final.pth     <-- Same weights, saved at end of training
  results/                    <-- Evaluation outputs (gitignored)
    normal_errors.npy         <-- Per-image reconstruction errors for test/normal
    fault_errors.npy          <-- Per-image reconstruction errors for test/fault
    eval_metrics.npy          <-- AUROC, threshold, precision, recall, F1
  data/
    CA_Training_Data/         <-- Raw curated images, organized by split
    train/normal/             <-- 9,099 normal images for training
      electric_motor/         <-- 168 PNG files
      induction_motor/        <-- 20 BMP files
      pv_om_inspection/       <-- 7,836 TIFF files (solar panels, adjacent domain)
      pv_thermal_inspection/  <-- 1,075 TIFF files (solar panels, adjacent domain)
      solar_modules/          <-- 2,302 JPG files (NOT USED -- too small at 24x40)
    test/normal/              <-- 33 normal images held back for testing
      electric_motor/         <-- 28 PNG
      induction_motor/        <-- 5 BMP
    test/fault/               <-- Fault images to test the model
      electric_motor/         <-- 173 PNG
      induction_motor/        <-- 279 BMP (65 ambiguous ones moved out)
      induction_motor_ambiguous/ <-- 65 suspect images (Pablo flagged these)
    README.md                 <-- Documents the dataset structure
    CA_Preprocessed/          <-- Output from preprocess.py (numpy arrays ready for training)
      train_normal.npy        <-- (9099, 240, 320) float32 -- all training images
      test_normal.npy         <-- (33, 240, 320) float32
      test_fault.npy          <-- (452, 240, 320) float32 (after removing ambiguous)
      norm_stats.npy          <-- Saved mean and std for use in production
  docs/                       <-- Project documents
    Training Plan Refined.docx
    Convolutional Autoencoder.odt
    Dataset_Gap_Analysis.docx
    code_walkthrough.md       <-- This file
  venv/                       <-- Python environment (gitignored)
  .gitignore                  <-- Keeps images and large files out of GitHub
```

---

## preprocess.py -- line by line

This script's job: take raw images from `CA_Training_Data/` and produce clean, standardized numpy arrays in `CA_Preprocessed/` that we can feed directly to the autoencoder.

### Imports and config (lines 22-48)

```python
import os
import numpy as np
from PIL import Image
from pathlib import Path

INPUT_DIR = Path("CA_Training_Data")
OUTPUT_DIR = Path("CA_Preprocessed")
TARGET_SIZE = (320, 240)  # width, height
```

- **numpy** -- the main library for working with arrays of numbers (our images become arrays)
- **PIL (Pillow)** -- library for opening and resizing image files
- **TARGET_SIZE = (320, 240)** -- every image gets resized to 320 pixels wide, 240 pixels tall. We chose this because it's the native resolution of the motor images (our primary target). The solar panel images (originally 640x512) get downsampled to match.

The three folder lists (`TRAIN_FOLDERS`, `TEST_NORMAL_FOLDERS`, `TEST_FAULT_FOLDERS`) define which subfolders to process. Notice `solar_modules` is NOT in the list -- those images are only 24x40 pixels, way too small to upscale to 320x240 without creating blurry garbage.

### load_image_raw() -- loading a single image (lines 51-81)

This function takes one image file and returns a standardized 320x240 grayscale array. It handles the fact that our images come from different cameras with different formats.

```python
def load_image_raw(filepath):
    img = Image.open(filepath)
    arr = np.array(img, dtype=np.float32)
```

Opens the image and converts it to a numpy array of floating point numbers. `dtype=np.float32` means each pixel becomes a decimal number (like 127.0) instead of an integer (like 127). We need decimals for the math that comes later.

```python
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
```

**Grayscale conversion.** Some images are RGB (3 channels -- red, green, blue), which makes `arr.ndim == 3` (height x width x 3). We average the 3 color channels into 1 grayscale channel. Why? The colors in thermal images are fake -- they're just a visual colormap applied to temperature data. The actual temperature info is the same across channels, so we only need one.

```python
    if arr.max() > 255:
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        else:
            arr = np.zeros_like(arr)
```

**Bit depth normalization.** This is the camera-agnostic part. Our dataset has two types of images:

1. **8-bit images** (PNG, BMP, JPG) -- pixel values range 0 to 255. These are "visual" thermal images where someone already applied a color palette.
2. **16-bit radiometric TIFFs** -- pixel values range from ~6000 to ~14000. These contain actual temperature readings from the camera sensor.

If the max pixel value is over 255, we know it's a 16-bit image. We scale it down to 0-255 using min-max normalization:
- Subtract the minimum (so the lowest value becomes 0)
- Divide by the range (so the highest value becomes 1)
- Multiply by 255 (so the range is 0-255)

This puts all images on the same scale regardless of what camera captured them.

```python
    img_resized = Image.fromarray(arr, mode="F")
    img_resized = img_resized.resize(TARGET_SIZE, Image.LANCZOS)
    arr_resized = np.array(img_resized, dtype=np.float32)
    return arr_resized
```

**Resizing.** Converts the array back to a PIL image (mode "F" = single-channel float), resizes to 320x240 using LANCZOS resampling (high-quality algorithm that avoids blurry artifacts), then converts back to a numpy array.

### load_folder() -- loading all images from a folder (lines 84-105)

Simple wrapper that loops through every file in a folder, calls `load_image_raw()` on each one, and collects the results into a list. Skips anything that fails to load (corrupted files, non-image files, etc.) and counts how many were skipped.

### main() -- the two-phase pipeline (lines 108-197)

This is where everything comes together. It runs in two phases:

**Phase 1: Compute global statistics from training data**

```python
train_arr = np.stack(train_images, axis=0)  # shape: (9099, 240, 320)
global_mean = train_arr.mean()
global_std = train_arr.std()
```

After loading all 9,099 training images, we stack them into one big 3D array and compute the mean and standard deviation across ALL pixels in ALL images. These two numbers (mean=171.97, std=45.68) capture the "average brightness" and "spread of brightness" of our training set.

Why only from training data? To avoid **data leakage** -- if we included test images in the stats, the model would have indirect information about the test set before ever seeing it.

**Phase 2: Z-score normalize everything**

```python
train_normalized = (train_arr - global_mean) / global_std
```

This is **z-score normalization**: for every pixel in every image, subtract the mean and divide by the standard deviation. After this:
- The training set has mean of exactly 0 and std of exactly 1
- Test images will have different stats (which is expected and informative)

Why z-score instead of just dividing by 255?
- Z-score preserves the **relative differences** between images. A hot motor that reads 200 and a cool motor that reads 50 will still be far apart after z-scoring.
- Per-image min-max normalization (our old approach) would have mapped EVERY image to 0-1, destroying the temperature differences between normal and fault images. That's exactly the signal the autoencoder needs.

```python
np.save(OUTPUT_DIR / "train_normal.npy", train_normalized.astype(np.float32))
```

Saves the normalized array as a `.npy` file -- numpy's binary format. Fast to load, compact, preserves exact values. The autoencoder training script will just do `np.load("train_normal.npy")` and it's ready to go.

The same normalization (using the SAME mean and std from training) is applied to test/normal and test/fault images. The stats are also saved to `norm_stats.npy` so we can use them in production.

---

## How preprocessing works in production

When deploying the model on a real camera feed, the same two-step process runs:

1. **Scale to 0-255**: If the camera outputs 16-bit radiometric data, min-max scale to 0-255. If it's already 8-bit, no conversion needed.
2. **Z-score normalize**: Subtract 171.97, divide by 45.68 (the saved training stats).
3. Feed to the autoencoder, get reconstruction error back.

This makes the system camera-agnostic -- any thermal camera works as long as we handle the bit depth conversion.

---

## Key numbers to remember

| What | Value |
|------|-------|
| Training images | 9,099 (all normal, no faults) |
| Test normal | 33 images |
| Test fault | 279 confirmed + 65 ambiguous (set aside) |
| Image size | 320 x 240 pixels, single channel grayscale |
| Normalization mean | 171.97 |
| Normalization std | 45.68 |
| Sources used | electric_motor, induction_motor, pv_om_inspection, pv_thermal_inspection |
| Sources dropped | solar_modules (too small at 24x40) |

---

## What the ambiguous images are

Pablo's visual quality check found 65 images in `test/fault/induction_motor/` (files 032.bmp through 096.bmp) that look wrong. They're almost entirely dark with no thermal signature -- mean pixel value around 34 with standard deviation under 5 (compare to normal images which have mean ~40 and std ~13).

These were moved to `test/fault/induction_motor_ambiguous/` so they don't corrupt our evaluation metrics. They may be mislabeled, or they could be images of a motor that was off/cold. We'll revisit them later.

---

## train_autoencoder.py -- line by line

This script builds and trains the autoencoder using PyTorch on your GPU (RTX 5070).

### The model architecture (ThermalAutoencoder class)

The autoencoder has two halves: an **encoder** that compresses the image down to a small representation, and a **decoder** that reconstructs it back to full size.

```
Input image (1, 240, 320)
    |
    v
[Encoder] -- compresses spatial information
  enc1: 2x Conv2d(1->32), ReLU, then MaxPool -> (32, 120, 160)
  enc2: 2x Conv2d(32->64), ReLU, then MaxPool -> (64, 60, 80)
  enc3: 2x Conv2d(64->128), ReLU, then MaxPool -> (128, 30, 40)
  enc4: 2x Conv2d(128->256), ReLU, then MaxPool -> (256, 15, 20)
    |
    v
[Bottleneck] -- (256, 15, 20) = 76,800 values
  The entire 320x240 image compressed to this tiny representation.
  This forces the model to learn the most important features.
    |
    v
[Decoder] -- reconstructs from compressed representation
  dec4: Upsample -> (256, 30, 40), 2x Conv2d(256->128)
  dec3: Upsample -> (128, 60, 80), 2x Conv2d(128->64)
  dec2: Upsample -> (64, 120, 160), 2x Conv2d(64->32)
  dec1: Upsample -> (32, 240, 320), 2x Conv2d(32->32)
  output_conv: Conv2d(32->1) -> (1, 240, 320)
    |
    v
Reconstructed image (1, 240, 320)
```

**Why 2 Conv2d layers per block?** One convolution captures simple patterns (edges, gradients). Stacking two lets the model learn more complex features (thermal hotspot shapes, cooling fin patterns) before downsampling.

**Why linear activation on the output?** Our data is z-score normalized, so pixel values can be negative. ReLU would clip negatives to zero, losing information.

**Why MaxPool for downsampling but Upsample for upsampling?** MaxPool picks the strongest feature in each 2x2 window (good for compression). Upsample just doubles each pixel (simple, no learned parameters). This asymmetry is a design choice -- some autoencoders use transposed convolutions for upsampling instead.

### The training loop (main function)

```python
x_train = torch.from_numpy(train_data[indices[n_val:]]).float()
x_val = torch.from_numpy(train_data[indices[:n_val]]).float()
```

Shuffles the 9,099 images randomly (with seed 42 for reproducibility), takes 90% for training (8,190) and 10% for validation (909). Converts numpy arrays to PyTorch tensors.

```python
train_loader = DataLoader(
    TensorDataset(x_train, x_train),  # input AND target are the same image
    batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
)
```

**Key detail:** `TensorDataset(x_train, x_train)` -- both the input and the target are the same image. The model's job is to output something as close to its input as possible. The loss measures how different the output is from the input.

`pin_memory=True` speeds up CPU-to-GPU data transfer.

The training loop itself:

```python
for epoch in range(EPOCHS):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(DEVICE)     # move batch to GPU
        output = model(batch_x)            # forward pass: input -> reconstruction
        loss = criterion(output, batch_y)  # MSE between reconstruction and original
        optimizer.zero_grad()              # clear previous gradients
        loss.backward()                    # compute gradients
        optimizer.step()                   # update weights
```

Each epoch loops through all 8,190 training images in batches of 32. For each batch:
1. Send images to GPU
2. Run through the autoencoder (forward pass)
3. Measure how different the reconstruction is from the original (MSE loss)
4. Compute how each weight contributed to the error (backward pass)
5. Adjust weights to reduce the error (optimizer step)

**Early stopping:** If validation loss doesn't improve for 10 epochs, training stops. This prevents overfitting -- the model could keep memorizing training images but that wouldn't help it generalize.

**ReduceLROnPlateau:** If val_loss stalls for 5 epochs, the learning rate is cut in half. Smaller steps help fine-tune when big steps keep overshooting.

### Training results

The model trained for 47 epochs (early stopped) on the RTX 5070:
- Loss went from 0.267 (epoch 1) to 0.0075 (epoch 37, best)
- Best model saved at `models/autoencoder_best.pth`
- 2,352,353 trainable parameters

---

## evaluate_autoencoder.py -- line by line

This script answers the question: can the autoencoder actually tell normal images from fault images?

### Computing reconstruction errors

```python
def compute_reconstruction_errors(model, data, batch_size=32):
    model.eval()  # turn off training-specific behavior
    ...
    mse = ((batch - output) ** 2).mean(dim=(1, 2, 3))
```

For each test image, we:
1. Feed it through the trained autoencoder
2. Compare the reconstruction to the original
3. Compute the mean squared error (MSE) per image

Normal images should have **low error** (the model learned to reconstruct these).
Fault images should have **high error** (the model has never seen faults, so it can't reconstruct them well).

### Finding the threshold

```python
def find_best_threshold(normal_errors, fault_errors):
```

We need a cutoff: "if reconstruction error is above X, flag it as a fault." This function tries 1,000 different thresholds and picks the one that maximizes the **F1 score** (the balance between precision and recall).

- **Precision:** Of all the images we flagged as faults, how many actually were? (Avoids false alarms)
- **Recall:** Of all the actual faults, how many did we catch? (Avoids missed faults)
- **F1:** The harmonic mean of precision and recall. Balances both concerns.

### AUROC

```python
def compute_auroc(normal_errors, fault_errors):
```

AUROC (Area Under the ROC Curve) measures how well the model separates normal from fault **across all possible thresholds**. It ranges from 0.0 to 1.0:
- 0.5 = random guessing (useless)
- 0.7-0.8 = decent
- 0.8-0.9 = good
- 0.9+ = excellent

### Evaluation results

```
Normal errors -- mean: 0.001043  (low, as expected)
Fault errors  -- mean: 0.004214  (4x higher than normal)

AUROC: 0.9092  (excellent separation)

Best threshold: 0.000602
Precision: 0.9516  (95% of flagged images are real faults)
Recall:    1.0000  (100% of faults caught, zero missed)
F1 Score:  0.9752

Confusion matrix:
  TP: 452  (faults correctly detected)
  FP:  23  (normals wrongly flagged as faults)
  FN:   0  (faults missed -- none!)
  TN:  10  (normals correctly passed)
```

The model catches every single fault image (452/452) with zero misses. The tradeoff is 23 false positives out of 33 normal test images -- it's being aggressive. This makes sense because our normal test set is very small (only 33 images) and comes from a slightly different distribution than the training set.

---

## What's next

1. **Generate heatmaps** -- visualize where the model sees anomalies in fault images (reconstruction error per pixel)
2. **Tune the threshold** -- the current threshold catches everything but flags some normals; we may want to adjust based on the use case (is it worse to miss a fault or to have a false alarm?)
3. **Data augmentation** -- implement the techniques Matias researched (horizontal flip, small rotation, Gaussian noise, cutout) to potentially improve the model further
4. **Test on more normal images** -- our normal test set is tiny (33 images); more data would give more reliable metrics
