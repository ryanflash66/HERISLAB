"""
Image-to-model-input transform (HER-125, HER-107).

Every array the autoencoders train and score on goes through the same steps,
defined once here:

  load_gray(path)                      open with Pillow, float32, RGB/RGBA averaged to one channel
  to_model_input(arr, repr, fit)       bring to the model's 0-255 range and resize to TARGET_SIZE
  normalize / denormalize              z-score with a dataset's {"mean", "std"} (norm_stats.npy)

Representations of ``to_model_input``:

  "scale_by_max"  the v2 / PV rule: min-max stretch to 0-255 only when the image
                  max is above 255 (16-bit radiometric TIFFs); 8-bit images pass
                  through unchanged. The branch is chosen from the pixel values,
                  not from file metadata, because every array under
                  data/CA_Preprocessed/{,pv,pv_om} was produced that way. It treats
                  16-bit and 8-bit sources differently, which is a source cue a
                  model can exploit (HER-107); kept so the stored arrays and the
                  bundles trained on them stay reproducible.
  "minmax"        always min-max stretch to 0-255 (India radiometric frames)
  "celsius"       clip to 0-120 degrees C, no rescale (India temperature frames)
  "equalize"      per-image histogram equalisation: every pixel is replaced by its
                  rank in the image, scaled to 0-255 (ties share their mean rank).
                  Only the order of the values is used, so a 16-bit radiometric
                  frame and an 8-bit palette render end up with the same, flat
                  value distribution, and no global brightness statistic can tell
                  the formats apart (HER-107). It does not make the values mean
                  the same thing: an RGB palette averaged to one channel is not
                  monotonic in temperature, so this is the fairest common mapping
                  for mixed formats, not a temperature mapping (HER-82 is that).

Fit (how the frame reaches TARGET_SIZE's 4:3 shape):

  "stretch"  resize straight to TARGET_SIZE, changing the aspect ratio
             (640x512 is 5:4, PVMD's 512x512 is 1:1). What every stored array used.
  "crop"     centre-crop to 4:3 first, then resize, so nothing is squashed. Both
             640x512 and 512x512 lose rows at the top and bottom (16 and 64 px).
             Cropping, not padding: a pad border would appear on one source only.

Order: crop, then the range step, then LANCZOS resize (float image, width x
height = TARGET_SIZE); "equalize" ranks after the resize so its output is flat
exactly. LANCZOS rings at sharp edges, so a resized "scale_by_max", "minmax" or
"celsius" array can overshoot its range by a little (a 16-bit frame slightly
outside 0-255, a Celsius frame slightly outside 0-120); the stored
CA_Preprocessed arrays carry that overshoot, and with fit="stretch" the order is
the original one, so they can be reproduced exactly. Nothing here reads the data
layout or the stats files; callers pass the stats dict.
"""
import numpy as np
from PIL import Image

TARGET_SIZE = (320, 240)   # width, height; the only declaration in the repo

REPRESENTATIONS = ("scale_by_max", "minmax", "celsius", "equalize")
FITS = ("stretch", "crop")
CELSIUS_RANGE = (0.0, 120.0)


def load_gray(path):
    """Open an image as a float32 2-D array; RGB/RGBA channels are averaged."""
    arr = np.array(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr


def crop_to_aspect(arr, size=TARGET_SIZE):
    """Centre-crop a 2-D array to the width:height ratio of `size` (width, height); no resampling."""
    h, w = arr.shape
    tw, th = size
    if w * th > h * tw:                       # too wide: drop columns
        nw = round(h * tw / th)
        x0 = (w - nw) // 2
        return arr[:, x0:x0 + nw]
    if w * th < h * tw:                       # too tall: drop rows
        nh = round(w * th / tw)
        y0 = (h - nh) // 2
        return arr[y0:y0 + nh, :]
    return arr


def _minmax_0_255(arr):
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        return (arr - lo) / (hi - lo) * 255.0
    return np.zeros_like(arr)


def _equalize_0_255(arr):
    """Rank of each value in the image, 0-255; equal values share their mean rank."""
    flat = arr.ravel()
    if flat.size < 2:
        return np.zeros_like(arr)
    _, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
    mean_rank = np.cumsum(counts) - (counts + 1) / 2.0          # 0-based
    return (mean_rank[inv.ravel()] / (flat.size - 1) * 255.0).reshape(arr.shape).astype(np.float32)


def to_model_input(arr, representation="scale_by_max", fit="stretch"):
    """2-D array -> float32 (TARGET_SIZE height, width) in the model's input range."""
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {representation!r}; expected one of {REPRESENTATIONS}")
    if fit not in FITS:
        raise ValueError(f"unknown fit {fit!r}; expected one of {FITS}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"to_model_input expects a 2-D array, got shape {arr.shape}")
    if fit == "crop":
        arr = crop_to_aspect(arr)
    if representation == "scale_by_max":
        if arr.max() > 255:
            arr = _minmax_0_255(arr)
    elif representation == "minmax":
        arr = _minmax_0_255(arr)
    elif representation == "celsius":
        arr = np.clip(arr, *CELSIUS_RANGE)
    img = Image.fromarray(np.ascontiguousarray(arr, dtype=np.float32)).resize(TARGET_SIZE, Image.LANCZOS)
    out = np.asarray(img, dtype=np.float32)
    if representation == "equalize":
        out = _equalize_0_255(out)
    return out


def normalize(arr, stats):
    """z-score with a dataset's train statistics: (arr - mean) / std."""
    return (np.asarray(arr, dtype=np.float32) - float(stats["mean"])) / float(stats["std"])


def denormalize(arr, stats):
    """Undo ``normalize``: arr * std + mean, back to the 0-255 model-input range."""
    return np.asarray(arr, dtype=np.float32) * float(stats["std"]) + float(stats["mean"])
