"""
Image-to-model-input transform (HER-125).

Every array the autoencoders train and score on goes through the same steps,
defined once here:

  load_gray(path)                 open with Pillow, float32, RGB/RGBA averaged to one channel
  to_model_input(arr, repr)       bring to the model's 0-255 range and resize to TARGET_SIZE
  normalize / denormalize         z-score with a dataset's {"mean", "std"} (norm_stats.npy)

Representations of ``to_model_input``:

  "scale_by_max"  the v2 / PV rule: min-max stretch to 0-255 only when the image
                  max is above 255 (16-bit radiometric TIFFs); 8-bit images pass
                  through unchanged. The branch is chosen from the pixel values,
                  not from file metadata, because every array under
                  data/CA_Preprocessed/ was produced that way. It is a source cue
                  a model can exploit (HER-107, HER-82); removing it is a modelling
                  decision, not part of this module.
  "minmax"        always min-max stretch to 0-255 (India radiometric frames)
  "celsius"       clip to 0-120 degrees C, no rescale (India temperature frames)

Resizing is LANCZOS on a float image, width x height = TARGET_SIZE, applied after
the range step in every representation. LANCZOS rings at sharp edges, so a
resized array can overshoot its range by a little (a 16-bit frame slightly
outside 0-255, a Celsius frame slightly outside 0-120); the stored
CA_Preprocessed arrays carry that overshoot, and the order is kept so they can
be reproduced exactly. Nothing here reads the data layout or the stats files;
callers pass the stats dict.
"""
import numpy as np
from PIL import Image

TARGET_SIZE = (320, 240)   # width, height; the only declaration in the repo

REPRESENTATIONS = ("scale_by_max", "minmax", "celsius")
CELSIUS_RANGE = (0.0, 120.0)


def load_gray(path):
    """Open an image as a float32 2-D array; RGB/RGBA channels are averaged."""
    arr = np.array(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr


def _minmax_0_255(arr):
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        return (arr - lo) / (hi - lo) * 255.0
    return np.zeros_like(arr)


def to_model_input(arr, representation="scale_by_max"):
    """2-D array -> float32 (TARGET_SIZE height, width) in the model's input range."""
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {representation!r}; expected one of {REPRESENTATIONS}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"to_model_input expects a 2-D array, got shape {arr.shape}")
    if representation == "scale_by_max":
        if arr.max() > 255:
            arr = _minmax_0_255(arr)
    elif representation == "minmax":
        arr = _minmax_0_255(arr)
    else:  # celsius
        arr = np.clip(arr, *CELSIUS_RANGE)
    img = Image.fromarray(np.ascontiguousarray(arr, dtype=np.float32)).resize(TARGET_SIZE, Image.LANCZOS)
    return np.asarray(img, dtype=np.float32)


def normalize(arr, stats):
    """z-score with a dataset's train statistics: (arr - mean) / std."""
    return (np.asarray(arr, dtype=np.float32) - float(stats["mean"])) / float(stats["std"])


def denormalize(arr, stats):
    """Undo ``normalize``: arr * std + mean, back to the 0-255 model-input range."""
    return np.asarray(arr, dtype=np.float32) * float(stats["std"]) + float(stats["mean"])
