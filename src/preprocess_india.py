"""
Preprocess the Thermal-Library India split (HER-82, milestone 06).

Reads docs/data_manifests/india_plot_split.csv (HER-18) and, for every frame:
  1. raw 16-bit counts from the R-JPG (src/flir_rjpg.py)
  2. min-max scale per frame to 0-255  (same rule as load_image_raw for 16-bit
     files; both classes come from one camera, so the 8-bit vs 16-bit gap
     should not separate healthy from faulty here. Per-frame stretching still
     discards absolute temperature.)
  3. resize to 320x240 (LANCZOS)
  4. z-score with mean/std from the TRAIN split only

Also records per-frame temperatures in degrees C (median, p99, max) for the
rule layer and baselines. Certified FLIR Vue Pro frames get NaN temperatures
until their calibration is verified.

Outputs (gitignored): data/CA_Preprocessed/india/
  {train,val,test,cross_site}.npy   float32 (N, 240, 320)
  {split}_meta.csv                  rel_path, label, plot, flight, t_median, t_p99, t_max
  norm_stats.npy                    {"mean", "std"} from train
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flir_rjpg import read_rjpg, raw_to_celsius  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPLIT_CSV = ROOT / "docs" / "data_manifests" / "india_plot_split.csv"
OUT = ROOT / "data" / "CA_Preprocessed" / "india"
TARGET_SIZE = (320, 240)
SPLITS = ["train", "val", "test", "cross_site"]


def to_model_input(raw):
    a = raw.astype(np.float32)
    lo, hi = a.min(), a.max()
    a = (a - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(a)
    img = Image.fromarray(a, mode="F").resize(TARGET_SIZE, Image.LANCZOS)
    return np.asarray(img, dtype=np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(SPLIT_CSV, encoding="utf-8")))
    arrays, metas = {s: [] for s in SPLITS}, {s: [] for s in SPLITS}
    t0, n = time.time(), 0
    for r in rows:
        s = r["split"]
        if s not in SPLITS:
            continue                      # train_faulty_unused
        raw, calib = read_rjpg(ROOT / r["rel_path"])
        arrays[s].append(to_model_input(raw))
        if r["camera"].startswith("FLIR"):
            tm = tp = tx = float("nan")
        else:
            t = raw_to_celsius(raw, calib)
            tm, tp, tx = float(np.median(t)), float(np.percentile(t, 99)), float(t.max())
        metas[s].append({"rel_path": r["rel_path"], "label": r["label"], "plot": r["plot"],
                         "flight": r["flight"], "t_median": round(tm, 2),
                         "t_p99": round(tp, 2), "t_max": round(tx, 2)})
        n += 1
        if n % 1000 == 0:
            print(f"{n} frames, {time.time() - t0:.0f}s", flush=True)

    train = np.stack(arrays["train"])
    mean, std = float(train.mean()), float(train.std())
    np.save(OUT / "norm_stats.npy", {"mean": mean, "std": std})
    print(f"train mean {mean:.3f}, std {std:.3f}", flush=True)
    for s in SPLITS:
        x = (np.stack(arrays[s]) - mean) / std
        np.save(OUT / f"{s}.npy", x.astype(np.float32))
        with open(OUT / f"{s}_meta.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(metas[s][0].keys()))
            w.writeheader(); w.writerows(metas[s])
        labels = [m["label"] for m in metas[s]]
        print(f"{s}: {x.shape}, healthy {labels.count('healthy')}, faulty {labels.count('faulty')}", flush=True)
    print(f"DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
