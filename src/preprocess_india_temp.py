"""
Preprocess the India split on a FIXED temperature scale (HER-82). Oct 26.

Difference from preprocess_india.py: no per-frame stretching. Every frame is
converted to degrees C and kept on ONE scale, so a hot spot stays hot and the
rest of the frame is not squashed. Decided before running:

  1. raw 16-bit counts -> degrees C (hlab.rjpg, Planck constants from the file)
  2. clip to [0, 120] C (guards against broken pixels; panels run 30 to 90 C)
     and resize to 320x240 (LANCZOS): hlab.transform.to_model_input(t, "celsius")
                                                        -> stored as degrees C
  3. z-score with ONE mean/std from the TRAIN split  (done by hlab.datasets at
     load time; the stored .npy files hold degrees C so anything can read real temperatures)

The 17 certified frames from the FLIR Vue Pro camera have unverified temperature
calibration and are left out of cross_site (120 frames remain).

Outputs (gitignored): data/CA_Preprocessed/india_temp/
  {train,val,test,cross_site}.npy   float32 (N, 240, 320), degrees C
  {split}_meta.csv                  rel_path, label, plot, flight, t_median, t_p99, t_max
  norm_stats.npy                    {"mean", "std"} in degrees C, from train
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hlab.paths import MANIFESTS, PREPROCESSED, ROOT  # noqa: E402
from hlab.rjpg import raw_to_celsius, read_rjpg  # noqa: E402
from hlab.transform import to_model_input  # noqa: E402

SPLIT_CSV = MANIFESTS / "india_plot_split.csv"
OUT = PREPROCESSED / "india_temp"
SPLITS = ["train", "val", "test", "cross_site"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(SPLIT_CSV, encoding="utf-8")))
    arrays, metas = {s: [] for s in SPLITS}, {s: [] for s in SPLITS}
    t0, n, skipped = time.time(), 0, 0
    for r in rows:
        s = r["split"]
        if s not in SPLITS:
            continue
        if r["camera"].startswith("FLIR"):
            skipped += 1; continue
        raw, calib = read_rjpg(ROOT / r["rel_path"])
        t = raw_to_celsius(raw, calib)
        arrays[s].append(to_model_input(t, "celsius"))
        metas[s].append({"rel_path": r["rel_path"], "label": r["label"], "plot": r["plot"],
                         "flight": r["flight"], "t_median": round(float(np.median(t)), 2),
                         "t_p99": round(float(np.percentile(t, 99)), 2), "t_max": round(float(t.max()), 2)})
        n += 1
        if n % 1000 == 0:
            print(f"{n} frames, {time.time() - t0:.0f}s", flush=True)
    train = np.stack(arrays["train"])
    mean, std = float(train.mean()), float(train.std())
    np.save(OUT / "norm_stats.npy", {"mean": mean, "std": std})
    print(f"train mean {mean:.2f} C, std {std:.2f} C  (skipped {skipped} FLIR Vue Pro frames)", flush=True)
    for s in SPLITS:
        x = np.stack(arrays[s]).astype(np.float32)
        np.save(OUT / f"{s}.npy", x)
        with open(OUT / f"{s}_meta.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(metas[s][0].keys())); w.writeheader(); w.writerows(metas[s])
        labels = [m["label"] for m in metas[s]]
        print(f"{s}: {x.shape}, healthy {labels.count('healthy')}, faulty {labels.count('faulty')}, "
              f"temp range {x.min():.1f} to {x.max():.1f} C, median {np.median(x):.1f} C", flush=True)
    print(f"DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
